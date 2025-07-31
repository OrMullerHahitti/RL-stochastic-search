"""
DCOP Agent Implementations

This module contains all agent implementations for different DCOP algorithms:
- Standard DSA agents
- Reinforcement Learning agents (DSA-RL)
- MGM agents
- Agents using learned policies
"""

import math
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

from .global_map import Msg
from .utils import compute_advantage, update_exponential_moving_average, sigmoid


# =============================================================================
# ABSTRACT BASE AGENT
# =============================================================================

class Agent(ABC):
    """
    Abstract base class for all DCOP agents.
    
    Provides common functionality for variable management, message passing,
    and constraint handling that all agent types can build upon.
    """
    
    def __init__(self, agent_id: int, domain_size: int):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            domain_size: Size of the agent's variable domain
        """
        # Agent identification and timing
        self.id_ = agent_id
        self.global_clock = 0
        self.local_clock = 0
        
        # Variable and domain management
        self.agent_random = random.Random(self.id_)
        self.variable = self.agent_random.randint(1, domain_size)
        self.domain = list(range(1, domain_size + 1))
        
        # Constraint and neighbor management
        self.neighbor_agent_ids: List[int] = []
        self.constraint_tables: Dict[int, Dict] = {}
        
        # Message passing infrastructure
        self.inbox = None
        self.outbox = None
        
        # Initialize attributes that may be checked dynamically (for backward compatibility)
        self.neighborhood_context = None  # Will be set by contextual agents
        self.previous_neighbor_values = {}  # For tracking neighbor activity
        self.recent_costs = []  # For tracking recent costs
        self.neighbor_changes = []  # For tracking neighbor activity
        self.policy_weights = None  # For learning agents
        self.baseline = 0.0  # For learning agents
        self.probability = 0.5  # Default probability
        self.current_features = None  # For feature-based agents
        self.feature_running_stats = None  # For feature statistics
        self.learning_rate = 0.01  # Default learning rate
        self.violation_count = 0  # For constraint violation tracking
        self.episode_data = []  # For episode tracking in learning agents
    
    def set_neighbors(self, constraint_relations: List) -> None:
        """
        Set up neighbor relationships and constraint tables.
        
        Args:
            constraint_relations: List of ConstraintRelation objects involving this agent
        """
        for relation in constraint_relations:
            other_agent = relation.get_other_agent(self)
            self.neighbor_agent_ids.append(other_agent)
            self.constraint_tables[other_agent] = relation.cost_table
        
        # Backward compatibility
        # self.neighbors_agents_id = self.neighbor_agent_ids
        # self.constraints = self.constraint_tables
    
    def calculate_local_cost(self, messages: List[Msg]) -> float:
        """
        Calculate local constraint cost based on received messages.
        
        Args:
            messages: Messages from neighboring agents
            
        Returns:
            Total local constraint cost
        """
        total_cost = 0.0
        
        for msg in messages:
            other_agent_id = msg.sender
            other_variable = msg.information
            
            constraint_table = self.constraint_tables[other_agent_id]
            
            # Create key with consistent ordering (lower ID first)
            if self.id_ < other_agent_id:
                key = ((str(self.id_), self.variable), (str(other_agent_id), other_variable))
            else:
                key = ((str(other_agent_id), other_variable), (str(self.id_), self.variable))
            
            cost = constraint_table[key] if key in constraint_table else 0.0
            total_cost += cost
        
        return total_cost
    
    # Backward compatibility method
    #TODO: Remove in future versions
    def calc_local_cost(self, msgs):
        """Backward compatibility wrapper for calculate_local_cost."""
        return self.calculate_local_cost(msgs)
    
    def find_best_assignment(self, messages: List[Msg]) -> Tuple[int, float]:
        """
        Find the variable assignment that minimizes local cost.
        
        Args:
            messages: Messages from neighboring agents
            
        Returns:
            Tuple of (best_variable, minimum_cost)
        """
        current_cost = self.calculate_local_cost(messages)
        best_variable = self.variable
        min_cost = current_cost
        
        # Try all possible variable values
        for candidate_variable in self.domain:
            # Temporarily set variable to calculate cost
            original_variable = self.variable
            self.variable = candidate_variable
            
            candidate_cost = self.calculate_local_cost(messages)
            
            if candidate_cost < min_cost:
                min_cost = candidate_cost
                best_variable = candidate_variable
            
            # Restore original variable
            self.variable = original_variable
        
        return best_variable, min_cost
    
    def initialize(self) -> None:
        """Initialize agent by sending initial messages."""
        self.send_messages()
    
    def execute_iteration(self, global_clock: int) -> None:
        """
        Execute one iteration of the agent's algorithm.
        
        Args:
            global_clock: Current global time step
        """
        self.global_clock = global_clock
        messages = self.inbox.extract()
        
        if messages:
            self.local_clock += 1
            self.compute(messages)
            self.send_messages()
    
    def execute_iteration_with_context(self, global_clock: int, global_context: Optional[Dict] = None) -> None:
        """
        Execute iteration with global context (for contextual learning agents).
        
        Args:
            global_clock: Current global time step
            global_context: Global problem context for contextual decision making
        """
        self.global_clock = global_clock
        messages = self.inbox.extract()
        
        if messages:
            self.local_clock += 1
            
            if self.is_contextual_agent():
                self.compute_with_context(messages, global_context)
            else:
                self.compute(messages)
            
            self.send_messages()
    
    # Base methods that can be overridden by subclasses (for backward compatibility)
    def is_contextual_agent(self):
        """Check if this agent supports contextual computation."""
        return self.neighborhood_context is not None
    
    def compute_with_context(self, msgs, global_context):
        """Default implementation for contextual computation - override in contextual agents."""
        return self.compute(msgs)
    
    def get_did_flip(self):
        """Default implementation - override in learning agents."""
        return False
    
    def get_local_gain(self):
        """Default implementation - override in learning agents."""
        return 0
    
    def count_violations(self, msgs=None):
        """Default implementation - override in specific agents."""
        return 0
    
    def finish_episode(self, episode_rewards=None):
        """Default implementation - override in learning agents."""
        pass
    
    # Backward compatibility methods
    def send_msgs(self):
        """Backward compatibility wrapper for send_messages."""
        return self.send_messages()
    
    @abstractmethod
    def compute(self, messages: List[Msg]) -> None:
        """
        Compute new variable assignment based on received messages.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def send_messages(self) -> None:
        """
        Send messages to neighboring agents.
        Must be implemented by subclasses.
        """
        pass


# =============================================================================
# DSA AGENTS
# =============================================================================

class DSAAgent(Agent):
    """
    Standard DSA (Distributed Stochastic Algorithm) agent.
    
    Makes stochastic decisions to change variables based on a fixed probability.
    """
    
    def __init__(self, agent_id: int, domain_size: int, probability: float):
        """
        Initialize DSA agent.
        
        Args:
            agent_id: Unique identifier for this agent
            domain_size: Size of the agent's variable domain
            probability: Probability of changing variable when improvement is possible
        """
        super().__init__(agent_id, domain_size)
        self.probability = probability
        self.p = probability  # Backward compatibility
    
    def compute(self, messages: List[Msg]) -> None:
        """Compute DSA decision with fixed probability."""
        current_cost = self.calculate_local_cost(messages)
        best_variable, min_cost = self.find_best_assignment(messages)
        
        # Change variable if improvement is possible and probability allows
        if min_cost < current_cost:
            if self.agent_random.random() < self.probability:
                self.variable = best_variable
    
    def send_messages(self) -> None:
        """Send current variable value to all neighbors."""
        for neighbor_id in self.neighbor_agent_ids:
            message = Msg(self.id_, neighbor_id, self.variable)
            self.outbox.insert([message])


class LearnedPolicyAgent(Agent):
    """
    DSA agent that uses learned probabilities without further learning.
    
    Uses probability values learned from previous DSA-RL training runs.
    """
    
    def __init__(
        self, 
        agent_id: int, 
        domain_size: int, 
        learned_probability: float, 
        policy_weights: Optional[np.ndarray] = None
    ):
        """
        Initialize agent with learned probability policy.
        
        Args:
            agent_id: Unique identifier for this agent
            domain_size: Size of the agent's variable domain
            learned_probability: Probability learned from DSA-RL training
            policy_weights: Optional policy weights from original learning
        """
        super().__init__(agent_id, domain_size)
        self.probability = learned_probability
        self.policy_weights = policy_weights
        self.is_learned_policy = True
        self.p = learned_probability  # Backward compatibility
    
    def compute(self, messages: List[Msg]) -> None:
        """DSA decision making with learned probability (no learning)."""
        current_cost = self.calculate_local_cost(messages)
        best_variable, min_cost = self.find_best_assignment(messages)
        
        # Use learned probability for decision making
        if min_cost < current_cost:
            if self.agent_random.random() < self.probability:
                self.variable = best_variable
    
    def send_messages(self) -> None:
        """Send current variable value to all neighbors."""
        for neighbor_id in self.neighbor_agent_ids:
            message = Msg(self.id_, neighbor_id, self.variable)
            self.outbox.insert([message])


# =============================================================================
# REINFORCEMENT LEARNING AGENT
# =============================================================================

class ReinforcementLearningAgent(Agent):
    """
    DSA agent with REINFORCE learning and actor-critic methods.

    Learns optimal probability policies through reinforcement learning
    using local features and global context.
    """

    def __init__(
        self,
        agent_id: int,
        domain_size: int,
        initial_probability: float = 0.9,
        learning_rate: float = 0.01,
        baseline_decay: float = 0.9
    ):
        """
        Initialize reinforcement learning agent.

        Args:
            agent_id: Unique identifier for this agent
            domain_size: Size of the agent's variable domain
            initial_probability: Initial probability value
            learning_rate: Learning rate for policy updates
            baseline_decay: Decay rate for baseline moving average
        """
        super().__init__(agent_id, domain_size)

        # Linear Actor-Critic parameters
        self.num_features = 6
        self.policy_weights = np.random.normal(0, 0.1, self.num_features + 1)  # +1 for bias

        # Initialize bias for desired initial probability
        logit_p0 = math.log(initial_probability / (1 - initial_probability))
        self.policy_weights[-1] = max(-5.0, min(5.0, logit_p0))

        self.probability = initial_probability
        self.learning_rate = learning_rate
        self.baseline_decay = baseline_decay
        self.baseline = 0.0
        self.episode_features = {}

        # Episode data collection for learning
        self.episode_data: List[Dict] = []
        self.current_local_cost = 0.0

        # Feature tracking and normalization
        self.previous_local_cost = 0.0
        self.recent_costs: List[float] = []
        self.neighbor_changes: List[int] = []
        self.violation_count = 0
        self.current_features = None

        # Adaptive feature normalization statistics
        self.feature_running_stats = {
            'violations': {'sum': 0, 'sum_sq': 0, 'count': 0, 'mean': 0, 'std': 1},
            'improvement': {'sum': 0, 'sum_sq': 0, 'count': 0, 'mean': 0, 'std': 1},
            'cost': {'sum': 0, 'sum_sq': 0, 'count': 0, 'mean': 0, 'std': 1},
            'gain': {'sum': 0, 'sum_sq': 0, 'count': 0, 'mean': 0, 'std': 1},
            'activity': {'sum': 0, 'sum_sq': 0, 'count': 0, 'mean': 0, 'std': 1}
        }

        # Contextual learning parameters
        self.neighborhood_context = {
            'neighbor_priorities': {},
            'local_conflict_density': 0.0,
            'neighborhood_stability': 0.0
        }
        self.global_context_weight = 0.1

        # Backward compatibility attributes
        self.p = initial_probability

    def update_running_stats(self, feature_name: str, value: float) -> None:
        """Update running statistics for feature normalization."""
        stats = self.feature_running_stats[feature_name]
        stats['sum'] += value
        stats['sum_sq'] += value * value
        stats['count'] += 1

        # Update mean and standard deviation
        stats['mean'] = stats['sum'] / stats['count']
        if stats['count'] > 1:
            variance = (stats['sum_sq'] / stats['count']) - (stats['mean'] * stats['mean'])
            stats['std'] = max(np.sqrt(max(variance, 0)), 0.01)  # Prevent zero std

    def normalize_with_running_stats(self, feature_name: str, value: float) -> float:
        """Normalize feature using running statistics."""
        stats = self.feature_running_stats[feature_name]
        if stats['count'] < 5:  # Not enough data for reliable stats
            return np.tanh(value / 100.0)  # Fallback normalization

        # Z-score normalization with tanh squashing
        normalized = (value - stats['mean']) / stats['std']
        return np.tanh(normalized)

    def extract_local_features(self, messages: List[Msg], current_iteration: int, max_iterations: int) -> np.ndarray:
        """Extract local state features for policy decision making."""
        # Update cost tracking
        self.previous_local_cost = self.current_local_cost
        self.current_local_cost = self.calculate_local_cost(messages)
        self.recent_costs.append(self.current_local_cost)

        if len(self.recent_costs) > 10:
            self.recent_costs.pop(0)

        # Feature 1: Constraint violations
        violations = self.count_violations(messages)

        # Feature 2: Normalized time
        time_normalized = current_iteration / max_iterations if max_iterations > 0 else 0.0

        # Feature 3: Recent improvement
        recent_improvement = self.calculate_recent_improvement()

        # Feature 4: Current local cost
        local_cost = self.current_local_cost

        # Feature 5: Best potential gain
        best_gain = self.calculate_best_potential_gain(messages)

        # Feature 6: Neighbor activity
        neighbor_activity = self.track_neighbor_activity(messages)

        # Update and normalize features
        self.update_running_stats('violations', violations)
        self.update_running_stats('improvement', recent_improvement)
        self.update_running_stats('cost', local_cost)
        self.update_running_stats('gain', best_gain)
        self.update_running_stats('activity', neighbor_activity)

        # Create normalized feature vector
        features = np.array([
            self.normalize_with_running_stats('violations', violations),
            time_normalized,
            self.normalize_with_running_stats('improvement', recent_improvement),
            self.normalize_with_running_stats('cost', local_cost),
            self.normalize_with_running_stats('gain', best_gain),
            self.normalize_with_running_stats('activity', neighbor_activity)
        ], dtype=np.float32)

        return features

    def count_violations(self, messages: List[Msg]) -> int:
        """Count constraint violations (same color as neighbors)."""
        violations = 0
        for msg in messages:
            if self.variable == msg.information:  # Same color = violation
                violations += 1

        self.violation_count = violations
        return violations

    def calculate_recent_improvement(self) -> float:
        """Calculate recent cost improvement trend."""
        if len(self.recent_costs) < 2:
            return 0.0
        return self.recent_costs[-2] - self.recent_costs[-1]

    def calculate_best_potential_gain(self, messages: List[Msg]) -> float:
        """Calculate potential gain from best possible move."""
        if not messages:
            return 0.0

        current_cost = self.calculate_local_cost(messages)
        _, min_cost = self.find_best_assignment(messages)

        return max(0.0, current_cost - min_cost)

    def track_neighbor_activity(self, messages: List[Msg]) -> float:
        """Track how many neighbors have changed recently."""
        current_neighbor_values = {msg.sender: msg.information for msg in messages}
        changed_neighbors = 0

        if self.previous_neighbor_values:
            for agent_id, current_value in current_neighbor_values.items():
                if agent_id in self.previous_neighbor_values:
                    if current_value != self.previous_neighbor_values[agent_id]:
                        changed_neighbors += 1

        self.previous_neighbor_values = current_neighbor_values.copy()
        self.neighbor_changes.append(changed_neighbors)

        if len(self.neighbor_changes) > 5:
            self.neighbor_changes.pop(0)

        return np.mean(self.neighbor_changes) if self.neighbor_changes else 0.0

    def compute_linear_policy_probability(self, features: np.ndarray) -> float:
        """Compute probability using linear policy with contextual adjustments."""
        features_with_bias = np.append(features, 1.0)
        linear_output = np.dot(self.policy_weights, features_with_bias)
        base_probability = sigmoid(linear_output)

        # Apply contextual adjustments
        return self.adjust_probability(base_probability)

    def adjust_probability(self, base_probability: float) -> float:
        """Apply contextual adjustments to base probability."""
        adjusted_prob = base_probability

        # Neighborhood conflict density adjustment
        conflict_density = self.neighborhood_context['local_conflict_density']
        if conflict_density > 0.7:
            adjusted_prob = min(adjusted_prob * 1.2, 0.95)  # More aggressive
        elif conflict_density < 0.3:
            adjusted_prob = max(adjusted_prob * 0.8, 0.05)  # More conservative

        return adjusted_prob

    def update_neighborhood_context(self, messages: List[Msg]) -> None:
        """Update contextual information about local neighborhood."""
        if messages:
            conflicts = sum(1 for msg in messages if msg.information == self.variable)
            self.neighborhood_context['local_conflict_density'] = conflicts / len(messages)

        # Update neighborhood stability
        if len(self.neighbor_changes) >= 3:
            recent_changes = self.neighbor_changes[-3:]
            avg_changes = np.mean(recent_changes)
            max_possible = len(self.neighbor_agent_ids)
            stability = 1.0 - (avg_changes / max(max_possible, 1)) #Fallback to number of neighbors == 0
            self.neighborhood_context['neighborhood_stability'] = stability

    def compute(self, messages: List[Msg]) -> None:
        """Compute decision with reinforcement learning."""
        # Extract features and update probability
        max_iterations = 100  # Default fallback
        features = self.extract_local_features(messages, self.global_clock, max_iterations)
        self.episode_features[self.local_clock] = features
        self.update_neighborhood_context(messages)

        self.current_features = features

        # Make DSA decision
        current_cost = self.calculate_local_cost(messages)
        best_variable, min_cost = self.find_best_assignment(messages)

        did_flip = False
        if min_cost < current_cost:
            if self.agent_random.random() < self.probability:
                self.variable = best_variable
                did_flip = True

        self.episode_data.append({
            'did_flip': did_flip,
            'beginning_iteration_local_cost': current_cost,
            'features': features.copy()
        })

    def send_messages(self) -> None:
        """Send current variable value to all neighbors."""
        for neighbor_id in self.neighbor_agent_ids:
            message = Msg(self.id_, neighbor_id, self.variable)
            self.outbox.insert([message])

    def finish_episode(self, episode_rewards: Optional[List[float]] = None) -> None:
        """Update policy weights using episode-level REINFORCE."""

        # 1. Compute episode-level features (aggregate across episode)
        if self.episode_data:
            all_features = [data['features'] for data in self.episode_data]
            episode_features = np.mean(all_features, axis=0)  # Average features
        else:
            return

        # 2. Compute episode-level "action"
        total_flips = sum(data['did_flip'] for data in self.episode_data)
        flip_rate = total_flips / len(self.episode_data)  # Episode action = flip rate

        # 3. Compute episode-level advantage
        episode_return = sum(episode_rewards) if episode_rewards else 0.0
        advantage = episode_return - self.baseline

        # 4. Compute episode-level gradient
        # ∇θ log π = [features, 1] * (episode_action - episode_probability)
        features_with_bias = np.append(episode_features, 1.0)
        episode_gradient = features_with_bias * (flip_rate - self.episode_probability)

        # 5. Single weight update for entire episode
        gradient_norm = np.linalg.norm(episode_gradient)
        if gradient_norm > 2.0:
            episode_gradient = episode_gradient * (2.0 / gradient_norm)

        self.policy_weights += self.learning_rate * episode_gradient * advantage
        self.policy_weights = np.clip(self.policy_weights, -20.0, 20.0)

        # 6. Update baseline
        self.baseline = (self.baseline_decay * self.baseline +
                         (1 - self.baseline_decay) * episode_return)

        # 7. Compute probability for NEXT episode
        self.episode_probability = self.compute_linear_policy_probability(episode_features)

        # 8. Clear episode data
        self.episode_data = []

    def get_local_gain(self) -> float:
        """Calculate local gain from the last decision."""
        if len(self.episode_data) > 1:
            last_cost = self.episode_data[-1]["beginning_iteration_local_cost"]
            previous_cost = self.episode_data[-2]["beginning_iteration_local_cost"]
            return previous_cost - last_cost
        return 0

    def get_did_flip(self) -> bool:
        """Check if agent flipped in the last iteration."""
        if self.episode_data:
            return self.episode_data[-1]['did_flip']
        return False


# =============================================================================
# MGM AGENT  
# =============================================================================

class MGMAgent(Agent):
    """
    MGM (Maximum Gain Messages) algorithm agent.
    
    Uses a two-phase approach: calculate maximum gain, then coordinate
    with neighbors to avoid conflicts when multiple agents want to change.
    """
    
    def __init__(self, agent_id: int, domain_size: int):
        """
        Initialize MGM agent.
        
        Args:
            agent_id: Unique identifier for this agent
            domain_size: Size of the agent's variable domain
        """
        super().__init__(agent_id, domain_size)
        self.local_reduction = 0.0  # Maximum possible cost reduction
        self.potential_variable = self.variable  # Variable that achieves max reduction
        
        # Backward compatibility attributes
        self.lr = 0.0
        self.lr_potential_variable = self.variable
    
    def compute(self, messages: List[Msg]) -> None:
        """Compute MGM decision using two-phase protocol."""
        if self.global_clock % 2 == 1:
            # Odd iterations: Calculate maximum local reduction
            self._calculate_maximum_reduction(messages)
        else:
            # Even iterations: Coordinate with neighbors
            self._coordinate_change_decision(messages)
    
    def _calculate_maximum_reduction(self, messages: List[Msg]) -> None:
        """Calculate maximum possible cost reduction."""
        current_cost = self.calculate_local_cost(messages)
        best_variable, min_cost = self.find_best_assignment(messages)
        
        if min_cost < current_cost:
            self.local_reduction = current_cost - min_cost
            self.potential_variable = best_variable
            # Backward compatibility
            self.lr = self.local_reduction
            self.lr_potential_variable = self.potential_variable
        else:
            self.local_reduction = 0.0
            self.lr = 0.0
    
    def _coordinate_change_decision(self, messages: List[Msg]) -> None:
        """Coordinate with neighbors to decide if this agent should change."""
        should_change = True
        
        for msg in messages:
            neighbor_reduction = msg.information
            
            # Check if neighbor has greater reduction
            if neighbor_reduction > self.local_reduction:
                should_change = False
                self.local_reduction = 0.0
                self.lr = 0.0
                break
            
            # Tie-breaking: lower ID wins
            if neighbor_reduction == self.local_reduction and self.id_ > msg.sender:
                should_change = False
                self.local_reduction = 0.0
                self.lr = 0.0
                break
        
        # Change variable if this agent wins the coordination
        if should_change and self.local_reduction > 0:
            self.variable = self.potential_variable
            self.local_reduction = 0.0
            self.lr = 0.0
    
    def send_messages(self) -> None:
        """Send messages based on current phase."""
        if self.global_clock % 2 == 0:
            # Even iterations: Send current variable
            for neighbor_id in self.neighbor_agent_ids:
                message = Msg(self.id_, neighbor_id, self.variable)
                self.outbox.insert([message])
        else:
            # Odd iterations: Send local reduction
            for neighbor_id in self.neighbor_agent_ids:
                message = Msg(self.id_, neighbor_id, self.local_reduction)
                self.outbox.insert([message])


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Keep old class names for backward compatibility
DSA_Agent = DSAAgent
DSA_Agent_Learned = LearnedPolicyAgent
DsaAgentAdaptive = ReinforcementLearningAgent
MGM_Agent = MGMAgent