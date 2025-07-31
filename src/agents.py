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
        self.p = 0.5  # Default p
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
    
    def finish_episode(self):
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
    
    Makes stochastic decisions to change variables based on a fixed p.
    """
    
    def __init__(self, agent_id: int, domain_size: int, p: float):
        """
        Initialize DSA agent.
        
        Args:
            agent_id: Unique identifier for this agent
            domain_size: Size of the agent's variable domain
            p: Probability of changing variable when improvement is possible
        """
        super().__init__(agent_id, domain_size)
        self.p = p
        self.p = p  # Backward compatibility
    
    def compute(self, messages: List[Msg]) -> None:
        """Compute DSA decision with fixed p."""
        current_cost = self.calculate_local_cost(messages)
        best_variable, min_cost = self.find_best_assignment(messages)
        
        # Change variable if improvement is possible and p allows
        if min_cost < current_cost:
            if self.agent_random.random() < self.p:
                self.variable = best_variable
    
    def send_messages(self) -> None:
        """Send current variable value to all neighbors."""
        for neighbor_id in self.neighbor_agent_ids:
            message = Msg(self.id_, neighbor_id, self.variable)
            self.outbox.insert([message])


class LearnedPolicyAgent(Agent):
    """
    DSA agent that uses learned probabilities without further learning.
    
    Uses p values learned from previous DSA-RL training runs.
    """
    
    def __init__(
        self, 
        agent_id: int, 
        domain_size: int, 
        learned_probability: float, 
        policy_weights: Optional[np.ndarray] = None
    ):
        """
        Initialize agent with learned p policy.
        
        Args:
            agent_id: Unique identifier for this agent
            domain_size: Size of the agent's variable domain
            learned_probability: Probability learned from DSA-RL training
            policy_weights: Optional policy weights from original learning
        """
        super().__init__(agent_id, domain_size)
        self.p = learned_probability
        self.policy_weights = policy_weights
        self.is_learned_policy = True
        self.p = learned_probability  # Backward compatibility
    
    def compute(self, messages: List[Msg]) -> None:
        """DSA decision making with learned p (no learning)."""
        current_cost = self.calculate_local_cost(messages)
        best_variable, min_cost = self.find_best_assignment(messages)
        
        # Use learned p for decision making
        if min_cost < current_cost:
            if self.agent_random.random() < self.p:
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

    Learns optimal p policies through reinforcement learning
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
            initial_probability: Initial p value
            learning_rate: Learning rate for policy updates
            baseline_decay: Decay rate for baseline moving average
        """
        super().__init__(agent_id, domain_size)

        

        self.p = initial_probability
        self.learning_rate = learning_rate
        self.decay_factor = baseline_decay
        self.baseline = 0.0
        # Prevent probabilities from becoming too low (maintains exploration)
        self.min_probability = 0.05  # Minimum exploration p
        self.episode_data: List[Dict] = []
        self.current_local_cost = 0.0
        self.probability_history = []
        self.convergence_threshold = 0.01
        self.convergence_window = 10
        self.is_converged = False
        self.episodes_since_convergence = 0
        
        self.theta = math.log(initial_probability / (1 - initial_probability))




    def compute(self, messages: List[Msg]) -> None:
        """Compute decision with reinforcement learning."""


        # Make DSA decision using frozen p
        current_cost = self.calculate_local_cost(messages)
        best_variable, min_cost = self.find_best_assignment(messages)
        self.recent_costs.append(current_cost)

        did_flip = False
        if min_cost < current_cost:
            if self.agent_random.random() < self.p:
                self.variable = best_variable
                did_flip = True

        action = 1.0 if did_flip else 0.0
        gardient = action - self.p
        reward = self.calculate_recent_improvement()
        #

        self.episode_data.append({
            'gardient': gardient,
            'reward': reward
        })

    def send_messages(self) -> None:
        """Send current variable value to all neighbors."""
        for neighbor_id in self.neighbor_agent_ids:
            message = Msg(self.id_, neighbor_id, self.variable)
            self.outbox.insert([message])

    def compute_gt(self):
        g_t = np.zeros(len(self.episode_data), dtype=np.float32)

        for i,_ in enumerate(self.episode_data):
            cumulative = 0
            for j, entry in enumerate(self.episode_data[i:]):
                cumulative += entry['reward'] * self.decay_factor ** j

            g_t[i] = cumulative

        return g_t


    def finish_episode(self) -> None:
        """Update policy weights using episode-level REINFORCE with enhanced variance reduction."""
        # Compute raw advantage relative to baseline
        episode_reward = self.compute_gt()
        advantages= episode_reward - self.baseline
        for i,advantage in enumerate(advantages):
            self.theta += self.learning_rate * advantage *self.episode_data[i]['gardient']
            
        self.baseline = self.decay_factor * self.baseline + (1 - self.decay_factor) * episode_reward.mean()
        self.p = sigmoid(self.theta)  
       # Clear episode data (do NOT update p here - let start_new_episode handle it)
        if self.p < self.min_probability:
            self.p = self.min_probability

        self.episode_data.clear()



    def calculate_recent_improvement(self):
        """Calculate recent improvement based on last few local costs."""
        if len(self.recent_costs) < 2:
            return 0.0
        # Calculate improvement as percentage change
        last_cost = self.recent_costs[-1]
        previous_cost = self.recent_costs[-2]
        
        if previous_cost == 0:
            return 0.0
        
        return previous_cost - last_cost


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