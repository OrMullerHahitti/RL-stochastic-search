from abc import ABC, abstractmethod
import random
from typing import Tuple
import math
import numpy as np

from .global_map import *
from .utils import  compute_advantage, update_exponential_moving_average,sigmoid



# Abstract base class for all agents
class Agent(ABC):

    def __init__(self,id_, d):
        self.global_clock = 0
        self.id_ = id_
        self.agent_random = random.Random(self.id_)
        self.variable = self.agent_random.randint(1, d)
        self.domain = []
        for i in range(1,d+1):  self.domain.append(i)  # Initialize domain
        self.neighbors_agents_id = []
        self.constraints={}
        self.inbox = None
        self.outbox = None
        self.local_clock = 0

    # Set neighbors for the agent
    def set_neighbors(self,neighbors):
        for n in neighbors:
            a_other = n.get_other_agent(self)
            self.neighbors_agents_id.append(a_other)
            self.constraints[a_other]=n.cost_table

    # Calculate the local cost based on messages received
    def calc_local_cost(self, msgs):
        local_cost = 0
        for msg in msgs:
            other_agent_id = msg.sender
            constraint = self.constraints[other_agent_id]
            if self.id_ < other_agent_id:
                cost = constraint[((str(self.id_), self.variable), (str(other_agent_id), msg.information))]
            else:
                cost = constraint[((str(other_agent_id), msg.information), (str(self.id_), self.variable))]
            local_cost += cost
        return local_cost

    # Initialize the agent (send initial messages)
    def initialize(self):
        self.send_msgs()

    # Execute an iteration of the algorithm
    def execute_iteration(self,global_clock):
        self.global_clock = global_clock
        msgs = self.inbox.extract()
        if len(msgs)!=0:
            self.local_clock = self.local_clock + 1
            self.compute(msgs)
            self.send_msgs()

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def compute(self, msgs): pass

    @abstractmethod
    def send_msgs(self): pass


# Agent class for the DSA algorithm with REINFORCE learning
class DsaAgentAdaptive(Agent):
    def __init__(self, id_, d, p0=0.7, learning_rate=0.01, baseline_decay=0.9):
        super().__init__(id_, d)
        
        # Linear Actor-Critic parameters  
        self.num_features = 6  # Number of local features
        # Initialize policy weights for linear model: p = sigmoid(w^T * features)
        # Start with very small random weights to prevent initial overflow
        self.policy_weights = np.random.normal(0, 0.01, self.num_features + 1)  # +1 for bias
        
        # Initialize bias more carefully to achieve p0, but clamp to prevent overflow
        logit_p0 = math.log(p0 / (1 - p0))
        self.policy_weights[-1] = max(-5.0, min(5.0, logit_p0))  # Clamp initial bias
        
        self.p = p0  # Starting probability
        self.learning_rate = learning_rate
        self.baseline_decay = baseline_decay
        self.baseline = 0.0  # Exponential moving average baseline
        
        # Episode data collection
        self.episode_data = []  # Store (gradient, reward) pairs during episode
        self.current_local_cost = 0
        
        # Feature tracking for actor-critic
        self.previous_local_cost = 0
        self.recent_costs = []  # Track recent local costs for trend analysis
        self.neighbor_changes = []  # Track recent neighbor activity
        self.violation_count = 0
        self.current_features = None  # Will be set during first compute() call

    def validate_policy_consistency(self, features, context="unknown"):
        """Validate that stored probability matches linear policy computation"""
        from .global_map import get_testing_parameters
        testing_params = get_testing_parameters()
        epsilon = testing_params["validation_epsilon"]
        
        expected_p = self.compute_linear_policy_probability(features)
        if abs(self.p - expected_p) > epsilon:
            error_msg = (f"Agent {self.id_} policy INCONSISTENCY at {context}: "
                        f"stored_p={self.p:.6f}, expected_p={expected_p:.6f}, "
                        f"error={abs(self.p - expected_p):.2e}")
            raise ValueError(error_msg)
        return True

    # Compute the new variable value based on messages received
    def compute(self, msgs):
        # Extract local features for linear policy
        # Use a reasonable max_iterations estimate (will be more accurate in DCOP context)
        max_iterations = 100  # Default fallback, can be improved with DCOP context
        features = self.extract_local_features(msgs, self.global_clock, max_iterations)
        
        # Update probability using linear policy: p = sigmoid(w^T * features)
        self.p = self.compute_linear_policy_probability(features)
        
        # Store features for learning (will be used in gradient calculation)
        self.current_features = features
        
        self.current_local_cost = self.calc_local_cost(msgs)
        min_possible_local_cost = self.current_local_cost  # Initialize minimum cost with current local cost
        best_local_variable = self.variable  # Initialize the best variable index with current variable index

        # Loop over all possible variables - try to minimize the estimated local cost for the choice of variable
        for variable in self.domain:
            current_variable_costs = 0
            for msg in msgs:
                other_agent_id= msg.sender
                constraint = self.constraints[other_agent_id]
                # The keys to the constraint dictionary are built in a way that the agent with the lower ID comes
                # first inside of it
                if self.id_<other_agent_id:
                    cost = constraint[((str(self.id_), variable), (str(other_agent_id), msg.information))]
                else:
                    cost = constraint[((str(other_agent_id), msg.information), (str(self.id_), variable))]
                current_variable_costs += cost
            # update min_possible_local_cost and best_local_variable in case a better local variable was found
            if current_variable_costs < min_possible_local_cost:
                min_possible_local_cost = current_variable_costs
                best_local_variable = variable

        # Decision making and gradient collection for linear policy
        did_flip = False
        if min_possible_local_cost < self.current_local_cost:
            # Make decision to flip or not (probability already updated above)
            if self.agent_random.random() < self.p:
                self.variable = best_local_variable
                did_flip = True
                
        # Calculate gradient of log policy for linear model
        # ∇w log π = features * (action - p) where action=1 if flipped, 0 if not
        features_with_bias = np.append(self.current_features, 1.0)
        if did_flip:
            # ∇w log π(flip|s) = features * (1 - p)
            gradient_weights = features_with_bias * (1 - self.p)
        else:
            # ∇w log π(no-flip|s) = features * (0 - p) = features * (-p)
            gradient_weights = features_with_bias * (-self.p)
            
        # Store gradient and episode data for reward calculation later
        self.episode_data.append({
            'gradient_weights': gradient_weights,
            'did_flip': did_flip,
            'beginning_iteration_local_cost': self.current_local_cost,
            'features': self.current_features.copy()
        })

    # Send messages to all neighbors
    def send_msgs(self):
        for n in self.neighbors_agents_id:
            message = Msg(self.id_, n, self.variable)
            self.outbox.insert([message])
    
    def get_local_gain(self):
        """Calculate local gain from the last decision"""
        if len(self.episode_data)>1:
            last_data,one_before_last = self.episode_data[-1]["beginning_iteration_local_cost"],self.episode_data[-2]["beginning_iteration_local_cost"]
            return last_data - one_before_last
        return 0
    
    def get_did_flip(self):
        """Check if agent flipped in the last iteration"""
        if self.episode_data:
            return self.episode_data[-1]['did_flip']
        return False
    
    def count_violations(self, msgs=None):
        """Count the number of constraint violations involving this agent"""
        if not msgs:
            return self.violation_count
        
        violations = 0
        for msg in msgs:
            other_agent_id = msg.sender
            other_variable = msg.information
            
            # Check if this creates a constraint violation
            if self.variable == other_variable:  # Same color = violation in graph coloring
                violations += 1
        
        self.violation_count = violations
        return violations
    
    def track_neighbor_activity(self, msgs):
        """Track how many neighbors have changed recently"""
        changed_neighbors = 0
        current_neighbor_values = {}
        
        for msg in msgs:
            current_neighbor_values[msg.sender] = msg.information
        
        # Count changes if we have previous data
        if hasattr(self, 'previous_neighbor_values'):
            for agent_id, current_value in current_neighbor_values.items():
                if agent_id in self.previous_neighbor_values:
                    if current_value != self.previous_neighbor_values[agent_id]:
                        changed_neighbors += 1
        
        # Store for next iteration
        self.previous_neighbor_values = current_neighbor_values.copy()
        
        # Track recent activity (sliding window)
        self.neighbor_changes.append(changed_neighbors)
        if len(self.neighbor_changes) > 5:  # Keep last 5 iterations
            self.neighbor_changes.pop(0)
        
        return changed_neighbors
    
    def calculate_recent_improvement(self):
        """Calculate recent improvement trend"""
        if len(self.recent_costs) < 2:
            return 0.0
        
        # Simple: difference from previous iteration
        return self.recent_costs[-2] - self.recent_costs[-1]
    
    def extract_local_features(self, msgs, current_iteration, max_iterations):
        """Extract local state features for actor-critic decision making"""
        
        # Update tracking information
        self.previous_local_cost = self.current_local_cost
        self.current_local_cost = self.calc_local_cost(msgs)
        
        # Update recent costs tracking
        self.recent_costs.append(self.current_local_cost)
        if len(self.recent_costs) > 10:  # Keep last 10 iterations
            self.recent_costs.pop(0)
        
        # Feature 1: v_i - number of violations (0 to max_neighbors)
        violations = self.count_violations(msgs)
        
        # Feature 2: t - normalized time (0 to 1)
        time_norm = current_iteration / max_iterations if max_iterations > 0 else 0.0
        
        # Feature 3: δ_i - recent improvement (can be negative)
        recent_improvement = self.calculate_recent_improvement()
        
        # Feature 4: local_cost - current local constraint cost
        local_cost = self.current_local_cost
        
        # Feature 5: best_gain - potential gain from best move
        best_gain = self.calculate_best_potential_gain(msgs)
        
        # Feature 6: neighbor_activity - how many neighbors changed recently
        neighbor_activity = self.track_neighbor_activity(msgs)
        recent_neighbor_activity = np.mean(self.neighbor_changes) if self.neighbor_changes else 0.0
        
        # Return feature vector
        features = np.array([
            violations,
            time_norm, 
            recent_improvement,
            local_cost,
            best_gain,
            recent_neighbor_activity
        ], dtype=np.float32)
        
        return features
    
    def calculate_best_potential_gain(self, msgs):
        """Calculate the potential gain from the best possible move"""
        if not msgs:
            return 0.0
        
        current_cost = self.calc_local_cost(msgs)
        min_cost = current_cost
        
        # Try all possible variable values
        for variable in self.domain:
            temp_cost = 0
            for msg in msgs:
                other_agent_id = msg.sender
                constraint = self.constraints[other_agent_id]
                
                # Calculate cost for this variable choice
                if self.id_ < other_agent_id:
                    cost = constraint[((str(self.id_), variable), (str(other_agent_id), msg.information))]
                else:
                    cost = constraint[((str(other_agent_id), msg.information), (str(self.id_), variable))]
                temp_cost += cost
            
            min_cost = min(min_cost, temp_cost)
        
        return max(0.0, current_cost - min_cost)  # Potential improvement
    
    def compute_linear_policy_probability(self, features):
        """Compute probability using linear policy: p = sigmoid(w^T * [features, 1])"""
        # Add bias term (1.0) to features
        features_with_bias = np.append(features, 1.0)
        
        # Linear combination: w^T * [features, bias]  
        linear_output = np.dot(self.policy_weights, features_with_bias)
        
        # Apply sigmoid to get probability
        probability = sigmoid(linear_output)
        
        return probability
            
    def finish_episode(self, episode_rewards):
        """Update policy weights using actor-critic advantages"""
        if not self.episode_data:
            return
            
        # episode_rewards now contains critic-based advantages (TD errors) per iteration
        # Ensure we have matching lengths
        min_length = min(len(self.episode_data), len(episode_rewards))
        
        # Apply policy gradient updates using critic-based advantages per iteration
        for i in range(min_length):
            data = self.episode_data[i]
            advantage = episode_rewards[i]  # Critic-based advantage for this iteration
            
            # Get gradient for this iteration
            gradient_weights = data['gradient_weights']
            
            # Apply gradient clipping per-iteration for stability
            max_gradient_norm = 1.0  # Smaller per-iteration clipping
            gradient_norm = np.linalg.norm(gradient_weights)
            if gradient_norm > max_gradient_norm:
                gradient_weights = gradient_weights * (max_gradient_norm / gradient_norm)
            
            # Update policy weights using critic-based advantage
            self.policy_weights += self.learning_rate * gradient_weights * advantage
            
            # Clamp policy weights to prevent numerical overflow in sigmoid
            self.policy_weights = np.clip(self.policy_weights, -10.0, 10.0)
        
        # Keep exponential moving average baseline for monitoring purposes
        total_advantage = sum(episode_rewards) if episode_rewards else 0
        self.baseline = update_exponential_moving_average(
            self.baseline, total_advantage, self.baseline_decay
        )
        
        # Clear episode data for next episode
        self.episode_data = []
    






# Simple DSA Agent without learning (for backward compatibility)
class DSA_Agent(Agent):
    def __init__(self, id_, d, p):
        super().__init__(id_, d)
        self.p = p  # Fixed probability threshold

    # Compute the new variable value based on messages received
    def compute(self, msgs):
        current_local_cost = self.calc_local_cost(msgs)
        min_possible_local_cost = current_local_cost  # Initialize minimum cost with current local cost
        best_local_variable = self.variable  # Initialize the best variable index with current variable index

        # Loop over all possible variables - try to minimize the estimated local cost for the choice of variable
        for variable in self.domain:
            current_variable_costs = 0
            for msg in msgs:
                other_agent_id= msg.sender
                constraint = self.constraints[other_agent_id]
                # The keys to the constraint dictionary are built in a way that the agent with the lower ID comes
                # first inside of it
                if self.id_<other_agent_id:
                    cost = constraint[((str(self.id_), variable), (str(other_agent_id), msg.information))]
                else:
                    cost = constraint[((str(other_agent_id), msg.information), (str(self.id_), variable))]
                current_variable_costs += cost
            # update min_possible_local_cost and best_local_variable in case a better local variable was found
            if current_variable_costs < min_possible_local_cost:
                min_possible_local_cost = current_variable_costs
                best_local_variable = variable

        # Change variable with probability p if a better min_possible_local_cost was found
        if min_possible_local_cost < current_local_cost:
            if self.agent_random.random()<self.p:
                self.variable = best_local_variable

    # Send messages to all neighbors
    def send_msgs(self):
        for n in self.neighbors_agents_id:
            message = Msg(self.id_, n, self.variable)
            self.outbox.insert([message])


# Agent class for the MGM algorithm
class MGM_Agent(Agent):
    def __init__(self, id_, d):
        super().__init__(id_, d)
        self.lr = 0  # Local reduction
        self.lr_potential_variable = self.variable  # Potential variable that leads to max local reduction


    def compute(self, msgs):
        # In odd iterations, the agent calculates its maximum local reduction and identifies the potential variable
        # that would lead to this maximum reduction in cost.
        if self.global_clock % 2 == 1:
            current_local_cost = self.calc_local_cost(msgs)  # Calculate the current local cost based on messages
            min_possible_local_cost = current_local_cost
            best_local_variable = self.variable

            # Loop over all possible variables in the agent's domain to find the one that minimizes the local cost
            for variable in self.domain:
                current_costs = 0  # Initialize the cost for the current variable
                # Calculate the sum of local costs if the agent adopts the current variable
                for msg in msgs:
                    other_agent_id= msg.sender
                    constraint = self.constraints[other_agent_id]
                    # The keys to the constraint dictionary are built in a way that the agent with the lower ID comes
                    # first inside of it
                    if self.id_<other_agent_id:
                        cost = constraint[((str(self.id_), variable), (str(other_agent_id), msg.information))]
                    else:
                        cost = constraint[((str(other_agent_id), msg.information), (str(self.id_), variable))]
                    current_costs += cost
                # Update the minimum possible local cost and the best variable if a better one is found
                if current_costs < min_possible_local_cost:
                    min_possible_local_cost = current_costs
                    best_local_variable = variable

            # If a lower local cost is found with a different variable, calculate the local reduction
            if min_possible_local_cost < current_local_cost:
                self.lr= current_local_cost- min_possible_local_cost  # Calculate the local reduction
                # Set the potential variable that achieves this reduction without assigning it to agent variable yet
                self.lr_potential_variable= best_local_variable

        # In even iterations, the agent checks if it has the greatest local reduction compared to its neighbors
        # If multiple agents have the same greatest local reduction,
        # the one with the lowest ID has higher priority to change its variable
        else:
            flag_greatest_local_lr=True
            for msg in msgs:
                neighbor_lr = msg.information
                if neighbor_lr > self.lr:
                    flag_greatest_local_lr = False # Another agent has a greater local reduction
                    self.lr = 0  # Reset the local reduction
                    break
                if neighbor_lr == self.lr and self.id_ > msg.sender:
                    flag_greatest_local_lr = False # Another agent has the same local reduction but a lower ID
                    self.lr = 0  # Reset the local reduction
                    break
            # If the agent has the greatest local reduction, it changes its variable to the potential variable
            if flag_greatest_local_lr and self.lr > 0:
                self.variable = self.lr_potential_variable
                self.lr = 0  # Reset the local reduction

    def send_msgs(self):
        # In even iterations, Send messages to all neighbors with the agent's current variable
        if self.global_clock % 2 == 0:
            for n in self.neighbors_agents_id:
                message = Msg(self.id_, n, self.variable)
                self.outbox.insert([message])
        # Send messages to all neighbors with the agent's greatest local reduction in odd iterations

        # In odd iterations, Send messages to all neighbors with agent's greatest local reduction
        else:
            for n in self.neighbors_agents_id:
                message = Msg(self.id_, n, self.lr)
                self.outbox.insert([message])