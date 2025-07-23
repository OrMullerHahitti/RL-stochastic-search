from abc import ABC, abstractmethod
import random
from typing import Tuple
import math

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
        
        # REINFORCE parameters
        self.theta = math.log(p0 / (1 - p0))  # Initialize Theta  corresponding to input probability
        self.p = p0  # Starting probability
        self.learning_rate = learning_rate
        self.baseline_decay = baseline_decay
        self.baseline = 0.0  # Exponential moving average baseline
        
        # Episode data collection
        self.episode_data = []  # Store (gradient, reward) pairs during episode
        self.current_local_cost = 0

    # Compute the new variable value based on messages received
    def compute(self, msgs):
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


        # Decision making and gradient collection for REINFORCE
        did_flip = False
        if min_possible_local_cost < self.current_local_cost:
            # Update probability from current theta
            self.p = sigmoid(self.theta)
            
            # Make decision to flip or not
            if self.agent_random.random() < self.p:
                self.variable = best_local_variable
                did_flip = True
                
        # Calculate gradient of log policy
        # ∇θ log π = (1-p) if flipped, -p if not flipped
        if did_flip:
            gradient = 1 - self.p
        else:
            gradient = -self.p
            
        # Store gradient and potential for reward calculation later
        self.episode_data.append({
            'gradient': gradient,
            'did_flip': did_flip,
            'beginning_iteration_local_cost': self.current_local_cost,
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
            
    def finish_episode(self, episode_rewards):
        """Update theta using REINFORCE with baseline at the end of episode"""
        if not self.episode_data:
            return
            
        # Calculate total reward for this episode
        total_reward = sum(episode_rewards)
        
        # Update baseline using exponential moving average
        self.baseline = update_exponential_moving_average(
            self.baseline, total_reward, self.baseline_decay
        )
        
        # Calculate advantage
        advantage = compute_advantage(total_reward, self.baseline)
        
        # Update theta using REINFORCE gradient ascent
        # θ ← θ + α * ∇log π * advantage
        total_gradient = sum(data['gradient'] for data in self.episode_data)
        self.theta += self.learning_rate * total_gradient * advantage
        
        # Clear episode data for next episode
        self.episode_data = []
        
        # Update probability for next episode
        self.p = sigmoid(self.theta)
    
    def set_p(self):
        """Update probability from current theta"""
        self.p = sigmoid(self.theta)





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