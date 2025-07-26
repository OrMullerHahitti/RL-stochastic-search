from .agents import *
from abc import ABC, abstractmethod
from .utils import distribute_rewards_proportionally
from .validation import validate_agent_mu_config, ensure_dictionary_keys
from .topology import SharedGraphTopology
from .global_map import Algorithm, get_testing_parameters, get_master_config
import random
import numpy as np
from typing import List, Dict, Optional, Any, Tuple

# Note: SharedGraphTopology is now imported from topology.py
# The updated version supports learning vs comparison modes

# Class to define neighbor relationships between agents in a DCOP
class Neighbors():
    def __init__(self, a1:Agent, a2:Agent,dcop_id, penalty_a1=None, penalty_a2=None):

        if a1.id_<a2.id_:
            self.a1 = a1
            self.a2 = a2
        else:
            self.a1 = a2
            self.a2 = a1

        self.dcop_id = dcop_id
        self.rnd_cost = random.Random((((dcop_id+1)+100)+((a1.id_+1)+10)+((a2.id_+1)*1))*17)
        
        # Store penalties for graph coloring cost model
        self.penalty_a1 = penalty_a1 if penalty_a1 is not None else self.rnd_cost.normalvariate(50, 10)
        self.penalty_a2 = penalty_a2 if penalty_a2 is not None else self.rnd_cost.normalvariate(50, 10)
        
        self.cost_table = {}
        self.create_dictionary_of_costs()  # Initialize the cost table with graph coloring costs

    # Get the cost associated with the given variables of two agents
    def get_cost(self, a1_id, a1_variable, a2_id, a2_variable):
        ap =((str(a1_id),a1_variable),(str(a2_id),a2_variable))
        ans = self.cost_table[ap]
        return ans

    # Create a dictionary of costs for all possible combinations of agent variables
    # Graph coloring cost model: cost = penalty_i + penalty_j if colors match, 0 otherwise
    def create_dictionary_of_costs(self):
        for d_a1 in self.a1.domain:
            for d_a2 in self.a2.domain:
                first_tuple = (str(self.a1.id_),d_a1)
                second_tuple = (str(self.a2.id_),d_a2)
                ap = (first_tuple,second_tuple)
                
                # Graph coloring cost: penalty sum if same color, 0 if different colors
                if d_a1 == d_a2:  # Same color (conflict)
                    cost = self.penalty_a1 + self.penalty_a2
                else:  # Different colors (no conflict)
                    cost = 0
                
                self.cost_table[ap] = cost

    # Check if a given agent is part of this neighbor relationship
    def is_agent_in_obj(self,agent_id_input):
        if agent_id_input == self.a1.id_  or agent_id_input == self.a2.id_ :
            return True
        return False

    # Get the other agent in the neighbor relationship
    def get_other_agent(self,agent_input):
        if agent_input.id_ == self.a1.id_:
            return self.a2.id_
        else:
            return self.a1.id_


# Class to define an unbounded message buffer
class UnboundedBuffer():

    def __init__(self):
        self.buffer = []

    # Insert a list of messages into the buffer
    def insert(self, list_of_msgs):
        for msg in list_of_msgs:
            self.buffer.append(msg)

    # Extract all messages from the buffer
    def extract(self):
        ans = []
        for msg in self.buffer:
            if msg is None:
                return None
            else:
                ans.append(msg)
        self.buffer = []
        return ans

# Class to manage message passing between agents in a DCOP
class Mailer():
    def __init__(self,agents):
        self.inbox = UnboundedBuffer()
        self.agents_outbox = {}
        for i, agent in enumerate(agents):
            # Defensive check to catch the error early
            if not hasattr(agent, 'id_'):
                raise TypeError(f"Agent at index {i} is not an agent object (type: {type(agent)}, value: {agent}). Expected agent object with 'id_' attribute.")
            
            outbox = UnboundedBuffer()
            self.agents_outbox[agent.id_] = outbox
            agent.inbox = outbox
            agent.outbox = self.inbox

    # Place messages in the agents' inboxes
    def place_messages_in_agents_inbox(self):
        msgs_to_send = self.inbox.extract()
        if len(msgs_to_send) == 0: return True
        msgs_by_receiver_dict = self.create_msgs_by_receiver_dict(msgs_to_send)
        for receiver,msgs_list in msgs_by_receiver_dict.items():
            self.agents_outbox[receiver].insert(msgs_list)

    # Create a dictionary of messages categorized by their receivers
    def create_msgs_by_receiver_dict(self,msgs_to_send):
        msgs_by_receiver_dict = {}
        for msg in msgs_to_send:
            receiver = msg.receiver
            if receiver not in msgs_by_receiver_dict.keys():
                msgs_by_receiver_dict[receiver] = []
            msgs_by_receiver_dict[receiver].append(msg)
        return msgs_by_receiver_dict

# Abstract base class for DCOP problems
class DCOP(ABC):
    def __init__(self,id_,A,d,dcop_name,algorithm, k, p = None, agent_mu_config=None, shared_topology=None, current_episode=0):
        self.dcop_id = id_
        self.A = A  # Number of agents
        self.d = d  # size of domain
        self.k = k  # Probability of edge creation between agents
        self.p=p  # Probability parameter for DSA
        self.algorithm = algorithm  # Algorithm to be used
        self.dcop_name = dcop_name
        self.agents = []

        # Agent priority configuration (mu values for penalty distribution)
        self.agent_mu_config = agent_mu_config or {}

        # Shared topology for synchronized experiments
        self.shared_topology = shared_topology
        self.use_shared_topology = shared_topology is not None
        self.current_episode = current_episode

        # Generate per-agent penalties for graph coloring cost model
        self.rnd_penalties = random.Random((id_+1)*23)
        self.agent_penalties = {}
        self.agent_mu_values = {}  # Store the mu value used for each agent

        self.create_agents()  # Initialize agents
        self.neighbors = []
        self.rnd_neighbors = random.Random((id_+5)*17)
        
        if self.use_shared_topology:
            # Use shared topology and mu values
            self.agent_mu_values = self.shared_topology.agent_mu_values.copy()
            self.create_neighbors_from_shared_topology(self.current_episode)
        else:
            # Use original random generation
            self.generate_agent_mu_values()  # Generate per-agent mu values for priorities
            self.create_neighbors()
        
        self.connect_agents_to_neighbors()
        self.mailer = Mailer(self.agents)  # Initialize mailer
        self.global_clock = 0
        self.global_costs=[]
        
        # Initialize attributes that may be checked dynamically
        self.agent_mu_values = {}  # Already initialized above but make explicit
        self.all_episode_costs = []  # For multi-episode algorithms

    # Connect each agent to its neighbors
    def connect_agents_to_neighbors(self):
        for agent in self.agents:
            neighbors_of_a = self.get_all_neighbors_obj_of_agent(agent)
            agent.set_neighbors(neighbors_of_a)

    # Get all neighbor objects for a given agent
    def get_all_neighbors_obj_of_agent(self, agent:Agent):
        ans = []
        for n in self.neighbors:
            if n.is_agent_in_obj(agent.id_):
                ans.append(n)
        return ans

    # Calculate the global cost of the current state
    def calc_global_cost(self):
        global_cost = 0
        for n in self.neighbors:
            a1_id, a1_variable= n.a1.id_, n.a1.variable
            a2_id, a2_variable= n.a2.id_, n.a2.variable
            global_cost += n.get_cost(a1_id, a1_variable, a2_id, a2_variable)
        return global_cost

    # Execute the DCOP algorithm
    def execute(self):
        initial_global_cost =self.calc_global_cost()
        self.global_costs.append(initial_global_cost)

        for agent in self.agents:
            agent.initialize()

        for i in range(iteration_per_episode):
            self.global_clock = self.global_clock + 1
            is_empty = self.mailer.place_messages_in_agents_inbox()
            if is_empty:
                print("DCOP:",str(self.dcop_id),"global clock:",str(self.global_clock), "is over because there are no messages in system ")
                break
            self.agents_perform_iteration(self.global_clock)
            current_global_cost=self.calc_global_cost()
            self.global_costs.append(current_global_cost)

        return self.global_costs

    def compute_global_context(self):
        """Default implementation for global context computation - override in learning algorithms."""
        return None
    
    def get_learning_statistics(self):
        """Default implementation for learning statistics - override in learning algorithms."""
        return None

    def generate_agent_mu_values(self):
        """Generate mu values for each agent based on configuration"""
        config = self.agent_mu_config
        validate_agent_mu_config(config)

        # Default: uniform mu = 50 for all agents
        default_mu = config['default_mu']
        default_sigma = config['default_sigma']

        for i in range(1, self.A + 1):
            self.agent_mu_values[i] = default_mu

        # Manual assignment: specific mu for specific agents
        if 'manual' in config:
            for agent_id, mu_value in config['manual'].items():
                if 1 <= agent_id <= self.A:
                    self.agent_mu_values[agent_id] = mu_value

        # Hierarchical assignment: mu based on agent ID ranges
        if 'hierarchical' in config:
            for priority_level, (start_id, end_id, mu_value) in config['hierarchical'].items():
                for agent_id in range(start_id, min(end_id + 1, self.A + 1)):
                    self.agent_mu_values[agent_id] = mu_value

        # Random stratified assignment
        if 'random_stratified' in config:
            agent_ids = list(range(1, self.A + 1))
            random.Random(self.dcop_id + 42).shuffle(agent_ids)  # Deterministic random

            idx = 0
            for priority_name, (count, mu_value, mu_sigma) in config['random_stratified'].items():
                for _ in range(min(count, len(agent_ids) - idx)):
                    if idx < len(agent_ids):
                        self.agent_mu_values[agent_ids[idx]] = mu_value
                        idx += 1

    def create_neighbors_from_shared_topology(self, episode_num=0):
        """Create neighbors using shared topology with episode-specific penalties"""
        # Get episode-specific penalties from shared topology
        episode_penalties, episode_assignments = self.shared_topology.get_episode_data(episode_num)
        self.agent_penalties = episode_penalties
        
        # Create neighbors based on shared topology
        topology = self.shared_topology.get_topology()
        
        for agent1_id, agent2_id in topology:
            # Find agent objects by their IDs
            a1 = next(agent for agent in self.agents if agent.id_ == agent1_id)
            a2 = next(agent for agent in self.agents if agent.id_ == agent2_id)
            
            penalty_a1 = self.agent_penalties[agent1_id]
            penalty_a2 = self.agent_penalties[agent2_id]
            
            self.neighbors.append(Neighbors(a1, a2, self.dcop_id, penalty_a1, penalty_a2))
        
        # Set starting assignments from shared topology
        for agent in self.agents:
            if agent.id_ in episode_assignments:
                agent.variable = episode_assignments[agent.id_]
    
    def update_costs_for_episode(self, episode_num):
        """Update constraint costs for a new episode using shared topology"""
        if not self.use_shared_topology:
            return  # No shared topology, skip
        
        # Get new episode penalties
        episode_penalties, episode_assignments = self.shared_topology.get_episode_data(episode_num)
        self.agent_penalties = episode_penalties
        
        # Update all neighbor cost tables with new penalties
        topology = self.shared_topology.get_topology()
        topology_dict = {(min(a1, a2), max(a1, a2)): (a1, a2) for a1, a2 in topology}
        
        for neighbor in self.neighbors:
            a1_id = neighbor.a1.id_
            a2_id = neighbor.a2.id_
            
            # Update penalties and regenerate cost table
            neighbor.penalty_a1 = episode_penalties[a1_id]
            neighbor.penalty_a2 = episode_penalties[a2_id]
            neighbor.create_dictionary_of_costs()
        
        # Reset agent variables to episode starting assignments
        for agent in self.agents:
            if agent.id_ in episode_assignments:
                agent.variable = episode_assignments[agent.id_]
    
    # Create neighbors based on the probability k (original method)
    def create_neighbors(self):
        # Generate penalties for all agents using their specific mu values
        validate_agent_mu_config(self.agent_mu_config)
        default_sigma = self.agent_mu_config['default_sigma']

        for i in range(1, self.A + 1):
            mu_i = self.agent_mu_values[i]
            self.agent_penalties[i] = self.rnd_penalties.normalvariate(mu_i, default_sigma)

        for i in range(self.A):
            a1 = self.agents[i]
            for j in range(i+1,self.A):
                a2 = self.agents[j]
                rnd_number = self.rnd_neighbors.random()
                if rnd_number<self.k:
                    penalty_a1 = self.agent_penalties[a1.id_]
                    penalty_a2 = self.agent_penalties[a2.id_]
                    self.neighbors.append(Neighbors(a1, a2, self.dcop_id, penalty_a1, penalty_a2))

    # Perform an iteration for all agents
    def agents_perform_iteration(self, global_clock):
        # Compute global context for contextual learning (DSA-RL only)
        global_context = self.compute_global_context()
        
        for agent in self.agents:
            if agent.is_contextual_agent():
                # DSA-RL agent with contextual learning
                agent.execute_iteration_with_context(global_clock, global_context)
            else:
                # Standard agent
                agent.execute_iteration(global_clock)

    # Credit assignment methods for REINFORCE learning
    def get_changers_and_gains(self):
        """Get agents that made changes and their local gains in the last iteration"""
        changers = []
        local_gains = {}

        for agent in self.agents:
            if agent.get_did_flip():
                changers.append(agent.id_)
                local_gains[agent.id_] = agent.get_local_gain()

        return changers, local_gains

    def calculate_global_improvement(self, prev_global_cost, current_global_cost):
        """Calculate global improvement from previous to current iteration"""
        return prev_global_cost - current_global_cost

    def distribute_episode_rewards(self, changers, local_gains, global_improvement):
        """Distribute rewards among changers based on their contributions"""
        return distribute_rewards_proportionally(changers, local_gains, global_improvement)

    # Abstract method to create agents, to be implemented by subclasses
    @abstractmethod
    def create_agents(self):
        pass


# Class for DCOP using DSA with Learned Probabilities (no learning, just application)
class DCOP_DSA_Learned(DCOP):
    def __init__(self, id_, A, d, dcop_name, algorithm, k, learned_probabilities, agent_mu_config=None, shared_topology=None, current_episode=0):
        # Store learned probabilities before calling parent init
        self.learned_probabilities = learned_probabilities
        
        # Call parent initialization
        DCOP.__init__(self, id_, A, d, dcop_name, algorithm, k, agent_mu_config=agent_mu_config, shared_topology=shared_topology, current_episode=current_episode)

    def create_agents(self):
        """Create DSA agents with learned probabilities"""
        from .agents import DSA_Agent_Learned
        for i in range(self.A):
            agent_id = i + 1
            # Get learned probability for this agent (fallback to 0.5 if not found)
            learned_p = self.learned_probabilities[agent_id] if agent_id in self.learned_probabilities else 0.5
            
            agent = DSA_Agent_Learned(agent_id, self.d, learned_p)
            self.agents.append(agent)
    
    def get_agent_probabilities(self):
        """Get the probabilities being used by all agents"""
        probabilities = {}
        for agent in self.agents:
            probabilities[agent.id_] = agent.p
        return probabilities


# Class for DCOP using the DSA algorithm
class DCOP_DSA(DCOP):

    def __init__(self, id_,A,d,dcop_name,algorithm, k, p, agent_mu_config=None, shared_topology=None, current_episode=0):
        DCOP.__init__(self,id_,A,d,dcop_name,algorithm, k, p, agent_mu_config, shared_topology, current_episode)

    # Create DSA agents
    def create_agents(self):
        for i in range(self.A):
            self.agents.append(DSA_Agent(i + 1, self.d, self.p))


# Class for DCOP using the MGM algorithm
class DCOP_MGM(DCOP):

    def __init__(self, id_,A,d,dcop_name,algorithm, k, agent_mu_config=None, shared_topology=None, current_episode=0):
        DCOP.__init__(self,id_,A,d,dcop_name,algorithm, k, agent_mu_config=agent_mu_config, shared_topology=shared_topology, current_episode=current_episode)

    # Create MGM agents
    def create_agents(self):
        for i in range(self.A):
            self.agents.append(MGM_Agent(i + 1, self.d))


# Class for DCOP using DSA with REINFORCE learning
class DCOP_DSA_RL(DCOP):
    def __init__(self, id_, A, d, dcop_name, algorithm, k, p0, learning_rate,
                 baseline_decay, iteration_per_episode, num_episodes, agent_mu_config=None, shared_topology=None, current_episode=0):

        # Initialize hyperparameters before calling parent init
        self.p0 = p0
        self.learning_rate = learning_rate
        self.baseline_decay = baseline_decay
        self.iteration_per_episode = iteration_per_episode
        self.num_episodes = num_episodes  # Number of learning episodes to run
        
        # Call parent initialization first
        DCOP.__init__(self, id_, A, d, dcop_name, algorithm, k, agent_mu_config=agent_mu_config, shared_topology=shared_topology, current_episode=current_episode)
        
        # Multi-episode tracking
        self.all_episode_costs = []  # List of cost curves, one per episode
        self.episode_statistics = []  # Learning statistics per episode
        self.probability_evolution = {}  # Track probability changes over episodes for each agent
        self.policy_convergence_metrics = {}  # Track convergence of policy parameters
        
        # Current episode state
        self.current_episode = 0
        self.iteration_in_episode = 0
        self.episode_rewards = {}  # agent_id -> list of rewards per current episode

        # Global feature tracking for actor-critic
        self.recent_global_costs = []  # Track recent global costs for trend analysis
        self.recent_activity_levels = []  # Track recent agent activity
        self.global_features_history = []  # Store global features for training
        
        # Adaptive global feature normalization statistics
        self.global_feature_running_stats = {
            'cost': {'sum': 0, 'sum_sq': 0, 'count': 0, 'mean': 0, 'std': 1},
            'violations': {'sum': 0, 'sum_sq': 0, 'count': 0, 'mean': 0, 'std': 1},
            'trend': {'sum': 0, 'sum_sq': 0, 'count': 0, 'mean': 0, 'std': 1},
            'convergence': {'sum': 0, 'sum_sq': 0, 'count': 0, 'mean': 0, 'std': 1}
        }
        
        # Linear Critic (Value Function) parameters
        self.num_global_features = 6  # Number of global features
        # Initialize value weights for linear model: V = φ^T * global_features
        self.value_weights = np.random.normal(0, 0.2, self.num_global_features + 1)  # +1 for bias
        
        # Adaptive learning rates based on problem scale
        self.base_learning_rate = learning_rate
        self.adaptive_actor_rate = self._calculate_adaptive_learning_rate(A, 'actor')
        self.adaptive_critic_rate = self._calculate_adaptive_learning_rate(A, 'critic')
        self.critic_learning_rate = self.adaptive_critic_rate
        self.gamma = 0.9  # Discount factor for future rewards (reduced for shorter episodes)
        
        # Multi-scale learning parameters
        self.problem_scale_factor = A / 30.0  # Normalize by baseline 30 agents
        self.curriculum_stage = self._determine_curriculum_stage(A, d)
        
        # Initialize episode rewards tracking for each agent
        for agent in self.agents:
            self.episode_rewards[agent.id_] = []
    
    def create_agents(self):
        """Create DSA agents with REINFORCE learning capabilities"""
        for i in range(self.A):
            agent = DsaAgentAdaptive(
                i + 1, self.d, self.p0, self.learning_rate, self.baseline_decay
            )
            self.agents.append(agent)
    
    def execute(self):
        """Execute multiple episodes of DCOP with persistent REINFORCE learning"""
        
        # Run multiple learning episodes with the same agents
        for episode_num in range(self.num_episodes):
            print(f"DSA-RL Episode {episode_num + 1}/{self.num_episodes} [{self.curriculum_stage} problem, scale={self.problem_scale_factor:.1f}]")

            # Update adaptive learning rates based on curriculum progress
            self.update_adaptive_learning_rates(episode_num)
            
            episode_costs = self.run_single_episode(episode_num)
            
            # Store episode results
            self.all_episode_costs.append(episode_costs)
            
            # Update agent parameters based on episode performance
            self.update_agent_parameters_after_episode()
            
            # Collect learning statistics
            episode_stats = self.get_learning_statistics()
            self.episode_statistics.append(episode_stats)
            
            # Track probability evolution
            self.track_probability_evolution(episode_num)
            
            self.prepare_agents_for_next_episode()
        
        # Return averaged cost curve across all episodes to match DSA/MGM comparison methodology
        if not self.all_episode_costs:
            return []
        
        # Calculate element-wise average across all episode cost curves
        max_length = max(len(episode_costs) for episode_costs in self.all_episode_costs)
        averaged_costs = []
        
        for i in range(max_length):
            # Average cost at iteration i across all episodes that have data for iteration i
            episode_costs_at_i = [episode_costs[i] for episode_costs in self.all_episode_costs if i < len(episode_costs)]
            if episode_costs_at_i:
                avg_cost = sum(episode_costs_at_i) / len(episode_costs_at_i)
                averaged_costs.append(avg_cost)
        
        return averaged_costs
    
    def prepare_agents_for_next_episode(self):
        """Reset agents for the next episode while keeping the graph structure intact."""
        # Reset episode tracking first
        self.current_episode += 1
        self.iteration_in_episode = 0
        self.episode_rewards = {agent.id_: [] for agent in self.agents}
        
        # Reset feature tracking for new episode
        self.reset_episode_features()
        
        # Reset agent-level feature tracking
        for agent in self.agents:
            agent.recent_costs = []
            agent.neighbor_changes = []
            agent.previous_neighbor_values = {}
        
        if self.use_shared_topology:
            # Use shared topology for synchronized cost updates and starting assignments
            self.update_costs_for_episode(self.current_episode)
        else:
            # Original behavior: regenerate costs randomly
            for neighbor in self.neighbors:
                neighbor.create_dictionary_of_costs()
            # Reset agent variables to a random value within their domain  
            for agent in self.agents:
                agent.variable = agent.agent_random.randint(1,self.d)
            self.generate_agent_mu_values()


    
    def run_single_episode(self, episode_num):
        """Run a single episode of the DCOP algorithm with actor-critic learning"""
        episode_costs = []
        global_features_sequence = []  # Store global features for critic learning
        
        # Calculate initial cost for this episode
        initial_global_cost = self.calc_global_cost()
        episode_costs.append(initial_global_cost)
        
        # Extract initial global features
        initial_features = self.extract_global_features(0)
        global_features_sequence.append(initial_features)
        
        # Initialize agents (send initial messages)
        for agent in self.agents:
            agent.initialize()
        
        prev_global_cost = initial_global_cost
        
        # Run for iteration_per_episode iterations
        for i in range(self.iteration_per_episode):
            self.global_clock += 1
            self.iteration_in_episode += 1
            
            # Check if messages exist
            is_empty = self.mailer.place_messages_in_agents_inbox()
            if is_empty:
                print(f"Episode {episode_num + 1} ended early at iteration {i + 1} (no messages)")
                break
            
            # Perform agent iterations  
            self.agents_perform_iteration(self.global_clock)
            
            # Calculate current global cost
            current_global_cost = self.calc_global_cost()
            episode_costs.append(current_global_cost)
            
            # Extract global features after iteration
            current_features = self.extract_global_features(i + 1)
            global_features_sequence.append(current_features)
            
            # Credit assignment: get changers and their local gains
            changers, local_gains = self.get_changers_and_gains()
            
            # Calculate global improvement as reward signal
            global_improvement = self.calculate_global_improvement(prev_global_cost, current_global_cost)
            
            # Use critic to compute advantages with proper temporal alignment
            if len(global_features_sequence) >= 2:
                prev_features = global_features_sequence[-2]
                current_features_iter = global_features_sequence[-1]
                
                # Use global_improvement as the immediate reward for transitioning from prev to current state
                # Update critic and get TD error as advantage estimate
                advantage = self.get_advantage_estimate(
                    prev_features, global_improvement, current_features_iter, done=False
                )
            else:
                # For first iteration, use simple reward signal since no previous state
                # This will be improved as the critic learns
                advantage = global_improvement
            
            # Distribute advantages among changers (instead of raw rewards)
            iteration_advantages = self.distribute_episode_rewards(changers, local_gains, advantage)
            
            # Store advantages for each agent (0 for non-changers)
            for agent in self.agents:
                agent_advantage = iteration_advantages[agent.id_] if agent.id_ in iteration_advantages else 0
                self.episode_rewards[agent.id_].append(agent_advantage)
            
            prev_global_cost = current_global_cost
        
        # Final critic update for terminal state
        if len(global_features_sequence) >= 2:
            # Update critic for the final transition to terminal state
            final_features = global_features_sequence[-1]
            # For terminal state, there's no next state, so we use done=True
            # The reward is 0 since we're not getting any more improvements
            self.update_critic(final_features, 0.0, next_features=None, done=True)
        
        # Store global features for potential cross-episode learning
        self.global_features_history.extend(global_features_sequence)
        
        return episode_costs
    
    def update_agent_parameters_after_episode(self):
        """Update agent theta parameters after episode completion"""
        for agent in self.agents:
            agent_rewards = self.episode_rewards[agent.id_]
            agent.finish_episode(agent_rewards)
    
    def get_learning_statistics(self):
        """Get current learning statistics for all agents"""
        stats = {}
        for agent in self.agents:
            if agent.policy_weights is not None:
                stats[agent.id_] = {
                    'policy_weights': agent.policy_weights.copy(),
                    'probability': agent.p,
                    'baseline': agent.baseline,
                    'features': agent.current_features.copy() if agent.current_features is not None else None,
                    'feature_stats': agent.feature_running_stats.copy() if agent.feature_running_stats is not None else None
                }
        return stats
    
    def get_all_episode_costs(self):
        """Get cost curves for all episodes"""
        return self.all_episode_costs
    
    def get_episode_statistics(self):
        """Get learning statistics progression across all episodes"""
        return self.episode_statistics
    
    def get_final_agent_statistics(self):
        """Get final learned parameters for each agent"""
        return self.get_learning_statistics()
    
    def track_probability_evolution(self, episode_num):
        """Track how agent probabilities evolve over episodes"""
        current_stats = self.get_learning_statistics()
        
        for agent_id, stats in current_stats.items():
            if agent_id not in self.probability_evolution:
                self.probability_evolution[agent_id] = {
                    'probabilities': [],
                    'policy_weights': [],
                    'baselines': [],
                    'episodes': []
                }
            
            self.probability_evolution[agent_id]['probabilities'].append(stats['probability'])
            self.probability_evolution[agent_id]['policy_weights'].append(stats['policy_weights'])
            self.probability_evolution[agent_id]['baselines'].append(stats['baseline'])
            self.probability_evolution[agent_id]['episodes'].append(episode_num)
    
    def analyze_policy_convergence(self):
        """Analyze convergence of learned policies"""
        convergence_analysis = {}
        
        for agent_id, evolution in self.probability_evolution.items():
            probs = evolution['probabilities']
            if len(probs) >= 10:  # Need enough data points
                # Calculate probability variance in recent episodes
                recent_probs = probs[-10:]
                prob_variance = np.var(recent_probs)
                prob_mean = np.mean(recent_probs)
                
                # Calculate policy weight changes
                if len(evolution['policy_weights']) >= 2:
                    recent_weights = evolution['policy_weights'][-5:]  # Last 5 episodes
                    weight_changes = []
                    for i in range(1, len(recent_weights)):
                        change = np.linalg.norm(recent_weights[i] - recent_weights[i-1])
                        weight_changes.append(change)
                    avg_weight_change = np.mean(weight_changes) if weight_changes else 0
                else:
                    avg_weight_change = float('inf')
                
                convergence_analysis[agent_id] = {
                    'probability_variance': prob_variance,
                    'probability_mean': prob_mean,
                    'policy_weight_change_rate': avg_weight_change,
                    'converged': prob_variance < 0.01 and avg_weight_change < 0.1,
                    'probability_range': (min(probs), max(probs)),
                    'final_probability': probs[-1]
                }
            else:
                convergence_analysis[agent_id] = {
                    'insufficient_data': True,
                    'final_probability': probs[-1] if probs else 0.5
                }
        
        return convergence_analysis
    
    def get_probability_evolution_summary(self):
        """Get summary of probability learning across all agents"""
        if not self.probability_evolution:
            return "No probability evolution data available"
        
        summary = {
            'total_agents': len(self.probability_evolution),
            'episodes_tracked': len(next(iter(self.probability_evolution.values()))['episodes']),
            'convergence_analysis': self.analyze_policy_convergence()
        }
        
        # Calculate overall learning metrics
        all_final_probs = [data['probabilities'][-1] for data in self.probability_evolution.values() if data['probabilities']]
        if all_final_probs:
            summary['final_probability_stats'] = {
                'mean': np.mean(all_final_probs),
                'std': np.std(all_final_probs),
                'min': min(all_final_probs),
                'max': max(all_final_probs)
            }
        
        return summary
    
    def _calculate_adaptive_learning_rate(self, num_agents, component):
        """Calculate adaptive learning rate based on problem scale"""
        
        # Base scaling: larger problems need smaller learning rates
        scale_factor = np.sqrt(30.0 / max(num_agents, 1))  # Normalize by baseline 30 agents
        
        if component == 'actor':
            # Actor rate decreases with problem size for stability
            adapted_rate = self.base_learning_rate * scale_factor
            # Clamp to reasonable bounds
            return max(0.001, min(adapted_rate, 0.02))
        
        elif component == 'critic':
            # Critic can handle slightly higher rates
            adapted_rate = self.base_learning_rate * scale_factor * 1.5
            # Clamp to reasonable bounds
            return max(0.001, min(adapted_rate, 0.03))
        
        return self.base_learning_rate
    
    def _determine_curriculum_stage(self, num_agents, domain_size):
        """Determine curriculum learning stage based on problem characteristics"""
        
        # Calculate problem difficulty ratio
        agent_to_color_ratio = num_agents / domain_size
        
        if agent_to_color_ratio <= 3:
            return 'easy'      # Plenty of colors available
        elif agent_to_color_ratio <= 6:
            return 'medium'    # Moderate constraint pressure
        else:
            return 'hard'      # High constraint pressure
    
    def update_adaptive_learning_rates(self, episode_num):
        """Update learning rates based on curriculum learning progress"""
        
        # Curriculum-based rate adjustment
        progress_factor = min(episode_num / (self.num_episodes * 0.3), 1.0)  # Ramp up over first 30%
        
        if self.curriculum_stage == 'easy':
            rate_multiplier = 1.2  # Can learn faster on easy problems
        elif self.curriculum_stage == 'medium':
            rate_multiplier = 1.0  # Standard rate
        else:  # hard
            rate_multiplier = 0.8  # More conservative on hard problems
        
        # Apply curriculum adjustment to agents
        current_actor_rate = self.adaptive_actor_rate * rate_multiplier * (0.5 + 0.5 * progress_factor)
        
        for agent in self.agents:
            if agent.policy_weights is not None:  # Only update learning agents
                agent.learning_rate = current_actor_rate
        
        # Update critic rate
        self.critic_learning_rate = self.adaptive_critic_rate * rate_multiplier * (0.5 + 0.5 * progress_factor)
    
    def compute_global_context(self):
        """Compute global context information for contextual agent decision making"""
        
        # Calculate convergence pressure based on recent cost improvements
        if len(self.recent_global_costs) >= 5:
            recent_costs = self.recent_global_costs[-5:]
            cost_variance = np.var(recent_costs)
            max_cost = max(recent_costs) if recent_costs else 1
            # High variance = low convergence pressure, low variance = high convergence pressure
            convergence_pressure = 1.0 - min(cost_variance / max(max_cost, 1), 1.0)
        else:
            convergence_pressure = 0.5  # Default moderate pressure
        
        # Calculate global activity level
        current_activity = self.calculate_activity_level()
        
        # Calculate solution quality trend
        if len(self.recent_global_costs) >= 3:
            recent_trend = (self.recent_global_costs[-3] - self.recent_global_costs[-1]) / 2
            max_possible_improvement = self.recent_global_costs[0] if self.recent_global_costs else 1
            trend_pressure = min(recent_trend / max(max_possible_improvement, 1), 1.0)
        else:
            trend_pressure = 0.0
        
        return {
            'convergence_pressure': convergence_pressure,
            'global_activity': current_activity,
            'improvement_trend': trend_pressure,
            'iteration_progress': self.iteration_in_episode / max(self.iteration_per_episode, 1)
        }
    
    def update_global_running_stats(self, feature_name, value):
        """Update running statistics for a global feature"""
        stats = self.global_feature_running_stats[feature_name]
        stats['sum'] += value
        stats['sum_sq'] += value * value
        stats['count'] += 1
        
        # Update mean and std
        stats['mean'] = stats['sum'] / stats['count']
        if stats['count'] > 1:
            variance = (stats['sum_sq'] / stats['count']) - (stats['mean'] * stats['mean'])
            stats['std'] = max(np.sqrt(max(variance, 0)), 0.01)  # Prevent zero std
    
    def normalize_global_with_running_stats(self, feature_name, value):
        """Normalize global feature using running statistics"""
        stats = self.global_feature_running_stats[feature_name]
        if stats['count'] < 5:  # Not enough data for reliable stats
            return np.tanh(value / 1000.0)  # Fallback normalization for global features
        
        # Z-score normalization: (value - mean) / std
        normalized = (value - stats['mean']) / stats['std']
        # Apply tanh to keep in reasonable range
        return np.tanh(normalized)
    
    def extract_global_features(self, current_iteration):
        """Extract global state features for centralized critic"""
        
        # Feature 1: total_cost - current global cost
        total_cost = self.calc_global_cost()
        
        # Feature 2: time_norm - normalized time (0 to 1)
        time_norm = current_iteration / self.iteration_per_episode if self.iteration_per_episode > 0 else 0.0
        
        # Feature 3: total_violations - sum of all constraint violations
        total_violations = sum(agent.count_violations() for agent in self.agents)
        
        # Feature 4: cost_trend - recent cost improvement rate
        cost_trend = self.calculate_cost_trend()
        
        # Feature 5: activity_level - fraction of agents that flipped recently
        activity_level = self.calculate_activity_level()
        
        # Feature 6: convergence_measure - stability of recent costs
        convergence = self.calculate_convergence_measure()
        
        # Update tracking
        self.recent_global_costs.append(total_cost)
        if len(self.recent_global_costs) > 10:  # Keep last 10 iterations
            self.recent_global_costs.pop(0)
        
        self.recent_activity_levels.append(activity_level)
        if len(self.recent_activity_levels) > 5:  # Keep last 5 iterations
            self.recent_activity_levels.pop(0)
        
        # Adaptive global feature normalization using running statistics
        # Update running statistics with current values
        self.update_global_running_stats('cost', total_cost)
        self.update_global_running_stats('violations', total_violations)
        self.update_global_running_stats('trend', cost_trend)
        self.update_global_running_stats('convergence', convergence)
        
        # Normalize features using adaptive statistics
        # Feature 1: total_cost - adaptive normalization
        normalized_cost = self.normalize_global_with_running_stats('cost', total_cost)
        
        # Feature 2: time_norm already normalized (0-1)
        
        # Feature 3: total_violations - adaptive normalization
        normalized_violations = self.normalize_global_with_running_stats('violations', total_violations)
        
        # Feature 4: cost_trend - adaptive normalization
        normalized_trend = self.normalize_global_with_running_stats('trend', cost_trend)
        
        # Feature 5: activity_level already normalized (0-1)
        
        # Feature 6: convergence - adaptive normalization
        normalized_convergence = self.normalize_global_with_running_stats('convergence', convergence)
        
        # Return normalized global feature vector
        features = np.array([
            normalized_cost,
            time_norm,
            normalized_violations,
            normalized_trend,
            activity_level,
            normalized_convergence
        ], dtype=np.float32)
        
        return features
    
    def calculate_cost_trend(self):
        """Calculate recent cost improvement trend"""
        if len(self.recent_global_costs) < 5:
            return 0.0
        
        # Linear trend over recent costs
        recent_costs = self.recent_global_costs[-5:]
        if len(recent_costs) > 1:
            # Simple: improvement rate = (first - last) / length
            return (recent_costs[0] - recent_costs[-1]) / len(recent_costs)
        return 0.0
    
    def calculate_activity_level(self):
        """Calculate fraction of agents that have been active recently"""
        if not self.agents:
            return 0.0
        
        active_agents = 0
        for agent in self.agents:
            if agent.get_did_flip():
                active_agents += 1
        
        return active_agents / len(self.agents)
    
    def calculate_convergence_measure(self):
        """Calculate convergence measure based on cost stability"""
        if len(self.recent_global_costs) < 3:
            return 0.0
        
        # Standard deviation of recent costs (lower = more converged)
        return float(np.std(self.recent_global_costs))
    
    def reset_episode_features(self):
        """Reset feature tracking for new episode"""
        self.recent_global_costs = []
        self.recent_activity_levels = []
        # Keep global_features_history for learning across episodes
    
    def compute_value_function(self, global_features):
        """Compute state value using linear value function: V = φ^T * [features, 1]"""
        # Add bias term (1.0) to features
        features_with_bias = np.append(global_features, 1.0)
        
        # Linear combination: φ^T * [features, bias]
        value = np.dot(self.value_weights, features_with_bias)
        
        return float(value)
    
    def update_critic(self, current_features, reward, next_features=None, done=False):
        """Update value function weights using temporal difference learning"""
        
        # Current state value
        current_value = self.compute_value_function(current_features)
        
        # Compute target value
        if done:
            target_value = reward
        else:
            if next_features is not None:
                next_value = self.compute_value_function(next_features)
                target_value = reward + self.gamma * next_value
            else:
                target_value = reward  # Fallback if no next state
        
        # Temporal difference error
        td_error = target_value - current_value
        
        # Update value weights using gradient descent
        # ∇φ V = features (for linear model)
        features_with_bias = np.append(current_features, 1.0)
        
        # Value function update: φ ← φ + α * td_error * ∇φ V
        self.value_weights += self.critic_learning_rate * td_error * features_with_bias
        
        # Clamp critic weights to prevent numerical issues
        self.value_weights = np.clip(self.value_weights, -100.0, 100.0)
        
        return td_error  # Return TD error as advantage estimate
    
    def get_advantage_estimate(self, current_features, reward, next_features=None, done=False):
        """Get advantage estimate using TD error from critic"""
        return self.update_critic(current_features, reward, next_features, done)


# =============================================================================
# CLEAN DCOP IMPLEMENTATIONS (Modern Architecture)
# =============================================================================

class DCOPBase(ABC):
    """
    Abstract base class for modern DCOP implementations.
    
    Provides cleaner architecture with proper typing and better separation of concerns.
    """
    
    def __init__(
        self,
        problem_id: int,
        num_agents: int,
        domain_size: int,
        problem_name: str,
        algorithm: Algorithm,
        edge_probability: float,
        probability: Optional[float] = None,
        agent_priority_config: Optional[Dict[str, Any]] = None,
        shared_topology: Optional[SharedGraphTopology] = None,
        current_episode: int = 0
    ):
        # Problem configuration
        self.problem_id = problem_id
        self.num_agents = num_agents
        self.domain_size = domain_size
        self.edge_probability = edge_probability
        self.probability = probability
        self.algorithm = algorithm
        self.problem_name = problem_name
        
        # Agent and constraint management
        self.agents: List[Agent] = []
        self.constraints = []
        
        # Priority/penalty configuration
        self.agent_priority_config = agent_priority_config or {}
        self.agent_penalties: Dict[int, float] = {}
        self.agent_penalty_means: Dict[int, float] = {}
        
        # Topology management  
        self.shared_topology = shared_topology
        self.use_shared_topology = shared_topology is not None
        self.current_episode = current_episode
        
        # Execution state
        self.global_clock = 0
        self.cost_history: List[float] = []
        
        # Initialize problem structure
        self._initialize_problem()
    
    def _initialize_problem(self) -> None:
        """Initialize the complete problem structure."""
        self.create_agents()
        
        if self.use_shared_topology:
            self._setup_from_shared_topology()
        else:
            self._setup_random_topology()
        
        self._connect_agents_to_constraints()
        self._setup_message_routing()
    
    def _setup_from_shared_topology(self) -> None:
        """Setup problem using shared topology."""
        if not self.shared_topology:
            return
            
        # Get episode data from shared topology
        episode_penalties, episode_assignments = self.shared_topology.get_episode_data(self.current_episode)
        self.agent_penalties = episode_penalties
        
        # Create constraints from shared topology
        topology_edges = self.shared_topology.get_topology()
        self._create_constraints_from_edges(topology_edges)
        
        # Set initial assignments
        self._set_initial_assignments(episode_assignments)
    
    def _setup_random_topology(self) -> None:
        """Setup problem using random topology generation."""
        self._generate_agent_penalties()
        self._create_random_constraints()
    
    def _generate_agent_penalties(self) -> None:
        """Generate random penalties for agents."""
        config = self.agent_priority_config
        validate_agent_mu_config(config)
        default_mu = config.get('default_mu', 50)
        default_sigma = config.get('default_sigma', 10)
        
        penalty_rng = random.Random(self.problem_id * 23)
        for i in range(1, self.num_agents + 1):
            self.agent_penalties[i] = penalty_rng.normalvariate(default_mu, default_sigma)
    
    def _create_constraints_from_edges(self, edges: List[Tuple[int, int]]) -> None:
        """Create constraints from edge list."""
        for agent1_id, agent2_id in edges:
            # Find agent objects
            agent1 = next(agent for agent in self.agents if agent.id_ == agent1_id)
            agent2 = next(agent for agent in self.agents if agent.id_ == agent2_id)
            
            penalty1 = self.agent_penalties[agent1_id]
            penalty2 = self.agent_penalties[agent2_id]
            
            constraint = Neighbors(agent1, agent2, self.problem_id, penalty1, penalty2)
            self.constraints.append(constraint)
    
    def _create_random_constraints(self) -> None:
        """Create constraints using random edge generation."""
        constraint_rng = random.Random(self.problem_id * 17)
        
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if constraint_rng.random() < self.edge_probability:
                    agent1 = self.agents[i]
                    agent2 = self.agents[j]
                    penalty1 = self.agent_penalties[agent1.id_]
                    penalty2 = self.agent_penalties[agent2.id_]
                    
                    constraint = Neighbors(agent1, agent2, self.problem_id, penalty1, penalty2)
                    self.constraints.append(constraint)
    
    def _connect_agents_to_constraints(self) -> None:
        """Connect agents to their relevant constraints."""
        for agent in self.agents:
            agent_constraints = [c for c in self.constraints if c.is_agent_in_obj(agent.id_)]
            agent.set_neighbors(agent_constraints)
    
    def _setup_message_routing(self) -> None:
        """Setup message routing between agents."""
        self.mailer = Mailer(self.agents)
    
    def _set_initial_assignments(self, assignments: Dict[int, int]) -> None:
        """Set initial variable assignments for agents."""
        for agent in self.agents:
            if agent.id_ in assignments:
                agent.variable = assignments[agent.id_]
    
    def calculate_global_cost(self) -> float:
        """Calculate total global cost of current state."""
        total_cost = 0.0
        for constraint in self.constraints:
            agent1_id = constraint.a1.id_
            agent2_id = constraint.a2.id_
            agent1_var = constraint.a1.variable
            agent2_var = constraint.a2.variable
            total_cost += constraint.get_cost(agent1_id, agent1_var, agent2_id, agent2_var)
        return total_cost
    
    def initialize_agents(self) -> None:
        """Initialize all agents."""
        for agent in self.agents:
            agent.initialize()
    
    def execute_iteration(self, iteration: int) -> None:
        """Execute one iteration of the algorithm."""
        self.global_clock = iteration
        
        # Route messages to agents
        self.mailer.place_messages_in_agents_inbox()
        
        # Execute agent iterations
        for agent in self.agents:
            agent.execute_iteration(iteration)
    
    @abstractmethod
    def create_agents(self) -> None:
        """Create agents for this problem. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def execute(self) -> List[float]:
        """Execute the DCOP algorithm. Must be implemented by subclasses."""
        pass


class StandardDSA(DCOPBase):
    """
    Standard DSA (Distributed Stochastic Algorithm) implementation.
    
    Agents make stochastic decisions to change variables based on a fixed probability.
    """
    
    def create_agents(self) -> None:
        """Create DSA agents with fixed probability."""
        for i in range(self.num_agents):
            agent_id = i + 1
            agent = DSAAgent(agent_id, self.domain_size, self.probability)
            self.agents.append(agent)
    
    def execute(self) -> List[float]:
        """Execute standard DSA algorithm."""
        # Record initial cost
        initial_cost = self.calculate_global_cost()
        self.cost_history.append(initial_cost)
        
        # Initialize agents
        self.initialize_agents()
        
        # Get iteration count from global config
        config = get_master_config()
        max_iterations = config.get('iterations', 100)
        
        # Run algorithm iterations
        for iteration in range(max_iterations):
            self.execute_iteration(iteration + 1)
            
            # Record cost after each iteration
            current_cost = self.calculate_global_cost()
            self.cost_history.append(current_cost)
        
        return self.cost_history


class LearnedPolicyDSA(DCOPBase):
    """
    DSA implementation using learned probability policies.
    
    Uses probabilities learned from previous DSA-RL training without further learning.
    """
    
    def __init__(
        self,
        problem_id: int,
        num_agents: int,
        domain_size: int,
        problem_name: str,
        algorithm: Algorithm,
        edge_probability: float,
        learned_probabilities: Dict[int, float],
        agent_priority_config: Optional[Dict[str, Any]] = None,
        shared_topology: Optional[SharedGraphTopology] = None,
        current_episode: int = 0
    ):
        self.learned_probabilities = learned_probabilities
        super().__init__(
            problem_id, num_agents, domain_size, problem_name, algorithm,
            edge_probability, None, agent_priority_config, shared_topology, current_episode
        )
    
    def create_agents(self) -> None:
        """Create agents with learned probabilities."""
        for i in range(self.num_agents):
            agent_id = i + 1
            learned_prob = self.learned_probabilities.get(agent_id, 0.5)
            agent = LearnedPolicyAgent(agent_id, self.domain_size, learned_prob)
            self.agents.append(agent)
    
    def execute(self) -> List[float]:
        """Execute DSA with learned probabilities."""
        # Record initial cost
        initial_cost = self.calculate_global_cost()
        self.cost_history.append(initial_cost)
        
        # Initialize agents
        self.initialize_agents()
        
        # Get iteration count from global config
        config = get_master_config()
        max_iterations = config.get('iterations', 100)
        
        # Run algorithm iterations
        for iteration in range(max_iterations):
            self.execute_iteration(iteration + 1)
            
            # Record cost after each iteration
            current_cost = self.calculate_global_cost()
            self.cost_history.append(current_cost)
        
        return self.cost_history
    
    def get_agent_probabilities(self) -> Dict[int, float]:
        """Get the probabilities being used by all agents."""
        return {agent.id_: agent.probability for agent in self.agents}


class MaximumGainMessages(DCOPBase):
    """
    MGM (Maximum Gain Messages) algorithm implementation.
    
    Uses coordinated decision making where agents calculate maximum gains
    and coordinate to avoid conflicts.
    """
    
    def create_agents(self) -> None:
        """Create MGM agents."""
        for i in range(self.num_agents):
            agent_id = i + 1
            agent = MGMAgent(agent_id, self.domain_size)
            self.agents.append(agent)
    
    def execute(self) -> List[float]:
        """Execute MGM algorithm."""
        # Record initial cost
        initial_cost = self.calculate_global_cost()
        self.cost_history.append(initial_cost)
        
        # Initialize agents
        self.initialize_agents()
        
        # Get iteration count from global config
        config = get_master_config()
        max_iterations = config.get('iterations', 100)
        
        # Run algorithm iterations
        for iteration in range(max_iterations):
            self.execute_iteration(iteration + 1)
            
            # Record cost after each iteration
            current_cost = self.calculate_global_cost()
            self.cost_history.append(current_cost)
        
        return self.cost_history


class ReinforcementLearningDSA(DCOPBase):
    """
    DSA with REINFORCE learning implementation.
    
    Agents learn optimal probability policies through reinforcement learning
    using actor-critic methods with local and global features.
    """
    
    def __init__(
        self,
        problem_id: int,
        num_agents: int,
        domain_size: int,
        problem_name: str,
        algorithm: Algorithm,
        edge_probability: float,
        initial_probability: float = 0.5,
        learning_rate: float = 0.01,
        baseline_decay: float = 0.9,
        iterations_per_episode: int = 100,
        num_episodes: int = 50,
        agent_priority_config: Optional[Dict[str, Any]] = None,
        shared_topology: Optional[SharedGraphTopology] = None,
        current_episode: int = 0
    ):
        self.initial_probability = initial_probability
        self.learning_rate = learning_rate
        self.baseline_decay = baseline_decay
        self.iterations_per_episode = iterations_per_episode
        self.num_episodes = num_episodes
        
        # Learning tracking
        self.all_episode_costs: List[List[float]] = []
        self.probability_evolution: Dict[int, List[float]] = {}
        
        super().__init__(
            problem_id, num_agents, domain_size, problem_name, algorithm,
            edge_probability, None, agent_priority_config, shared_topology, current_episode
        )
    
    def create_agents(self) -> None:
        """Create reinforcement learning agents."""
        for i in range(self.num_agents):
            agent_id = i + 1
            agent = ReinforcementLearningAgent(
                agent_id, self.domain_size, self.initial_probability,
                self.learning_rate, self.baseline_decay
            )
            self.agents.append(agent)
            
            # Initialize probability tracking
            self.probability_evolution[agent_id] = []
    
    def execute_single_episode(self, episode_num: int) -> List[float]:
        """Execute a single learning episode."""
        episode_costs = []
        
        # Reset episode data for all agents
        for agent in self.agents:
            agent.episode_data = []
        
        # Record initial cost
        initial_cost = self.calculate_global_cost()
        episode_costs.append(initial_cost)
        
        # Initialize agents
        self.initialize_agents()
        
        # Run episode iterations
        for iteration in range(self.iterations_per_episode):
            self.execute_iteration(iteration + 1)
            
            # Record cost
            current_cost = self.calculate_global_cost()
            episode_costs.append(current_cost)
            
            # Track probability evolution
            for agent in self.agents:
                self.probability_evolution[agent.id_].append(agent.probability)
        
        return episode_costs
    
    def finish_episode_learning(self, episode_costs: List[float]) -> None:
        """Apply learning updates after episode completion."""
        if len(episode_costs) < 2:
            return
        
        # Calculate per-iteration rewards based on global cost improvement
        iteration_rewards = []
        for i in range(len(episode_costs) - 1):
            reward = episode_costs[i] - episode_costs[i + 1]  # Positive for improvement
            iteration_rewards.append(reward)
        
        # Distribute rewards to agents based on their local contributions
        for agent in self.agents:
            if agent.episode_data:
                # Get local gains for this agent
                local_gains = [agent.get_local_gain() for _ in range(len(agent.episode_data))]
                
                # Distribute global rewards proportionally to local gains
                agent_rewards = distribute_rewards_proportionally(
                    iteration_rewards[:len(agent.episode_data)], 
                    local_gains
                )
                
                # Apply learning update
                agent.finish_episode(agent_rewards)
    
    def execute(self) -> List[float]:
        """Execute multiple episodes of reinforcement learning."""
        print(f"Starting DSA-RL training: {self.num_episodes} episodes, {self.iterations_per_episode} iterations each")
        
        final_costs = []
        
        for episode in range(self.num_episodes):
            # Prepare episode if using shared topology
            if self.shared_topology:
                self.shared_topology.prepare_episode(episode)
                # Update for new episode
                self._setup_from_shared_topology()
            
            # Execute episode
            episode_costs = self.execute_single_episode(episode)
            
            # Apply learning
            self.finish_episode_learning(episode_costs)
            
            # Track episode results
            self.all_episode_costs.append(episode_costs)
            final_costs = episode_costs  # Keep last episode costs
            
            # Progress reporting
            if episode % 10 == 0 or episode == self.num_episodes - 1:
                initial_cost = episode_costs[0]
                final_cost = episode_costs[-1]
                improvement = initial_cost - final_cost
                print(f"Episode {episode + 1}/{self.num_episodes}: "
                      f"Initial={initial_cost:.1f}, Final={final_cost:.1f}, "
                      f"Improvement={improvement:.1f}")
        
        return final_costs
    
    def get_final_agent_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get final statistics for all agents after training."""
        stats = {}
        
        for agent in self.agents:
            agent_stats = {
                'probability': getattr(agent, 'probability', 0.5),
                'baseline': getattr(agent, 'baseline', 0.0)
            }
            
            if hasattr(agent, 'policy_weights') and agent.policy_weights is not None:
                agent_stats['policy_weights'] = agent.policy_weights
            
            if hasattr(agent, 'feature_running_stats') and agent.feature_running_stats is not None:
                agent_stats['feature_stats'] = agent.feature_running_stats
            
            stats[agent.id_] = agent_stats
        
        return stats


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_dcop_problem(
    algorithm: Algorithm,
    problem_id: int = 0,
    num_agents: Optional[int] = None,
    domain_size: Optional[int] = None,
    edge_probability: Optional[float] = None,
    probability: Optional[float] = None,
    agent_priority_config: Optional[Dict[str, Any]] = None,
    shared_topology: Optional[SharedGraphTopology] = None,
    current_episode: int = 0,
    **kwargs
) -> DCOPBase:
    """
    Factory function to create the appropriate DCOP implementation.
    
    Args:
        algorithm: Algorithm type to use
        problem_id: Unique identifier for this problem
        num_agents: Number of agents (uses global config if None)
        domain_size: Domain size (uses global config if None)
        edge_probability: Edge probability for constraint graph
        probability: Probability parameter for DSA
        agent_priority_config: Agent priority configuration
        shared_topology: Shared topology for synchronized experiments
        current_episode: Current episode number
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Appropriate DCOP implementation instance
    """
    # Get defaults from global config if not provided
    config = get_master_config()
    if num_agents is None:
        num_agents = config['agents']
    if domain_size is None:
        domain_size = config['domain_size']
    if agent_priority_config is None:
        agent_priority_config = config['priority_variant']['stratified']
    
    problem_name = f"{algorithm.name}_{problem_id}"
    
    if algorithm == Algorithm.DSA:
        return StandardDSA(
            problem_id, num_agents, domain_size, problem_name, algorithm,
            edge_probability, probability, agent_priority_config, shared_topology, current_episode
        )
    elif algorithm == Algorithm.MGM:
        return MaximumGainMessages(
            problem_id, num_agents, domain_size, problem_name, algorithm,
            edge_probability, None, agent_priority_config, shared_topology, current_episode
        )
    elif algorithm == Algorithm.DSA_RL:
        # Get DSA-RL specific parameters from global config
        dsa_rl_config = config['dsa_rl']
        return ReinforcementLearningDSA(
            problem_id, num_agents, domain_size, problem_name, algorithm,
            edge_probability,
            initial_probability=kwargs.get('p0', dsa_rl_config['p0']),
            learning_rate=kwargs.get('learning_rate', dsa_rl_config['learning_rate']),
            baseline_decay=kwargs.get('baseline_decay', dsa_rl_config['baseline_decay']),
            iterations_per_episode=kwargs.get('iteration_per_episode', config['iterations']),
            num_episodes=kwargs.get('num_episodes', dsa_rl_config['num_episodes']),
            agent_priority_config=agent_priority_config,
            shared_topology=shared_topology,
            current_episode=current_episode
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def create_learned_policy_dcop(
    learned_probabilities: Dict[int, float],
    problem_id: int = 0,
    num_agents: Optional[int] = None,
    domain_size: Optional[int] = None,
    edge_probability: Optional[float] = None,
    agent_priority_config: Optional[Dict[str, Any]] = None,
    shared_topology: Optional[SharedGraphTopology] = None,
    current_episode: int = 0
) -> LearnedPolicyDSA:
    """
    Factory function to create DSA with learned probabilities.
    
    Args:
        learned_probabilities: Dictionary mapping agent IDs to learned probabilities
        Other args: Same as create_dcop_problem
        
    Returns:
        LearnedPolicyDSA instance
        :param learned_probabilities:
        :param current_episode:
        :param shared_topology:
        :param agent_priority_config:
        :param edge_probability:
        :param domain_size:
        :param num_agents:
        :param problem_id:
    """
    # Get defaults from global config if not provided
    config = get_master_config()
    if num_agents is None:
        num_agents = config['agents']
    if domain_size is None:
        domain_size = config['domain_size']
    if edge_probability is None:
        edge_probability = config.get('default_edge_probability', 0.3)
    if agent_priority_config is None:
        agent_priority_config = config['priority_variant']['stratified']
    
    problem_name = f"DSA_Learned_{problem_id}"
    
    return LearnedPolicyDSA(
        problem_id, num_agents, domain_size, problem_name, Algorithm.DSA,
        edge_probability, learned_probabilities, agent_priority_config, 
        shared_topology, current_episode
    )


# =============================================================================
# BACKWARD COMPATIBILITY AND LEGACY EXPERIMENT FUNCTIONS
# =============================================================================

def create_selected_dcop(i, algorithm, k, p=None, shared_topology=None, current_episode=0, **kwargs):
    """Legacy function for creating DCOP instances - backward compatibility"""
    from .validation import ensure_dictionary_keys
    
    config = get_master_config()
    A = config['agents']  # Number of agents from global config
    D = config['domain_size']  # Domain size from global config
    
    # Extract agent priority configuration
    ensure_dictionary_keys(kwargs, ['agent_mu_config'], {'agent_mu_config': config['priority_variant']['stratified']})
    agent_mu_config = kwargs['agent_mu_config']
    
    if algorithm == Algorithm.DSA:
        # Use create_dcop_problem for DSA
        return create_dcop_problem(
            algorithm=algorithm,
            problem_id=0 if shared_topology else i,
            num_agents=A,
            domain_size=D,
            edge_probability=k,
            probability=p,
            agent_priority_config=agent_mu_config,
            shared_topology=shared_topology,
            current_episode=current_episode
        )
    
    elif algorithm == Algorithm.MGM:
        # Use create_dcop_problem for MGM
        return create_dcop_problem(
            algorithm=algorithm,
            problem_id=0 if shared_topology else i,
            num_agents=A,
            domain_size=D,
            edge_probability=k,
            agent_priority_config=agent_mu_config,
            shared_topology=shared_topology,
            current_episode=current_episode
        )
    
    elif algorithm == Algorithm.DSA_RL:
        # Extract RL-specific parameters with global defaults
        dsa_rl_defaults = get_dsa_rl_hyperparameters()
        rl_keys = ['p0', 'learning_rate', 'baseline_decay', 'iteration_per_episode', 'num_episodes']
        rl_defaults = {
            'p0': dsa_rl_defaults['p0'],
            'learning_rate': dsa_rl_defaults['learning_rate'],
            'baseline_decay': dsa_rl_defaults['baseline_decay'],
            'iteration_per_episode': config['iterations'],
            'num_episodes': dsa_rl_defaults['num_episodes']
        }
        ensure_dictionary_keys(kwargs, rl_keys, rl_defaults)
        
        return create_dcop_problem(
            algorithm=algorithm,
            problem_id=0,
            num_agents=A,
            domain_size=D,
            edge_probability=k,
            agent_priority_config=agent_mu_config,
            shared_topology=shared_topology,
            current_episode=current_episode,
            **{k: v for k, v in kwargs.items() if k in rl_keys}
        )
    
    return None


def solve_synchronized_experiment(dcop_configs, k, shared_topology=None):
    """
    Run synchronized DSA vs DSA-RL experiment with shared topology and synchronized cost updates.
    
    Ensures:
    1. Same graph topology across all algorithms
    2. Same cost changes for each episode/"repetition" 
    3. Same starting assignments for fair comparison
    
    Args:
        dcop_configs: List of algorithm configurations
        k: Graph density (edge probability)
        shared_topology: Optional pre-created SharedGraphTopology instance
    
    Returns:
        Tuple of (results_dict, dsa_rl_stats, dsa_rl_dcop)
    """
    from .validation import ensure_dictionary_keys
    from .topology import SharedGraphTopology
    
    config = get_master_config()
    testing_params = get_testing_parameters()
    max_iterations = testing_params["max_cost_iterations"]
    repetitions = config["repetitions"]
    
    # Create shared topology if not provided
    if shared_topology is None:
        ensure_dictionary_keys(dcop_configs[0], ['agent_mu_config'], {'agent_mu_config': config['priority_variant']['stratified']})
        agent_mu_config = dcop_configs[0]['agent_mu_config']
        shared_topology = SharedGraphTopology(
            num_agents=config['agents'], 
            domain_size=config['domain_size'], 
            edge_probability=k, 
            agent_priority_config=agent_mu_config,
            base_seed=42,  # Fixed seed for reproducible experiments
            mode="comparison"  # Use comparison mode for fair evaluation
        )
    
    results = {}
    dsa_rl_stats = None
    dsa_rl_dcop = None
    
    for dcop_config in dcop_configs:
        algorithm = dcop_config['algorithm']
        ensure_dictionary_keys(dcop_config, ['name'], {'name': algorithm.name})
        
        if algorithm == Algorithm.DSA_RL:
            # DSA-RL: Single instance with multiple episodes using shared topology
            # Prepare episode 0 before creating DSA-RL to ensure synchronized starting conditions
            shared_topology.prepare_episode(0)
            
            rl_params = {k: v for k, v in dcop_config.items() if k not in ['algorithm', 'name']}
            rl_params['num_episodes'] = repetitions  # Set number of episodes
            
            dsa_rl_dcop = create_selected_dcop(0, algorithm, k, shared_topology=shared_topology, **rl_params)
            final_episode_costs = dsa_rl_dcop.execute()
            results[dcop_config['name']] = final_episode_costs
            
            # Store learning statistics
            if hasattr(dsa_rl_dcop, 'get_final_agent_statistics'):
                dsa_rl_stats = dsa_rl_dcop.get_final_agent_statistics()
            
        else:
            # DSA/MGM: Multiple instances, each using shared topology for different episodes
            total_costs = [0.0] * max_iterations
            
            for episode in range(repetitions):
                # Prepare shared topology for this episode
                shared_topology.prepare_episode(episode)
                
                # Create DCOP instance with shared topology
                if algorithm == Algorithm.DSA:
                    dcop = create_selected_dcop(episode, algorithm, k, dcop_config.get('probability', dcop_config.get('p', 0.5)), shared_topology, current_episode=episode)
                else:  # MGM
                    dcop = create_selected_dcop(episode, algorithm, k, shared_topology=shared_topology, current_episode=episode)
                
                # Execute this episode
                episode_costs = dcop.execute()
                
                # Accumulate costs
                for i in range(min(len(episode_costs), max_iterations)):
                    total_costs[i] += episode_costs[i]
            
            # Calculate average
            avg_costs = [cost / repetitions for cost in total_costs]
            results[dcop_config['name']] = avg_costs
    
    return results, dsa_rl_stats, dsa_rl_dcop

