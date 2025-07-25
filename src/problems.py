from .agents import *
from abc import ABC, abstractmethod
from .utils import distribute_rewards_proportionally
import random
import numpy as np

# Class to manage shared graph topology and synchronized cost updates between DSA and DSA-RL
class SharedGraphTopology:
    """
    Manages shared graph topology and synchronized cost updates for fair DSA vs DSA-RL comparison.
    
    Ensures:
    1. Same graph connections (edges) across all algorithms and episodes  
    2. Synchronized cost changes that simulate "new day" scenarios
    3. Same starting agent assignments for fair comparison
    """
    
    def __init__(self, A, d, k, agent_mu_config, base_seed=42):
        self.A = A  # Number of agents
        self.d = d  # Domain size
        self.k = k  # Edge probability
        self.agent_mu_config = agent_mu_config or {}
        self.base_seed = base_seed
        
        # Fixed graph topology - determined once and shared
        self.graph_topology = []  # List of (agent1_id, agent2_id) tuples representing edges
        self.agent_mu_values = {}  # Fixed mu values for agents
        
        # Episode state for synchronized cost updates
        self.current_episode = 0
        self.episode_agent_penalties = {}  # Episode -> {agent_id -> penalty}
        self.episode_starting_assignments = {}  # Episode -> {agent_id -> variable}
        
        self._generate_fixed_topology()
        self._generate_fixed_mu_values()
    
    def _generate_fixed_topology(self):
        """Generate fixed graph topology that will be shared across all algorithms"""
        # Use deterministic seed for topology generation
        topology_rng = random.Random(self.base_seed * 17)
        
        self.graph_topology = []
        for i in range(1, self.A + 1):
            for j in range(i + 1, self.A + 1):
                if topology_rng.random() < self.k:
                    self.graph_topology.append((i, j))
        
        print(f"Generated shared topology with {len(self.graph_topology)} edges")
    
    def _generate_fixed_mu_values(self):
        """Generate fixed mu values for agents that remain constant"""
        # Default: uniform mu = 50 for all agents  
        config = self.agent_mu_config
        default_mu = config.get('default_mu', 50)
        
        for i in range(1, self.A + 1):
            self.agent_mu_values[i] = default_mu
        
        # Apply configuration overrides (manual, hierarchical, etc.)
        if 'manual' in config:
            for agent_id, mu_value in config['manual'].items():
                if 1 <= agent_id <= self.A:
                    self.agent_mu_values[agent_id] = mu_value
        
        if 'hierarchical' in config:
            for priority_level, (start_id, end_id, mu_value) in config['hierarchical'].items():
                for agent_id in range(start_id, min(end_id + 1, self.A + 1)):
                    self.agent_mu_values[agent_id] = mu_value
        
        if 'random_stratified' in config:
            agent_ids = list(range(1, self.A + 1))
            random.Random(self.base_seed + 42).shuffle(agent_ids)  # Deterministic shuffle
            
            idx = 0
            for priority_name, (count, mu_value, mu_sigma) in config['random_stratified'].items():
                for _ in range(min(count, len(agent_ids) - idx)):
                    if idx < len(agent_ids):
                        self.agent_mu_values[agent_ids[idx]] = mu_value
                        idx += 1
    
    def prepare_episode(self, episode_num):
        """
        Prepare synchronized episode data for both DSA and DSA-RL.
        
        Generates:
        1. New constraint costs (simulating "new day")
        2. Identical starting agent assignments
        """
        self.current_episode = episode_num
        
        # Generate episode-specific agent penalties with deterministic randomness
        penalty_rng = random.Random((self.base_seed + episode_num) * 23)
        default_sigma = self.agent_mu_config.get('default_sigma', 10)
        
        episode_penalties = {}
        for agent_id in range(1, self.A + 1):
            mu_i = self.agent_mu_values[agent_id]
            episode_penalties[agent_id] = penalty_rng.normalvariate(mu_i, default_sigma)
        
        self.episode_agent_penalties[episode_num] = episode_penalties
        
        # Generate episode-specific starting assignments
        assignment_rng = random.Random((self.base_seed + episode_num) * 31)
        episode_assignments = {}
        for agent_id in range(1, self.A + 1):
            episode_assignments[agent_id] = assignment_rng.randint(1, self.d)
        
        self.episode_starting_assignments[episode_num] = episode_assignments
        
        print(f"Prepared episode {episode_num}: {len(episode_penalties)} agent penalties, {len(episode_assignments)} starting assignments")
        
        return episode_penalties, episode_assignments
    
    def get_topology(self):
        """Get the shared graph topology"""
        return self.graph_topology.copy()
    
    def get_episode_data(self, episode_num):
        """Get penalty and assignment data for a specific episode"""
        if episode_num not in self.episode_agent_penalties:
            return self.prepare_episode(episode_num)
        
        return (
            self.episode_agent_penalties[episode_num], 
            self.episode_starting_assignments[episode_num]
        )

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
        for agent in agents:
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

    def generate_agent_mu_values(self):
        """Generate mu values for each agent based on configuration"""
        config = self.agent_mu_config

        # Default: uniform mu = 50 for all agents
        default_mu = config.get('default_mu', 50)
        default_sigma = config.get('default_sigma', 10)

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
        default_sigma = self.agent_mu_config.get('default_sigma', 10)

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
    def agents_perform_iteration(self,global_clock):
        for agent in self.agents:
            agent.execute_iteration(global_clock)

    # Credit assignment methods for REINFORCE learning
    def get_changers_and_gains(self):
        """Get agents that made changes and their local gains in the last iteration"""
        changers = []
        local_gains = {}

        for agent in self.agents:
            if hasattr(agent, 'get_did_flip') and hasattr(agent, 'get_local_gain'):
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
        
        # Current episode state
        self.current_episode = 0
        self.iteration_in_episode = 0
        self.episode_rewards = {}  # agent_id -> list of rewards per current episode

        # Global feature tracking for actor-critic
        self.recent_global_costs = []  # Track recent global costs for trend analysis
        self.recent_activity_levels = []  # Track recent agent activity
        self.global_features_history = []  # Store global features for training
        
        # Linear Critic (Value Function) parameters
        self.num_global_features = 6  # Number of global features
        # Initialize value weights for linear model: V = φ^T * global_features
        self.value_weights = np.random.normal(0, 0.1, self.num_global_features + 1)  # +1 for bias
        self.critic_learning_rate = learning_rate * 2  # Critic often needs higher learning rate
        self.gamma = 0.95  # Discount factor for future rewards
        
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
            print(f"DSA-RL Episode {episode_num + 1}/{self.num_episodes}")

            episode_costs = self.run_single_episode(episode_num)
            
            # Store episode results
            self.all_episode_costs.append(episode_costs)
            
            # Update agent parameters based on episode performance
            self.update_agent_parameters_after_episode()
            
            # Collect learning statistics
            episode_stats = self.get_learning_statistics()
            self.episode_statistics.append(episode_stats)
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
            if hasattr(agent, 'recent_costs'):
                agent.recent_costs = []
            if hasattr(agent, 'neighbor_changes'):
                agent.neighbor_changes = []
            if hasattr(agent, 'previous_neighbor_values'):
                delattr(agent, 'previous_neighbor_values')
        
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
            
            # Use critic to compute advantages instead of simple reward distribution
            if len(global_features_sequence) >= 2:
                prev_features = global_features_sequence[-2]
                current_features_iter = global_features_sequence[-1]
                
                # Update critic and get advantage estimate
                advantage = self.get_advantage_estimate(
                    prev_features, global_improvement, current_features_iter, done=False
                )
            else:
                # Fallback for first iteration
                advantage = global_improvement
            
            # Distribute advantages among changers (instead of raw rewards)
            iteration_advantages = self.distribute_episode_rewards(changers, local_gains, advantage)
            
            # Store advantages for each agent (0 for non-changers)
            for agent in self.agents:
                agent_advantage = iteration_advantages.get(agent.id_, 0)
                self.episode_rewards[agent.id_].append(agent_advantage)
            
            prev_global_cost = current_global_cost
        
        # Final critic update for terminal state
        if global_features_sequence:
            final_features = global_features_sequence[-1]
            final_reward = 0  # Terminal reward
            self.update_critic(final_features, final_reward, done=True)
        
        # Store global features for potential cross-episode learning
        self.global_features_history.extend(global_features_sequence)
        
        return episode_costs
    
    def update_agent_parameters_after_episode(self):
        """Update agent theta parameters after episode completion"""
        for agent in self.agents:
            if hasattr(agent, 'finish_episode'):
                agent_rewards = self.episode_rewards[agent.id_]
                agent.finish_episode(agent_rewards)
    
    def get_learning_statistics(self):
        """Get current learning statistics for all agents"""
        stats = {}
        for agent in self.agents:
            if hasattr(agent, 'theta') and hasattr(agent, 'baseline'):
                stats[agent.id_] = {
                    'theta': agent.theta,
                    'probability': agent.p,
                    'baseline': agent.baseline
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
    
    def extract_global_features(self, current_iteration):
        """Extract global state features for centralized critic"""
        
        # Feature 1: total_cost - current global cost
        total_cost = self.calc_global_cost()
        
        # Feature 2: time_norm - normalized time (0 to 1)
        time_norm = current_iteration / self.iteration_per_episode if self.iteration_per_episode > 0 else 0.0
        
        # Feature 3: total_violations - sum of all constraint violations
        total_violations = sum(agent.count_violations() for agent in self.agents if hasattr(agent, 'count_violations'))
        
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
        
        # Return global feature vector
        features = np.array([
            total_cost,
            time_norm,
            total_violations,
            cost_trend,
            activity_level,
            convergence
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
            if hasattr(agent, 'get_did_flip') and agent.get_did_flip():
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
        self.value_weights = np.clip(self.value_weights, -50.0, 50.0)
        
        return td_error  # Return TD error as advantage estimate
    
    def get_advantage_estimate(self, current_features, reward, next_features=None, done=False):
        """Get advantage estimate using TD error from critic"""
        return self.update_critic(current_features, reward, next_features, done)

