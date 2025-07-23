from .Agents import *
from abc import ABC, abstractmethod
from .utils import distribute_rewards_proportionally

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
        for a in agents:
            outbox = UnboundedBuffer()
            self.agents_outbox[a.id_] = outbox
            a.inbox = outbox
            a.outbox = self.inbox

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
    def __init__(self,id_,A,d,dcop_name,algorithm, k, p = None, agent_mu_config=None):
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

        # Generate per-agent penalties for graph coloring cost model
        self.rnd_penalties = random.Random((id_+1)*23)
        self.agent_penalties = {}
        self.agent_mu_values = {}  # Store the mu value used for each agent

        self.create_agents()  # Initialize agents
        self.neighbors = []
        self.rnd_neighbors = random.Random((id_+5)*17)
        self.generate_agent_mu_values()  # Generate per-agent mu values for priorities
        self.create_neighbors()
        self.connect_agents_to_neighbors()
        self.mailer = Mailer(self.agents)  # Initialize mailer
        self.global_clock = 0
        self.global_costs=[]

    # Connect each agent to its neighbors
    def connect_agents_to_neighbors(self):
        for a in self.agents:
            neighbors_of_a = self.get_all_neighbors_obj_of_agent(a)
            a.set_neighbors(neighbors_of_a)

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

        for a in self.agents:
            a.initialize()

        for i in range(incomplete_iterations):
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

    # Create neighbors based on the probability k
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
        for a in self.agents:
            a.execute_iteration(global_clock)

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

    def __init__(self, id_,A,d,dcop_name,algorithm, k, p, agent_mu_config=None):
        DCOP.__init__(self,id_,A,d,dcop_name,algorithm, k, p, agent_mu_config)

    # Create DSA agents
    def create_agents(self):
        for i in range(self.A):
            self.agents.append(DSA_Agent(i + 1, self.d, self.p))


# Class for DCOP using the MGM algorithm
class DCOP_MGM(DCOP):

    def __init__(self, id_,A,d,dcop_name,algorithm, k, agent_mu_config=None):
        DCOP.__init__(self,id_,A,d,dcop_name,algorithm, k, agent_mu_config=agent_mu_config)

    # Create MGM agents
    def create_agents(self):
        for i in range(self.A):
            self.agents.append(MGM_Agent(i + 1, self.d))


# Class for DCOP using DSA with REINFORCE learning
class DCOP_DSA_RL(DCOP):
    def __init__(self, id_, A, d, dcop_name, algorithm, k, p0=0.7, learning_rate=0.01,
                 baseline_decay=0.9, episode_length=20, agent_mu_config=None):

        # Initialize hyperparameters before calling parent init
        self.p0 = p0
        self.learning_rate = learning_rate
        self.baseline_decay = baseline_decay
        self.episode_length = episode_length
        
        # Call parent initialization first
        DCOP.__init__(self, id_, A, d, dcop_name, algorithm, k, agent_mu_config=agent_mu_config)
        
        # Episode tracking - initialize after agents are created
        self.current_episode = 0
        self.iteration_in_episode = 0
        self.episode_rewards = {}  # agent_id -> list of rewards per episode

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
        """Execute DCOP with episodic REINFORCE learning"""
        initial_global_cost = self.calc_global_cost()
        self.global_costs.append(initial_global_cost)
        
        # Initialize agents
        for a in self.agents:
            a.initialize()
        
        prev_global_cost = initial_global_cost
        
        for i in range(incomplete_iterations):
            self.global_clock = self.global_clock + 1
            self.iteration_in_episode += 1
            
            # Check if messages exist
            is_empty = self.mailer.place_messages_in_agents_inbox()
            if is_empty:
                print("DCOP:", str(self.dcop_id), "global clock:", str(self.global_clock), 
                      "is over because there are no messages in system")
                break
            
            # Perform agent iterations
            self.agents_perform_iteration(self.global_clock)
            
            # Calculate current global cost
            current_global_cost = self.calc_global_cost()
            self.global_costs.append(current_global_cost)
            
            # Credit assignment: get changers and their local gains
            changers, local_gains = self.get_changers_and_gains()
            
            # Calculate global improvement and distribute rewards
            global_improvement = self.calculate_global_improvement(prev_global_cost, current_global_cost)
            iteration_rewards = self.distribute_episode_rewards(changers, local_gains, global_improvement)
            
            # Store rewards for each agent (0 for non-changers)
            for agent in self.agents:
                agent_reward = iteration_rewards.get(agent.id_, 0)
                if agent.id_ not in self.episode_rewards:
                    self.episode_rewards[agent.id_] = []
                self.episode_rewards[agent.id_].append(agent_reward)
            
            prev_global_cost = current_global_cost
            
            # Check if episode is complete
            if self.iteration_in_episode >= self.episode_length:
                self.finish_episode()
                self.start_new_episode()
        
        # Finish final episode if not already finished
        if self.iteration_in_episode > 0:
            self.finish_episode()
        
        return self.global_costs
    
    def finish_episode(self):
        """Finish current episode and update agent parameters"""
        for agent in self.agents:
            if hasattr(agent, 'finish_episode'):
                agent_rewards = self.episode_rewards.get(agent.id_, [])
                agent.finish_episode(agent_rewards)
        
        self.current_episode += 1
    
    def start_new_episode(self):
        """Start a new episode"""
        self.iteration_in_episode = 0
        # Clear reward history for new episode
        for agent_id in self.episode_rewards:
            self.episode_rewards[agent_id] = []
    
    def get_learning_statistics(self):
        """Get learning statistics for analysis"""
        stats = {}
        for agent in self.agents:
            if hasattr(agent, 'theta') and hasattr(agent, 'baseline'):
                stats[agent.id_] = {
                    'theta': agent.theta,
                    'probability': agent.p,
                    'baseline': agent.baseline
                }
        return stats