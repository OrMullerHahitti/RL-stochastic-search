from .agents import *
from abc import ABC, abstractmethod
from .utils import distribute_rewards_proportionally
from .validation import validate_agent_mu_config, ensure_dictionary_keys
from .topology import SharedGraphTopology
from .global_map import Algorithm, get_testing_parameters, get_master_config, get_dsa_rl_hyperparameters
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
        p: Optional[float] = None,
        agent_priority_config: Optional[Dict[str, Any]] = None,
        shared_topology: Optional[SharedGraphTopology] = None,
        current_episode: int = 0
    ):
        # Problem configuration
        self.problem_id = problem_id
        self.num_agents = num_agents
        self.domain_size = domain_size
        self.edge_probability = edge_probability
        self.p = p
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
    
    Agents make stochastic decisions to change variables based on a fixed p.
    """
    
    def create_agents(self) -> None:
        """Create DSA agents with fixed p."""
        for i in range(self.num_agents):
            agent_id = i + 1
            agent = DSAAgent(agent_id, self.domain_size, self.p)
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
    DSA implementation using learned p policies.
    
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
            agent = DSAAgent(agent_id, self.domain_size, learned_prob)
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
        return {agent.id_: agent.p for agent in self.agents}


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
    
    Agents learn optimal p policies through reinforcement learning
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
        self.decay_factor = baseline_decay
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
                self.learning_rate, self.decay_factor
            )
            self.agents.append(agent)
            
            # Initialize p tracking
            self.probability_evolution[agent_id] = []
    
    def execute_single_episode(self, episode_num: int) -> List[float]:
        """Execute a single learning episode."""
        episode_costs = []
        
        # Reset episode data and start new episode for all agents
        for agent in self.agents:
            agent.episode_data = []

        
        # Record initial cost
        initial_cost = self.calculate_global_cost()
        episode_costs.append(initial_cost)
        
        # DIAGNOSTIC: Check episode setup
        if episode_num < 3 or episode_num % 10 == 0:  # Log first few episodes and every 10th
            agent_probs = [getattr(agent, 'p', getattr(agent, 'p', 'N/A'))
                          for agent in self.agents if hasattr(agent, 'p')]
            avg_prob = sum(p for p in agent_probs if isinstance(p, (int, float))) / len(agent_probs) if agent_probs else 0
            print(f"Episode {episode_num}: Initial cost={initial_cost:.1f}, Avg p={avg_prob:.3f}")

        # Initialize agents
        self.initialize_agents()

        # Run episode iterations
        for iteration in range(self.iterations_per_episode):
            self.execute_iteration(iteration + 1)

            # Record cost
            current_cost = self.calculate_global_cost()
            episode_costs.append(current_cost)

            # Track p evolution
            for agent in self.agents:
                self.probability_evolution[agent.id_].append(agent.p)

        # DIAGNOSTIC: Check actions taken
        if episode_num < 3 or episode_num % 10 == 0:
            total_flips = sum(sum(1 for entry in getattr(agent, 'episode_data', [])
                                 if entry.get('did_flip', False)) for agent in self.agents)
            final_cost = episode_costs[-1]
            improvement = initial_cost - final_cost
            print(f"Episode {episode_num}: {total_flips} flips, Final cost={final_cost:.1f}, Improvement={improvement:.1f}")
        
        return episode_costs

    def finish_episode_learning(self, episode_costs: List[float], episode_count: int = 0) -> None:
        """Apply learning updates after episode completion."""

        # Calculate episode-level improvement (single reward signal)


        # Compute cross-agent feature statistics for normalization
        total_feature_magnitude = 0.0
        agent_count = 0
        for agent in self.agents:
            if hasattr(agent, 'episode_feature_magnitude') and agent.episode_feature_magnitude > 0:
                total_feature_magnitude += agent.episode_feature_magnitude
                agent_count += 1
        
        # Update average feature norm for all agents
        if agent_count > 0:
            average_feature_norm = total_feature_magnitude / agent_count
            for agent in self.agents:
                agent.average_feature_norm = average_feature_norm

        # Apply learning update to each agent with the same episode reward
        for agent in self.agents:
                # Pass episode count if agent supports it

            agent.finish_episode()

    def execute(self) -> List[float]:
        """Execute multiple episodes of reinforcement learning."""
        print(f"Starting DSA-RL training: {self.num_episodes} episodes, {self.iterations_per_episode} iterations each")
        
        final_costs = []
        
        for episode in range(self.num_episodes):
            # Prepare episode if using shared topology

            self.shared_topology.prepare_episode(episode)
            # Update for new episode
            self._setup_from_shared_topology()
            
            # Execute episode
            episode_costs = self.execute_single_episode(episode)
            
            # Apply learning
            self.finish_episode_learning(episode_costs, episode)
            
            # Track episode results
            self.all_episode_costs.append(episode_costs)
            final_costs = episode_costs  # Keep last episode costs
            
            # Progress reporting every 5 episodes and final episode
            if (episode + 1) % 5 == 0 or episode == self.num_episodes - 1:
                initial_cost = episode_costs[0]
                final_cost = episode_costs[-1]
                improvement = initial_cost - final_cost
                print(f"Running episode {episode + 1}/{self.num_episodes}: "
                      f"Initial={initial_cost:.1f}, Final={final_cost:.1f}, "
                      f"Improvement={improvement:.1f}")
        
        return final_costs
    
    def get_final_agent_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get final statistics for all agents after training."""
        stats = {}
        
        for agent in self.agents:
            agent_stats = {'p': getattr(agent, 'p', 0.5),
                           'baseline': getattr(agent, 'baseline', 0.0), 'policy_weights': agent.policy_weights}

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
    p: Optional[float] = None,
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
        edge_probability: Edge p for constraint graph
        p: Probability parameter for DSA
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
            edge_probability, p, agent_priority_config, shared_topology, current_episode
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
            p=p,
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
        k: Graph density (edge p)
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
                    dcop = create_selected_dcop(episode, algorithm, k, dcop_config.get('p', dcop_config.get('p', 0.5)), shared_topology, current_episode=episode)
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

