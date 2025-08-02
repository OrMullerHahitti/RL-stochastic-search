from enum import Enum
import os


# Define a class to represent messages exchanged between agents
class Msg:
    def __init__(self, sender, receiver, information):
        self.sender = sender  # The agent sending the message
        self.receiver = receiver  # The agent receiving the message
        self.information = information  # The information contained in the message


# Define an enumeration for the algorithms used in DCOPs
class Algorithm(Enum):
    DSA = 1  # DSA algorithm
    MGM = 2  # MGM algorithm
    DSA_RL = 3  # DSA with REINFORCE learning

# =============================================================================
# MASTER EXPERIMENT CONFIGURATION - CHANGE ONLY HERE
# =============================================================================
DEFAULT_PRIORITY_CONFIG = {"default_mu": 50, "default_sigma": 10}

MANUAL_PRIORITY_CONFIG = {
    "default_mu": 50,
    "default_sigma": 10,
    "manual": {i: (1, 50 + i * 50, 10) for i in range(1, 31)},
}

STRATIFIED_PRIORITY_CONFIG = {
    "default_mu": 50,
    "default_sigma": 10,
    "random_stratified": {
        "high": (10, 150, 10),
        "medium": (10, 100, 10),
        "low": (10, 50, 10),
    },
}


MASTER_CONFIG = {
    # Priority configuration variant - affects ALL algorithms in comparative experiments
    # This determines the penalty distribution (μ values) for constraint costs
    "priority_variant": {
        "uniform": DEFAULT_PRIORITY_CONFIG,
        "manual": MANUAL_PRIORITY_CONFIG,
        "stratified": STRATIFIED_PRIORITY_CONFIG,
    },
    "graph_densities": [0.2, 0.7],  # k values for sparse/dense graphs
    "default_edge_probability": 0.3,  # Default edge p when not specified
    "agents": 30,  # Number of agents (countries) - large scale for meaningful p learning
    "domain_size": 4,  # Domain size - balanced for challenging but solvable large-scale problems
    "repetitions": 30,  # Number of repetitions per algorithm
    "iterations": 100,  # Iterations per experiment run or episode length
    
    # DSA-RL specific hyperparameters - centralized single source of truth
    "dsa_rl": {
        "p0": 0.5,  # Initial p for all agents
        "learning_rate": 0.05,  # Actor learning rate (reduced for large-scale stability)
        "baseline_decay": 0.9,  # Exponential moving average decay (bethe)
        "num_episodes": 50,  # Number of learning episodes (increased for large-scale learning)
        "gamma": 0.9,  # Discount factor for future rewards
        "critic_init_std": 0.2,  # Standard deviation for critic weight initialization
        "baseline_agents": 30,  # Baseline number of agents for scaling
    },
    
    # Testing and debugging parameters
    "testing": {
        "max_cost_iterations": 100,  # Maximum iterations for cost tracking arrays
        "reduced_iterations": 20,  # Reduced iterations for quick testing
        "validation_epsilon": 1e-6,  # Tolerance for mathematical consistency checks
        "random_seed_offset": 42,  # Seed offset for deterministic random generation
    },
    
    # RL Lifecycle Configuration - Controls learning vs comparison phase behavior
    "rl_lifecycle": {
        # Learning Phase: DSA-RL training on varied problem instances
        "learning_phase": {
            "vary_penalties": True,          # Allow penalty variation for robust learning
            "vary_initial_assignments": True, # Allow initial assignment variation  
            "constant_graph_structure": True, # Always maintain same neighbor relationships
            "constant_density_functions": True, # Always maintain agent penalty distribution parameters
            "use_shared_topology": True,     # Use SharedGraphTopology for consistency
            "mode_name": "learning"          # Mode identifier for SharedGraphTopology
        },
        # Comparison Phase: Fair evaluation across all algorithms
        "comparison_phase": {
            "vary_penalties": False,         # Fixed penalties across all algorithms per repetition
            "vary_initial_assignments": False, # Fixed initial assignments across all algorithms per repetition
            "constant_graph_structure": True, # Always maintain same neighbor relationships
            "constant_density_functions": True, # Always maintain agent penalty distribution parameters  
            "use_shared_topology": True,     # Use SharedGraphTopology for consistency
            "mode_name": "comparison"        # Mode identifier for SharedGraphTopology
        },
        # Global lifecycle settings
        "always_use_synchronized": True,    # Always use synchronized experiments (replaces hardcoded USE_SYNCHRONIZED)
        "validate_consistency": True,       # Enable validation checks for graph/penalty consistency
        "log_phase_transitions": True       # Log when switching between learning and comparison phases
    },
}


def get_master_config():
    """Get the complete master configuration - single source of truth for all parameters"""
    return MASTER_CONFIG.copy()  # Return copy to prevent accidental modifications


def get_dsa_rl_hyperparameters():
    """Get DSA-RL hyperparameters from centralized config"""
    return MASTER_CONFIG["dsa_rl"].copy()


def get_testing_parameters():
    """Get testing and debugging parameters from centralized config"""
    return MASTER_CONFIG["testing"].copy()


def get_learning_phase_config():
    """Get learning phase configuration"""
    return MASTER_CONFIG["rl_lifecycle"]["learning_phase"].copy()


def get_comparison_phase_config():
    """Get comparison phase configuration"""
    return MASTER_CONFIG["rl_lifecycle"]["comparison_phase"].copy()


def should_log_phase_transitions():
    """Check if phase transition logging is enabled"""
    return False  # Disabled to reduce verbosity
