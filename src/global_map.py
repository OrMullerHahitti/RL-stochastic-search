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
    "priority_variant": {
        "uniform": DEFAULT_PRIORITY_CONFIG,
        "manual": MANUAL_PRIORITY_CONFIG,
        "stratified": STRATIFIED_PRIORITY_CONFIG,
    },
    "graph_densities": [0.2, 0.7],
    "default_edge_probability": 0.3,
    "agents": 30,
    "domain_size": 4,
    "repetitions": 30,
    "iterations": 100,
    
    "dsa_rl": {
        "p0": 0.5,
        "learning_rate": 0.05,
        "baseline_decay": 0.9,
        "num_episodes": 50,
        "gamma": 0.9,
        "critic_init_std": 0.2,
        "baseline_agents": 30,
    },
    
    # Testing and debugging parameters
    "testing": {
        "max_cost_iterations": 100,
        "reduced_iterations": 20,
        "validation_epsilon": 1e-6,
        "random_seed_offset": 42,
    },
    
    "rl_lifecycle": {
        "learning_phase": {
            "vary_penalties": True,
            "vary_initial_assignments": True,
            "constant_graph_structure": True,
            "constant_density_functions": True,
            "use_shared_topology": True,
            "mode_name": "learning"
        },
        "comparison_phase": {
            "vary_penalties": False,
            "vary_initial_assignments": False,
            "constant_graph_structure": True,
            "constant_density_functions": True,
            "use_shared_topology": True,
            "mode_name": "comparison"
        },
        "always_use_synchronized": True,
        "validate_consistency": True,
        "log_phase_transitions": True
    },
}


def get_master_config():
    """Get the complete master configuration"""
    return MASTER_CONFIG.copy()  # Return copy to prevent accidental modifications


def get_dsa_rl_hyperparameters():
    """Get DSA-RL hyperparameters"""
    return MASTER_CONFIG["dsa_rl"].copy()


def get_testing_parameters():
    """Get testing and debugging parameters"""
    return MASTER_CONFIG["testing"].copy()


def get_learning_phase_config():
    """Get learning phase configuration"""
    return MASTER_CONFIG["rl_lifecycle"]["learning_phase"].copy()


def get_comparison_phase_config():
    """Get comparison phase configuration"""
    return MASTER_CONFIG["rl_lifecycle"]["comparison_phase"].copy()


def should_log_phase_transitions():
    """Check if phase transition logging is enabled"""
    return False
