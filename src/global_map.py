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

HIERARCHICAL_PRIORITY_CONFIG = {
    "default_mu": 50,
    "default_sigma": 10,
    "hierarchical": {
        "high": (1, 10, 80),  # agents 1-10: high priority
        "medium": (11, 20, 50),  # agents 11-20: medium priority
        "low": (21, 30, 20),  # agents 21-30: low priority
    },
}

MANUAL_PRIORITY_CONFIG = {
    "default_mu": 50,
    "default_sigma": 10,
    "manual": {
        1: 90,
        2: 85,
        3: 80,  # VIP agents
        28: 30,
        29: 25,
        30: 20,  # Low priority agents
    },
}

STRATIFIED_PRIORITY_CONFIG = {
    "default_mu": 50,
    "default_sigma": 10,
    "random_stratified": {
        "high": (5, 80, 20),  # 5 agents, μ=80, σ=20
        "medium": (15, 50, 20),  # 15 agents, μ=50, σ=5
        "low": (10, 20, 5),  # 10 agents, μ=20, σ=5
    },
}


MASTER_CONFIG = {
    # Priority configuration variant - affects ALL algorithms in comparative experiments
    # This determines the penalty distribution (μ values) for constraint costs
    "priority_variant": {
        "uniform": DEFAULT_PRIORITY_CONFIG,
        "hierarchical": HIERARCHICAL_PRIORITY_CONFIG,
        "manual": MANUAL_PRIORITY_CONFIG,
        "stratified": STRATIFIED_PRIORITY_CONFIG,
    },  # Options: 'uniform', 'hierarchical', 'manual', 'stratified'
    # Graph topology parameters - used by ALL algorithms
    # These k values determine edge probability in constraint graphs
    "graph_densities": [0.2, 0.7],  # k values for sparse/dense graphs
    # Problem size parameters - consistent across ALL algorithms
    "agents": 30,  # Number of agents (countries)
    "domain_size": 10,
    "repetitions": 30,  # Number of repetitions per algorithm
    "iterations":  100,
      # Iterations per experiment run
}



DEFAULT_PRIORITY_VARIANT = MASTER_CONFIG["priority_variant"]['stratified']  # Default priority variant for DSA and MGM
repetitions = MASTER_CONFIG["repetitions"] # Number of repetitions per algorithm OR number of episodes per RL run
iteration_per_episode = MASTER_CONFIG["iterations"] # Number of iterations per experiment run or *episode length*
DEFAULT_AGENTS_NUM = MASTER_CONFIG["agents"]
DEFAULT_DOMAIN_SIZE = MASTER_CONFIG["domain_size"]
DEFAULT_GRAPH_DENSITIES = MASTER_CONFIG["graph_densities"]




# =============================================================================
# ALGORITHM CONFIGURATIONS
# =============================================================================
# Function to create DSA configs with specified priority configuration
def create_dsa_configs(agent_mu_config):
    return [
        {
            "algorithm": Algorithm.DSA,
            "p": 0.2,
            "name": "DSA_p02",
            "agent_mu_config": agent_mu_config,
        },
        {
            "algorithm": Algorithm.DSA,
            "p": 0.7,
            "name": "DSA_p07",
            "agent_mu_config": agent_mu_config,
        },
        {
            "algorithm": Algorithm.DSA,
            "p": 1.0,
            "name": "DSA_p10",
            "agent_mu_config": agent_mu_config,
        },
    ]


# Function to create MGM config with specified priority configuration
def create_mgm_config(agent_mu_config):
    return {
        "algorithm": Algorithm.MGM,
        "name": "MGM",
        "agent_mu_config": agent_mu_config,
    }


def create_dsa_rl_config(
    agent_mu_config, p0, learning_rate, baseline_decay, iteration_per_episode
):
    """Create DSA-RL configuration with specified parameters"""
    return {
        "algorithm": Algorithm.DSA_RL,
        "p0": p0,
        "learning_rate": learning_rate,
        "baseline_decay": baseline_decay,
        "iteration_per_episode": iteration_per_episode,
        "agent_mu_config": agent_mu_config,
        "name": f'DSA_RL_{agent_mu_config["default_mu"]}',
    }


# Default DSA and MGM configurations (for backward compatibility)
DSA_CONFIGS = create_dsa_configs(DEFAULT_PRIORITY_VARIANT)

MGM_CONFIG = create_mgm_config(DEFAULT_PRIORITY_VARIANT)

DSA_RL_CONFIG = create_dsa_rl_config(
    DEFAULT_PRIORITY_VARIANT, 0.5, 0.01, 0.9, iteration_per_episode
)


# =============================================================================
# CONFIGURATION VALIDATION AND UTILITIES
# =============================================================================
def get_current_experiment_config():
    """Get the current master configuration being used by ALL algorithms"""
    return {
        "priority_variant": MASTER_CONFIG["priority_variant"],
        "graph_densities": MASTER_CONFIG["graph_densities"],
        "agents": MASTER_CONFIG["agents"],
        "domain_size": MASTER_CONFIG["domain_size"],
        "repetitions": MASTER_CONFIG["repetitions"],
        "iterations": MASTER_CONFIG["iterations"],
    }


def print_master_config():
    """Print the master configuration that ALL algorithms will use"""
    config = get_current_experiment_config()
    print("=" * 80)
    print("MASTER CONFIGURATION - Applied to ALL Algorithms")
    print("=" * 80)
    print(f"Priority Variant:    {config['priority_variant']}")
    print(f"Graph Densities:     {config['graph_densities']}")
    print(f"Agents:              {config['agents']}")
    print(f"Domain Size:         {config['domain_size']}")
    print(f"Repetitions:         {config['repetitions']}")
    print(f"Iterations:          {config['iterations']}")
    print("=" * 80)
    print("All algorithms (DSA, MGM, DSA-RL) will use these EXACT parameters")
    print("=" * 80)


def validate_experiment_consistency(experiment_configs):
    """Validate that all algorithms in an experiment use identical core parameters"""
    master_priority = MASTER_CONFIG["priority_variant"]
    expected_priority_config = DSA_RL_CONFIGS[master_priority]["agent_mu_config"]

    for config in experiment_configs:
        if "agent_mu_config" in config:
            if config["agent_mu_config"] != expected_priority_config:
                raise ValueError(
                    f"Algorithm {config.get('name', 'Unknown')} has inconsistent priority config!"
                )

    return True


# =============================================================================
# EXPERIMENT SCENARIOS - ALL USE MASTER_CONFIG
# =============================================================================
# Function to create standard experiment with consistent priority config
def create_standard_experiment(dsa_rl_variant=None):
    """Create standard experiment where all algorithms use the MASTER_CONFIG priority variant"""
    if dsa_rl_variant is None:
        dsa_rl_variant = MASTER_CONFIG["priority_variant"]

    dsa_rl_config = DSA_RL_CONFIGS[dsa_rl_variant]
    priority_config = dsa_rl_config["agent_mu_config"]

    configs = create_dsa_configs(priority_config) + [
        create_mgm_config(priority_config),
        dsa_rl_config,
    ]
    validate_experiment_consistency(configs)
    return configs


# Function to create full experiment with consistent priority config
def create_full_experiment(dsa_rl_variant=None):
    """Create full experiment with DSA, MGM using MASTER_CONFIG, plus ALL DSA-RL variants"""
    if dsa_rl_variant is None:
        dsa_rl_variant = MASTER_CONFIG["priority_variant"]

    dsa_rl_config = DSA_RL_CONFIGS[dsa_rl_variant]
    priority_config = dsa_rl_config["agent_mu_config"]

    # For full experiment: DSA and MGM use master config, but include all DSA-RL variants
    # (DSA-RL variants are allowed to have different priority configs for comparison)
    configs = (
        create_dsa_configs(priority_config)
        + [create_mgm_config(priority_config)]
        + list(DSA_RL_CONFIGS.values())
    )

    # Only validate consistency for DSA and MGM (not DSA-RL variants in full experiment)
    dsa_mgm_configs = create_dsa_configs(priority_config) + [
        create_mgm_config(priority_config)
    ]
    validate_experiment_consistency(dsa_mgm_configs)

    return configs


# Function to create minimal experiment with consistent priority config
def create_minimal_experiment(dsa_rl_variant=None):
    """Create minimal experiment with one DSA, MGM, and DSA_RL using MASTER_CONFIG priority variant"""
    if dsa_rl_variant is None:
        dsa_rl_variant = MASTER_CONFIG["priority_variant"]

    dsa_rl_config = DSA_RL_CONFIGS[dsa_rl_variant]
    priority_config = dsa_rl_config["agent_mu_config"]

    configs = [
        create_dsa_configs(priority_config)[1],
        create_mgm_config(priority_config),
        dsa_rl_config,
    ]
    validate_experiment_consistency(configs)
    return configs


# =============================================================================
# EXPERIMENT CONFIGURATIONS - AUTO-GENERATED FROM MASTER_CONFIG
# =============================================================================
# These are automatically generated using MASTER_CONFIG settings
# WARNING: Do not modify these - change MASTER_CONFIG instead

# Default experiment configurations - DSA, MGM, and DSA-RL comparison
STANDARD_EXPERIMENT = DSA_CONFIGS + [MGM_CONFIG, DSA_RL_CONFIG]

# Priority comparison experiment (DSA-RL only - not implemented yet)
PRIORITY_COMPARISON_EXPERIMENT = [DSA_RL_CONFIG]

# Full comparison experiment (all algorithms)
FULL_EXPERIMENT = DSA_CONFIGS + [MGM_CONFIG, DSA_RL_CONFIG]

# Minimal experiment for testing (one of each type)
MINIMAL_EXPERIMENT = [DSA_CONFIGS[1], MGM_CONFIG, DSA_RL_CONFIG]  # DSA p=0.7, MGM, DSA-RL

# =============================================================================
# USAGE SUMMARY - HOW TO USE THE CENTRALIZED CONFIGURATION SYSTEM
# =============================================================================
