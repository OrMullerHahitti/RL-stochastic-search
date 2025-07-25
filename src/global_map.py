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
        "high": (15, 80, 20),  # 5 agents, μ=80, σ=20
        "medium": (15, 50, 20),  # 15 agents, μ=50, σ=5
        "low": (20, 20, 5),  # 10 agents, μ=20, σ=5
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
    "agents": 50,  # Number of agents (countries)
    "domain_size": 20,
    "repetitions": 30,  # Number of repetitions per algorithm
    "iterations": 100,  # Iterations per experiment run or episode length
    
    # DSA-RL specific hyperparameters - centralized single source of truth
    "dsa_rl": {
        "p0": 0.5,  # Initial probability for all agents
        "learning_rate": 0.005,  # REINFORCE learning rate (α)
        "baseline_decay": 0.99,  # Exponential moving average decay (β)
        "num_episodes": 30,  # Number of learning episodes (same as repetitions)
    },
    
    # Testing and debugging parameters
    "testing": {
        "max_cost_iterations": 100,  # Maximum iterations for cost tracking arrays
        "reduced_iterations": 20,  # Reduced iterations for quick testing
        "validation_epsilon": 1e-6,  # Tolerance for mathematical consistency checks
    },
}



DEFAULT_PRIORITY_VARIANT = MASTER_CONFIG["priority_variant"]['stratified']  # Default priority variant for DSA and MGM
repetitions = MASTER_CONFIG["repetitions"] # Number of repetitions per algorithm OR number of episodes per RL run
iteration_per_episode = MASTER_CONFIG["iterations"] # Number of iterations per experiment run or *episode length*
DEFAULT_AGENTS = MASTER_CONFIG["agents"]  # Fixed naming consistency: was DEFAULT_AGENTS_NUM
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


def get_master_config():
    """Get the complete master configuration - single source of truth for all parameters"""
    return MASTER_CONFIG.copy()  # Return copy to prevent accidental modifications

def get_dsa_rl_hyperparameters():
    """Get DSA-RL hyperparameters from centralized config"""
    return MASTER_CONFIG["dsa_rl"].copy()

def get_testing_parameters():
    """Get testing and debugging parameters from centralized config"""
    return MASTER_CONFIG["testing"].copy()

def create_test_config_override(**overrides):
    """Create a configuration override for testing with specific parameters"""
    test_config = get_master_config()
    
    # Apply overrides to the copied config
    for key, value in overrides.items():
        if key in test_config:
            test_config[key] = value
        elif key in test_config.get("testing", {}):
            test_config["testing"][key] = value
        elif key in test_config.get("dsa_rl", {}):
            test_config["dsa_rl"][key] = value
        else:
            # Add new test-specific parameters
            test_config[key] = value
    
    return test_config

def create_dsa_rl_config(agent_mu_config, override_params=None):
    """Create DSA-RL configuration using centralized hyperparameters with optional overrides"""
    dsa_rl_params = get_dsa_rl_hyperparameters()
    
    # Apply any override parameters
    if override_params:
        dsa_rl_params.update(override_params)
    
    return {
        "algorithm": Algorithm.DSA_RL,
        "p0": dsa_rl_params["p0"],
        "learning_rate": dsa_rl_params["learning_rate"],
        "baseline_decay": dsa_rl_params["baseline_decay"],
        "iteration_per_episode": MASTER_CONFIG["iterations"],  # Use global iterations
        "num_episodes": dsa_rl_params["num_episodes"],
        "agent_mu_config": agent_mu_config,
        "name": f'DSA_RL_{agent_mu_config}',
    }


# Default DSA and MGM configurations (for backward compatibility)
DSA_CONFIGS = create_dsa_configs(DEFAULT_PRIORITY_VARIANT)

MGM_CONFIG = create_mgm_config(DEFAULT_PRIORITY_VARIANT)

DSA_RL_CONFIG = create_dsa_rl_config(DEFAULT_PRIORITY_VARIANT)


# =============================================================================
# CONFIGURATION VALIDATION AND UTILITIES
# =============================================================================
def validate_master_config():
    """Validate that MASTER_CONFIG contains all required parameters"""
    required_keys = {
        "priority_variant", "graph_densities", "agents", "domain_size", 
        "repetitions", "iterations", "dsa_rl", "testing"
    }
    
    missing_keys = required_keys - set(MASTER_CONFIG.keys())
    if missing_keys:
        raise ValueError(f"MASTER_CONFIG missing required keys: {missing_keys}")
    
    # Validate DSA-RL parameters
    required_dsa_rl_keys = {"p0", "learning_rate", "baseline_decay", "num_episodes"}
    missing_dsa_rl_keys = required_dsa_rl_keys - set(MASTER_CONFIG["dsa_rl"].keys())
    if missing_dsa_rl_keys:
        raise ValueError(f"MASTER_CONFIG['dsa_rl'] missing required keys: {missing_dsa_rl_keys}")
    
    # Validate testing parameters
    required_testing_keys = {"max_cost_iterations", "reduced_iterations", "validation_epsilon"}
    missing_testing_keys = required_testing_keys - set(MASTER_CONFIG["testing"].keys())
    if missing_testing_keys:
        raise ValueError(f"MASTER_CONFIG['testing'] missing required keys: {missing_testing_keys}")
    
    return True

def get_current_experiment_config():
    """Get the current master configuration being used by ALL algorithms"""
    validate_master_config()  # Ensure config is valid before returning
    return {
        "priority_variant": MASTER_CONFIG["priority_variant"],
        "graph_densities": MASTER_CONFIG["graph_densities"],
        "agents": MASTER_CONFIG["agents"],
        "domain_size": MASTER_CONFIG["domain_size"],
        "repetitions": MASTER_CONFIG["repetitions"],
        "iterations": MASTER_CONFIG["iterations"],
        "dsa_rl": MASTER_CONFIG["dsa_rl"],
        "testing": MASTER_CONFIG["testing"],
    }


def print_master_config():
    """Print the master configuration that ALL algorithms will use"""
    config = get_current_experiment_config()
    print("=" * 80)
    print("MASTER CONFIGURATION - Applied to ALL Algorithms")
    print("=" * 80)
    print(f"Priority Variant:    {list(config['priority_variant'].keys())}")
    print(f"Graph Densities:     {config['graph_densities']}")
    print(f"Agents:              {config['agents']}")
    print(f"Domain Size:         {config['domain_size']}")
    print(f"Repetitions:         {config['repetitions']}")
    print(f"Iterations:          {config['iterations']}")
    print()
    print("DSA-RL Hyperparameters:")
    for key, value in config['dsa_rl'].items():
        print(f"  {key}:             {value}")
    print()
    print("Testing Parameters:")
    for key, value in config['testing'].items():
        print(f"  {key}:        {value}")
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
