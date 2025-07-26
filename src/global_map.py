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
    "default_edge_probability": 0.3,  # Default edge probability when not specified
    "agents": 60,  # Number of agents (countries) - large scale for meaningful probability learning
    "domain_size": 12,  # Domain size - balanced for challenging but solvable large-scale problems
    "repetitions": 30,  # Number of repetitions per algorithm
    "iterations": 100,  # Iterations per experiment run or episode length
    
    # DSA-RL specific hyperparameters - centralized single source of truth
    "dsa_rl": {
        "p0": 0.5,  # Initial probability for all agents
        "learning_rate": 0.005,  # Actor learning rate (reduced for large-scale stability)
        "baseline_decay": 0.99,  # Exponential moving average decay (β)
        "num_episodes": 50,  # Number of learning episodes (increased for large-scale learning)
        "gamma": 0.9,  # Discount factor for future rewards
        "num_global_features": 6,  # Number of global features for critic
        "critic_init_std": 0.2,  # Standard deviation for critic weight initialization
        "baseline_agents": 30,  # Baseline number of agents for scaling
        "curriculum_easy_threshold": 3,  # Agent-to-color ratio threshold for easy problems
        "curriculum_medium_threshold": 6,  # Agent-to-color ratio threshold for medium problems
        "curriculum_progress_ratio": 0.3,  # Fraction of episodes for curriculum ramp-up
        "curriculum_rate_multipliers": {  # Learning rate multipliers by curriculum stage
            "easy": 1.2,
            "medium": 1.0,
            "hard": 0.8
        },
        "critic_weight_clip_bounds": 100.0,  # Bounds for clipping critic weights
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

def get_rl_lifecycle_config():
    """Get RL lifecycle configuration from centralized config"""
    return MASTER_CONFIG["rl_lifecycle"].copy()

def get_learning_phase_config():
    """Get learning phase configuration"""
    return MASTER_CONFIG["rl_lifecycle"]["learning_phase"].copy()

def get_comparison_phase_config():
    """Get comparison phase configuration"""
    return MASTER_CONFIG["rl_lifecycle"]["comparison_phase"].copy()

def should_use_synchronized_experiments():
    """Check if synchronized experiments should always be used"""
    return MASTER_CONFIG["rl_lifecycle"]["always_use_synchronized"]

def should_validate_consistency():
    """Check if consistency validation is enabled"""
    return MASTER_CONFIG["rl_lifecycle"]["validate_consistency"]

def should_log_phase_transitions():
    """Check if phase transition logging is enabled"""
    return MASTER_CONFIG["rl_lifecycle"]["log_phase_transitions"]

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
        "name": f'DSA_RL',
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
        "repetitions", "iterations", "dsa_rl", "testing", "rl_lifecycle"
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
    
    # Validate RL lifecycle parameters
    required_rl_lifecycle_keys = {"learning_phase", "comparison_phase", "always_use_synchronized", "validate_consistency", "log_phase_transitions"}
    missing_rl_lifecycle_keys = required_rl_lifecycle_keys - set(MASTER_CONFIG["rl_lifecycle"].keys())
    if missing_rl_lifecycle_keys:
        raise ValueError(f"MASTER_CONFIG['rl_lifecycle'] missing required keys: {missing_rl_lifecycle_keys}")
    
    # Validate learning phase parameters
    required_learning_keys = {"vary_penalties", "vary_initial_assignments", "constant_graph_structure", "constant_density_functions", "use_shared_topology", "mode_name"}
    missing_learning_keys = required_learning_keys - set(MASTER_CONFIG["rl_lifecycle"]["learning_phase"].keys())
    if missing_learning_keys:
        raise ValueError(f"MASTER_CONFIG['rl_lifecycle']['learning_phase'] missing required keys: {missing_learning_keys}")
    
    # Validate comparison phase parameters
    required_comparison_keys = {"vary_penalties", "vary_initial_assignments", "constant_graph_structure", "constant_density_functions", "use_shared_topology", "mode_name"}
    missing_comparison_keys = required_comparison_keys - set(MASTER_CONFIG["rl_lifecycle"]["comparison_phase"].keys())
    if missing_comparison_keys:
        raise ValueError(f"MASTER_CONFIG['rl_lifecycle']['comparison_phase'] missing required keys: {missing_comparison_keys}")
    
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
        "rl_lifecycle": MASTER_CONFIG["rl_lifecycle"],
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
    print()
    print("RL Lifecycle Configuration:")
    print(f"  Always Use Synchronized: {config['rl_lifecycle']['always_use_synchronized']}")
    print(f"  Validate Consistency:    {config['rl_lifecycle']['validate_consistency']}")
    print(f"  Log Phase Transitions:   {config['rl_lifecycle']['log_phase_transitions']}")
    print("  Learning Phase:")
    for key, value in config['rl_lifecycle']['learning_phase'].items():
        print(f"    {key}:   {value}")
    print("  Comparison Phase:")
    for key, value in config['rl_lifecycle']['comparison_phase'].items():
        print(f"    {key}: {value}")
    print("=" * 80)
    print("All algorithms (DSA, MGM, DSA-RL) will use these EXACT parameters")
    print("RL learning and comparison phases are now explicitly controlled")
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
