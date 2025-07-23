from enum import Enum
import os

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
# Basic experiment parameters with environment variable override support
repetitions = int(os.environ.get('DSA_REPETITIONS', 30))
incomplete_iterations = int(os.environ.get('DSA_ITERATIONS', 100))

# Problem instance parameters
DEFAULT_AGENTS = int(os.environ.get('DSA_AGENTS', 30))
DEFAULT_DOMAIN_SIZE = int(os.environ.get('DSA_DOMAIN_SIZE', 10))
DEFAULT_GRAPH_DENSITIES = [0.2, 0.7]  # k values for sparse/dense graphs

# =============================================================================
# AGENT PRIORITY CONFIGURATIONS
# =============================================================================
# Predefined priority configurations for different scenarios
DEFAULT_PRIORITY_CONFIG = {
    'default_mu': 50,
    'default_sigma': 10
}

HIERARCHICAL_PRIORITY_CONFIG = {
    'default_mu': 50,
    'default_sigma': 10,
    'hierarchical': {
        'high': (1, 10, 80),    # agents 1-10: high priority
        'medium': (11, 20, 50), # agents 11-20: medium priority  
        'low': (21, 30, 20)     # agents 21-30: low priority
    }
}

MANUAL_PRIORITY_CONFIG = {
    'default_mu': 50,
    'default_sigma': 10,
    'manual': {
        1: 90, 2: 85, 3: 80,  # VIP agents
        28: 30, 29: 25, 30: 20  # Low priority agents
    }
}

STRATIFIED_PRIORITY_CONFIG = {
    'default_mu': 50,
    'default_sigma': 10,
    'random_stratified': {
        'high': (5, 80, 20),    # 5 agents, μ=80, σ=5
        'medium': (15, 50, 20), # 15 agents, μ=50, σ=5
        'low': (10, 20, 5)     # 10 agents, μ=20, σ=5
    }
}

# Define a class to represent messages exchanged between agents
class Msg():
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
# ALGORITHM CONFIGURATIONS
# =============================================================================
# DSA configurations
DSA_CONFIGS = [
    {'algorithm': Algorithm.DSA, 'p': 0.2, 'name': 'DSA_p02'},
    {'algorithm': Algorithm.DSA, 'p': 0.7, 'name': 'DSA_p07'},
    {'algorithm': Algorithm.DSA, 'p': 1.0, 'name': 'DSA_p10'}
]

# MGM configuration
MGM_CONFIG = {'algorithm': Algorithm.MGM, 'name': 'MGM'}

# DSA-RL configurations with different priority settings
DSA_RL_CONFIGS = {
    'uniform': {
        'algorithm': Algorithm.DSA_RL,
        'p0': 0.5,
        'learning_rate': 0.01,
        'baseline_decay': 0.9,
        'episode_length': 20,
        'agent_mu_config': DEFAULT_PRIORITY_CONFIG,
        'name': 'DSA_RL_Uniform'
    },
    'hierarchical': {
        'algorithm': Algorithm.DSA_RL,
        'p0': 0.5,
        'learning_rate': 0.01,
        'baseline_decay': 0.9,
        'episode_length': 20,
        'agent_mu_config': HIERARCHICAL_PRIORITY_CONFIG,
        'name': 'DSA_RL_Hierarchical'
    },
    'manual': {
        'algorithm': Algorithm.DSA_RL,
        'p0': 0.5,
        'learning_rate': 0.01,
        'baseline_decay': 0.9,
        'episode_length': 20,
        'agent_mu_config': MANUAL_PRIORITY_CONFIG,
        'name': 'DSA_RL_Manual'
    },
    'stratified': {
        'algorithm': Algorithm.DSA_RL,
        'p0': 0.5,
        'learning_rate': 0.01,
        'baseline_decay': 0.9,
        'episode_length': 20,
        'agent_mu_config': STRATIFIED_PRIORITY_CONFIG,
        'name': 'DSA_RL_Stratified'
    }
}

# =============================================================================
# EXPERIMENT SCENARIOS
# =============================================================================
# Standard comparison experiment
STANDARD_EXPERIMENT = DSA_CONFIGS + [MGM_CONFIG, DSA_RL_CONFIGS['hierarchical']]

# Priority comparison experiment
PRIORITY_COMPARISON_EXPERIMENT = [
    DSA_RL_CONFIGS['uniform'],
    DSA_RL_CONFIGS['hierarchical'], 
    DSA_RL_CONFIGS['manual'],
    DSA_RL_CONFIGS['stratified']
]

# Full comparison experiment
FULL_EXPERIMENT = DSA_CONFIGS + [MGM_CONFIG] + list(DSA_RL_CONFIGS.values())

# Minimal experiment for testing (faster execution)
MINIMAL_EXPERIMENT = [DSA_CONFIGS[1], MGM_CONFIG, DSA_RL_CONFIGS['hierarchical']]

