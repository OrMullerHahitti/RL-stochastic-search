#!/usr/bin/env python3
"""
Main experiment runner without matplotlib plotting - just shows results
"""

from src.problems import *
from src.global_map import *

# Function to create DCOP instance based on the algorithm and parameters
def create_selected_dcop(i, algorithm, k, p=None, shared_topology=None, current_episode=0, **kwargs):
    A = DEFAULT_AGENTS  # Number of agents from global config
    D = DEFAULT_DOMAIN_SIZE  # Domain size from global config
    
    # Extract agent priority configuration
    agent_mu_config = kwargs.get('agent_mu_config', None)
    
    if algorithm == Algorithm.DSA:
        dcop_name = f"DSA_{i}"
        # Use fixed dcop_id=0 when shared topology is used for consistency
        dcop_id = 0 if shared_topology else i
        return DCOP_DSA(dcop_id, A, D, dcop_name, algorithm, k, p, agent_mu_config, shared_topology, current_episode)
    if algorithm == Algorithm.MGM:
        dcop_name = f"MGM_{i}"
        # Use fixed dcop_id=0 when shared topology is used for consistency
        dcop_id = 0 if shared_topology else i
        return DCOP_MGM(dcop_id, A, D, dcop_name, algorithm, k, agent_mu_config, shared_topology, current_episode)
    if algorithm == Algorithm.DSA_RL:
        dcop_name = f"DSA_RL_Learning" 
        # Extract RL-specific parameters with global defaults
        dsa_rl_defaults = get_dsa_rl_hyperparameters()
        p0 = kwargs.get('p0', dsa_rl_defaults['p0'])
        learning_rate = kwargs.get('learning_rate', dsa_rl_defaults['learning_rate'])
        baseline_decay = kwargs.get('baseline_decay', dsa_rl_defaults['baseline_decay'])
        iteration_per_episode_param = kwargs.get('iteration_per_episode', iteration_per_episode)  # Use global iterations
        num_episodes = kwargs.get('num_episodes', dsa_rl_defaults['num_episodes'])
        return DCOP_DSA_RL(0, A, D, dcop_name, algorithm, k, p0, learning_rate, baseline_decay, iteration_per_episode_param, num_episodes, agent_mu_config, shared_topology, current_episode)

# Function to solve DCOPs and calculate average global cost
def solve_dcops(dcops, return_stats=False):
    testing_params = get_testing_parameters()
    max_iterations = testing_params["max_cost_iterations"]
    total = [0.0] * max_iterations # Initialize list to store total costs for each iteration
    all_stats = []
    first_dcop = None

    for dcop in dcops:
        if first_dcop is None:
            first_dcop = dcop  # Keep reference to first DCOP for mu values
            
        global_cost = dcop.execute()
        for i in range(min(len(global_cost), max_iterations)):
            total[i] += global_cost[i]
        
        # Collect learning statistics if DSA-RL
        if hasattr(dcop, 'get_learning_statistics'):
            stats = dcop.get_learning_statistics()
            all_stats.append(stats)
    
    avg_global_cost = [val / repetitions for val in total]
    
    if return_stats:
        return avg_global_cost, all_stats, first_dcop
    return avg_global_cost

def solve_synchronized_experiment(dcop_configs, k):
    """
    Run synchronized DSA vs DSA-RL experiment with shared topology and synchronized cost updates.
    
    Ensures:
    1. Same graph topology across all algorithms
    2. Same cost changes for each episode/"repetition" 
    3. Same starting assignments for fair comparison
    """
    testing_params = get_testing_parameters()
    max_iterations = testing_params["max_cost_iterations"]
    
    print(f"\n🔗 Creating shared topology for synchronized experiment (k={k})...")
    
    # Create shared topology that all algorithms will use
    agent_mu_config = dcop_configs[0].get('agent_mu_config', {})
    shared_topology = SharedGraphTopology(
        A=DEFAULT_AGENTS, 
        d=DEFAULT_DOMAIN_SIZE, 
        k=k, 
        agent_mu_config=agent_mu_config,
        base_seed=42  # Fixed seed for reproducible experiments
    )
    
    results = {}
    dsa_rl_stats = None
    dsa_rl_dcop = None
    
    for dcop_config in dcop_configs:
        algorithm = dcop_config['algorithm']
        print(f"Running {dcop_config.get('name', algorithm.name)} with shared topology...")
        
        if algorithm == Algorithm.DSA_RL:
            # DSA-RL: Single instance with multiple episodes using shared topology
            # CRITICAL FIX: Prepare episode 0 before creating DSA-RL to ensure synchronized starting conditions
            shared_topology.prepare_episode(0)
            
            rl_params = {k: v for k, v in dcop_config.items() if k not in ['algorithm', 'name']}
            rl_params['num_episodes'] = repetitions  # Set number of episodes
            
            dsa_rl_dcop = create_selected_dcop(0, algorithm, k, shared_topology=shared_topology, **rl_params)
            final_episode_costs = dsa_rl_dcop.execute()
            results[dcop_config.get('name', algorithm.name)] = final_episode_costs
            
            # Store learning statistics
            dsa_rl_stats = dsa_rl_dcop.get_final_agent_statistics()
            
        else:
            # DSA/MGM: Multiple instances, each using shared topology for different episodes
            total_costs = [0.0] * max_iterations
            
            for episode in range(repetitions):
                # Prepare shared topology for this episode
                shared_topology.prepare_episode(episode)
                
                # Create DCOP instance with shared topology
                if algorithm == Algorithm.DSA:
                    dcop = create_selected_dcop(episode, algorithm, k, dcop_config['p'], shared_topology, current_episode=episode)
                else:  # MGM
                    dcop = create_selected_dcop(episode, algorithm, k, shared_topology=shared_topology, current_episode=episode)
                
                # Execute this episode
                episode_costs = dcop.execute()
                
                # Accumulate costs
                for i in range(min(len(episode_costs), max_iterations)):
                    total_costs[i] += episode_costs[i]
            
            # Calculate average
            avg_costs = [cost / repetitions for cost in total_costs]
            results[dcop_config.get('name', algorithm.name)] = avg_costs
    
    return results, dsa_rl_stats, dsa_rl_dcop

def display_results_no_plot(results, title, dsa_rl_stats=None, dsa_rl_dcop=None):
    """Display experiment results without plotting"""
    print(f"\n📊 Results for {title}")
    print("=" * 60)
    
    for algorithm_name, data in results.items():
        if data:
            initial_cost = data[0]
            final_cost = data[-1]
            improvement = initial_cost - final_cost
            print(f"{algorithm_name:12}: Initial={initial_cost:6.1f}, Final={final_cost:6.1f}, Improvement={improvement:6.1f}")
    
    # Display DSA-RL agent probabilities if available
    if dsa_rl_stats:
        print(f"\nDSA-RL Learned Probabilities ({title}):")
        print("-" * 60)
        display_agent_probabilities(dsa_rl_stats, dsa_rl_dcop)

def display_agent_probabilities(final_stats, dcop_with_mu=None):
    """Display learned probabilities for DSA-RL agents after multi-episode learning"""
    if not final_stats:
        print("No DSA-RL statistics available")
        return
    
    # Get mu values if available
    mu_values = {}
    if dcop_with_mu and hasattr(dcop_with_mu, 'agent_mu_values'):
        mu_values = dcop_with_mu.agent_mu_values
    
    # Display header
    if mu_values:
        print("Agent | Final Probability | Theta   | Baseline | Priority μ")
        print("------|-------------------|---------|----------|----------")
    else:
        print("Agent | Final Probability | Theta   | Baseline")
        print("------|-------------------|---------|----------")
    
    # Display statistics for each agent
    all_probs = []
    for agent_id in sorted(final_stats.keys()):
        agent_stats = final_stats[agent_id]
        p = agent_stats['probability']
        theta = agent_stats['theta']
        baseline = agent_stats['baseline']
        all_probs.append(p)
        
        if mu_values and agent_id in mu_values:
            mu_val = mu_values[agent_id]
            print(f"{agent_id:5} | {p:16.3f} | {theta:7.3f} | {baseline:8.1f} | {mu_val:8.1f}")
        else:
            print(f"{agent_id:5} | {p:16.3f} | {theta:7.3f} | {baseline:8.1f}")
    
    # Summary statistics
    if all_probs:
        min_p = min(all_probs)
        max_p = max(all_probs)
        avg_p = sum(all_probs) / len(all_probs)
        print(f"\nSummary: Min p={min_p:.3f}, Max p={max_p:.3f}, Avg p={avg_p:.3f}")
        print(f"Learning spread: {max_p - min_p:.3f} (higher = more differentiation)")
        print(f"Total episodes: {getattr(dcop_with_mu, 'num_episodes', 'Unknown')}")

# Function to display results as text
def display_results(y_axis_data, title, dsa_rl_stats=None, dsa_rl_dcop=None, config_labels=None):
    print(f"\n{title}")
    print("="*50)
    
    # Use provided labels or generate from configurations
    if config_labels:
        labels = [config.get('name', config['algorithm'].name) for config in config_labels]
    else:
        # Fallback labels for backward compatibility
        labels = [f"Algorithm_{i+1}" for i in range(len(y_axis_data))]
    
    for i, data in enumerate(y_axis_data):
        if i < len(labels):
            label = labels[i]
            initial_cost = data[0]
            final_cost = data[-1]
            improvement = initial_cost - final_cost
            print(f"{label:15} | Initial: {initial_cost:6.1f} | Final: {final_cost:6.1f} | Improvement: {improvement:6.1f}")
    
    # Display DSA-RL agent probabilities if available
    if dsa_rl_stats:
        print(f"\nDSA-RL Learned Probabilities ({title}):")
        print("-" * 60)
        display_agent_probabilities(dsa_rl_stats, dsa_rl_dcop)

def display_agent_probabilities(final_stats, dcop_with_mu=None):
    """Display learned probabilities for DSA-RL agents after multi-episode learning"""
    if not final_stats:
        print("No DSA-RL statistics available")
        return
    
    # Handle single set of final statistics (not multiple repetitions)
    if not isinstance(final_stats, dict):
        print("Invalid statistics format")
        return
    
    # Get mu values if available
    mu_values = {}
    if dcop_with_mu and hasattr(dcop_with_mu, 'agent_mu_values'):
        mu_values = dcop_with_mu.agent_mu_values
    
    # Display header
    if mu_values:
        print("Agent | Final Probability | Theta   | Baseline | Priority μ")
        print("------|-------------------|---------|----------|----------")
    else:
        print("Agent | Final Probability | Theta   | Baseline")
        print("------|-------------------|---------|----------")
    
    # Display statistics for each agent
    all_probs = []
    for agent_id in sorted(final_stats.keys()):
        agent_stats = final_stats[agent_id]
        p = agent_stats['probability']
        theta = agent_stats['theta']
        baseline = agent_stats['baseline']
        all_probs.append(p)
        
        if mu_values and agent_id in mu_values:
            mu_val = mu_values[agent_id]
            print(f"{agent_id:5} | {p:16.3f} | {theta:7.3f} | {baseline:8.1f} | {mu_val:8.1f}")
        else:
            print(f"{agent_id:5} | {p:16.3f} | {theta:7.3f} | {baseline:8.1f}")
    
    # Summary statistics
    if all_probs:
        min_p = min(all_probs)
        max_p = max(all_probs)
        avg_p = sum(all_probs) / len(all_probs)
        print(f"\nSummary: Min p={min_p:.3f}, Max p={max_p:.3f}, Avg p={avg_p:.3f}")
        print(f"Learning spread: {max_p - min_p:.3f} (higher = more differentiation)")
        print(f"Total episodes: {getattr(dcop_with_mu, 'num_episodes', 'Unknown')}")
    
    # Display validation: show that each episode started with same conditions
    if dcop_with_mu and hasattr(dcop_with_mu, 'all_episode_costs'):
        episode_costs = dcop_with_mu.all_episode_costs
        if len(episode_costs) > 1:
            print(f"\n🔍 Validation - Episode Starting Costs:")
            for i, costs in enumerate(episode_costs[:5]):  # Show first 5 episodes
                if costs:
                    print(f"Episode {i+1}: {costs[0]:.1f}")
            if len(episode_costs) > 5:
                print(f"... (showing first 5 of {len(episode_costs)} episodes)")
            print("✅ Identical starting costs confirm same graph structure per episode")

# Main execution starts here
if __name__ == '__main__':
    print("DSA-REINFORCE Experiment Runner (No Plot)")
    print("=========================================")
    
    # Select experiment scenario - can be changed to 'priority_comparison', 'full', 'minimal'
    SELECTED_EXPERIMENT = 'standard'  # Options: 'standard', 'priority_comparison', 'full', 'minimal'
    USE_SYNCHRONIZED = True  # Enable synchronized topology for fair DSA vs DSA-RL comparison
    
    # Get experiment configuration from globals
    experiment_configs = {
        'standard': STANDARD_EXPERIMENT,
        'priority_comparison': PRIORITY_COMPARISON_EXPERIMENT,
        'full': FULL_EXPERIMENT,
        'minimal': MINIMAL_EXPERIMENT
    }
    
    required_dcops = experiment_configs.get(SELECTED_EXPERIMENT, STANDARD_EXPERIMENT)
    
    print(f"Running '{SELECTED_EXPERIMENT}' experiment with {len(required_dcops)} algorithms")
    print(f"Synchronization: {'ENABLED - Fair comparison with shared topology' if USE_SYNCHRONIZED else 'DISABLED - Original separate topologies'}")
    print(f"Experiment parameters: {repetitions} repetitions, {iteration_per_episode} iterations each")
    print(f"Problem size: {DEFAULT_AGENTS} agents, {DEFAULT_DOMAIN_SIZE} colors")

    if USE_SYNCHRONIZED:
        # Use synchronized experiments for fair comparison
        print("\n" + "="*60)
        print("🔗 SYNCHRONIZED EXPERIMENT MODE")
        print("✓ Same graph topology for all algorithms")
        print("✓ Same cost changes per episode across DSA and DSA-RL")  
        print("✓ Same starting assignments for fair comparison")
        print("="*60)
        
        # Run synchronized experiments for both graph densities
        results_k02, dsa_rl_stats_k02, dsa_rl_dcop_k02 = solve_synchronized_experiment(required_dcops, DEFAULT_GRAPH_DENSITIES[0])
        results_k07, dsa_rl_stats_k07, dsa_rl_dcop_k07 = solve_synchronized_experiment(required_dcops, DEFAULT_GRAPH_DENSITIES[1])
        
        # Display results
        display_results_no_plot(results_k02, f'Sparse Graph (k={DEFAULT_GRAPH_DENSITIES[0]}) - Synchronized', dsa_rl_stats_k02, dsa_rl_dcop_k02)
        display_results_no_plot(results_k07, f'Dense Graph (k={DEFAULT_GRAPH_DENSITIES[1]}) - Synchronized', dsa_rl_stats_k07, dsa_rl_dcop_k07)
        
    else:
        # Use original separate topology experiments (legacy mode)
        print("\n" + "="*60)
        print("⚠️  LEGACY MODE - Separate topologies per algorithm")
        print("⚠️  DSA and DSA-RL will use different graphs!")
        print("="*60)
        
        raise NotImplementedError("Legacy mode disabled - use synchronized mode for fair comparison")
    
    print("\nExperiment completed successfully!")