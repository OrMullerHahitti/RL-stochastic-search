#!/usr/bin/env python3
"""
Main experiment runner without matplotlib plotting - just shows results
"""

from src.problems import *
from src.Globals_ import *

# Function to create DCOP instance based on the algorithm and parameters
def create_selected_dcop(i,algorithm, k, p=None, **kwargs):
    A = DEFAULT_AGENTS  # Number of agents from global config
    D = domain_size  # Domain size from global config
    
    # Extract agent priority configuration
    agent_mu_config = kwargs.get('agent_mu_config', None)
    
    if algorithm == Algorithm.DSA:
        dcop_name = f"DSA_{i}"
        return DCOP_DSA(i,A,D,dcop_name,algorithm, k, p, agent_mu_config)
    if algorithm == Algorithm.MGM:
        dcop_name = f"MGM_{i}"
        return DCOP_MGM(i,A,D,dcop_name,algorithm, k, agent_mu_config)
    if algorithm == Algorithm.DSA_RL:
        dcop_name = f"DSA_RL_{i}"
        # Extract RL-specific parameters
        p0 = kwargs.get('p0', 0.5)
        learning_rate = kwargs.get('learning_rate', 0.01)
        baseline_decay = kwargs.get('baseline_decay', 0.9)
        episode_length = kwargs.get('episode_length', 70)
        return DCOP_DSA_RL(i,A,D,dcop_name,algorithm, k, p0, learning_rate, baseline_decay, episode_length, agent_mu_config)

# Function to solve DCOPs and calculate average global cost
def solve_dcops(dcops, return_stats=False):
    total = [0.0] * 100 # Initialize list to store total costs for each iteration
    all_stats = []
    first_dcop = None

    for dcop in dcops:
        if first_dcop is None:
            first_dcop = dcop  # Keep reference to first DCOP for mu values
            
        global_cost = dcop.execute()
        for i in range(min(len(global_cost), 100)):
            total[i] += global_cost[i]
        
        # Collect learning statistics if DSA-RL
        if hasattr(dcop, 'get_learning_statistics'):
            stats = dcop.get_learning_statistics()
            all_stats.append(stats)
    
    avg_global_cost = [val / repetitions for val in total]
    
    if return_stats:
        return avg_global_cost, all_stats, first_dcop
    return avg_global_cost

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

def display_agent_probabilities(all_stats, dcop_with_mu=None):
    """Display learned probabilities for DSA-RL agents"""
    if not all_stats:
        print("No DSA-RL statistics available")
        return
    
    # Calculate average probabilities across all repetitions
    num_agents = len(all_stats[0]) if all_stats else 0
    if num_agents == 0:
        print("No agent data available")
        return
    
    avg_probs = {}
    avg_thetas = {}
    avg_baselines = {}
    
    # Aggregate statistics across repetitions
    for stats in all_stats:
        for agent_id, agent_stats in stats.items():
            if agent_id not in avg_probs:
                avg_probs[agent_id] = []
                avg_thetas[agent_id] = []
                avg_baselines[agent_id] = []
            
            avg_probs[agent_id].append(agent_stats['probability'])
            avg_thetas[agent_id].append(agent_stats['theta'])
            avg_baselines[agent_id].append(agent_stats['baseline'])
    
    # Get mu values if available
    mu_values = {}
    if dcop_with_mu and hasattr(dcop_with_mu, 'agent_mu_values'):
        mu_values = dcop_with_mu.agent_mu_values
    
    # Calculate and display averages
    if mu_values:
        print("Agent | Final Probability | Theta   | Baseline | Priority μ")
        print("------|-------------------|---------|----------|----------")
    else:
        print("Agent | Final Probability | Theta   | Baseline")
        print("------|-------------------|---------|----------")
    
    for agent_id in sorted(avg_probs.keys()):
        avg_p = sum(avg_probs[agent_id]) / len(avg_probs[agent_id])
        avg_theta = sum(avg_thetas[agent_id]) / len(avg_thetas[agent_id])
        avg_baseline = sum(avg_baselines[agent_id]) / len(avg_baselines[agent_id])
        
        if mu_values and agent_id in mu_values:
            mu_val = mu_values[agent_id]
            print(f"{agent_id:5} | {avg_p:16.3f} | {avg_theta:7.3f} | {avg_baseline:8.1f} | {mu_val:8.1f}")
        else:
            print(f"{agent_id:5} | {avg_p:16.3f} | {avg_theta:7.3f} | {avg_baseline:8.1f}")
    
    # Summary statistics
    all_probs = [p for probs in avg_probs.values() for p in probs]
    if all_probs:
        min_p = min(all_probs)
        max_p = max(all_probs)
        avg_p = sum(all_probs) / len(all_probs)
        print(f"\nSummary: Min p={min_p:.3f}, Max p={max_p:.3f}, Avg p={avg_p:.3f}")
        print(f"Learning spread: {max_p - min_p:.3f} (higher = more differentiation)")
    
    # Priority analysis if mu values available
    if mu_values:
        mu_vals = [mu_values[aid] for aid in sorted(mu_values.keys())]
        prob_vals = [sum(avg_probs[aid])/len(avg_probs[aid]) for aid in sorted(avg_probs.keys()) if aid in avg_probs]
        
        if len(mu_vals) == len(prob_vals):
            # Simple correlation analysis
            high_mu_agents = [i for i, mu in enumerate(mu_vals) if mu > 50]
            low_mu_agents = [i for i, mu in enumerate(mu_vals) if mu < 50]
            
            if high_mu_agents and low_mu_agents:
                high_mu_avg_p = sum(prob_vals[i] for i in high_mu_agents) / len(high_mu_agents)
                low_mu_avg_p = sum(prob_vals[i] for i in low_mu_agents) / len(low_mu_agents)
                print(f"Priority Analysis: High-μ agents avg p={high_mu_avg_p:.3f}, Low-μ agents avg p={low_mu_avg_p:.3f}")
                
                if high_mu_avg_p < low_mu_avg_p:
                    print("✓ Learning working: High priority agents are more conservative (lower p)")
                else:
                    print("! Learning insight: High priority agents are more aggressive (higher p)")

# Main execution starts here
if __name__ == '__main__':
    print("DSA-REINFORCE Experiment Runner")
    print("===============================")
    
    # Select experiment scenario - can be changed to 'priority_comparison', 'full', 'minimal'
    SELECTED_EXPERIMENT = 'standard'  # Options: 'standard', 'priority_comparison', 'full', 'minimal'
    
    # Get experiment configuration from globals
    experiment_configs = {
        'standard': STANDARD_EXPERIMENT,
        'priority_comparison': PRIORITY_COMPARISON_EXPERIMENT,
        'full': FULL_EXPERIMENT,
        'minimal': MINIMAL_EXPERIMENT
    }
    
    required_dcops = experiment_configs.get(SELECTED_EXPERIMENT, STANDARD_EXPERIMENT)

    y_axis_data_k_02=[]
    y_axis_data_k_07=[]
    
    print(f"Running '{SELECTED_EXPERIMENT}' experiment with {len(required_dcops)} algorithms")
    print(f"Experiment parameters: {repetitions} repetitions, {incomplete_iterations} iterations each")
    print(f"Problem size: {DEFAULT_AGENTS} agents, {domain_size} colors")

    # Solve and collect results for k=0.2 (sparse graph)
    print(f"\nProcessing sparse graphs (k={DEFAULT_GRAPH_DENSITIES[0]})...")
    dsa_rl_stats_k02 = None
    dsa_rl_dcop_k02 = None
    for dcop_config in required_dcops:
        initialized_dcops=[]
        for i in range(repetitions):
            if dcop_config['algorithm'] == Algorithm.DSA:
                initialized_dcops.append(create_selected_dcop(i, dcop_config['algorithm'], DEFAULT_GRAPH_DENSITIES[0], dcop_config['p']))
            elif dcop_config['algorithm'] == Algorithm.DSA_RL:
                # Pass RL-specific parameters
                rl_params = {k: v for k, v in dcop_config.items() if k not in ['algorithm', 'name']}
                initialized_dcops.append(create_selected_dcop(i, dcop_config['algorithm'], DEFAULT_GRAPH_DENSITIES[0], **rl_params))
            else:
                initialized_dcops.append(create_selected_dcop(i, dcop_config['algorithm'], DEFAULT_GRAPH_DENSITIES[0]))
        
        # Collect stats for DSA-RL
        if dcop_config['algorithm'] == Algorithm.DSA_RL:
            avg_global_cost, stats, first_dcop = solve_dcops(initialized_dcops, return_stats=True)
            dsa_rl_stats_k02 = stats
            dsa_rl_dcop_k02 = first_dcop
        else:
            avg_global_cost = solve_dcops(initialized_dcops)
        
        y_axis_data_k_02.append(avg_global_cost)
        print(f"  {dcop_config.get('name', dcop_config['algorithm'].name)} completed")

    # Solve and collect results for k=0.7 (dense graph)
    print(f"\nProcessing dense graphs (k={DEFAULT_GRAPH_DENSITIES[1]})...")
    dsa_rl_stats_k07 = None
    dsa_rl_dcop_k07 = None
    for dcop_config in required_dcops:
        initialized_dcops=[]
        for i in range(repetitions):
            if dcop_config['algorithm'] == Algorithm.DSA:
                initialized_dcops.append(create_selected_dcop(i, dcop_config['algorithm'], DEFAULT_GRAPH_DENSITIES[1], dcop_config['p']))
            elif dcop_config['algorithm'] == Algorithm.DSA_RL:
                # Pass RL-specific parameters
                rl_params = {k: v for k, v in dcop_config.items() if k not in ['algorithm', 'name']}
                initialized_dcops.append(create_selected_dcop(i, dcop_config['algorithm'], DEFAULT_GRAPH_DENSITIES[1], **rl_params))
            else:
                initialized_dcops.append(create_selected_dcop(i, dcop_config['algorithm'], DEFAULT_GRAPH_DENSITIES[1]))
        
        # Collect stats for DSA-RL
        if dcop_config['algorithm'] == Algorithm.DSA_RL:
            avg_global_cost, stats, first_dcop = solve_dcops(initialized_dcops, return_stats=True)
            dsa_rl_stats_k07 = stats
            dsa_rl_dcop_k07 = first_dcop
        else:
            avg_global_cost = solve_dcops(initialized_dcops)
        
        y_axis_data_k_07.append(avg_global_cost)
        print(f"  {dcop_config.get('name', dcop_config['algorithm'].name)} completed")

    # Display the results
    display_results(y_axis_data_k_02, f'Sparse Graph (k={DEFAULT_GRAPH_DENSITIES[0]})', dsa_rl_stats_k02, dsa_rl_dcop_k02, required_dcops)
    display_results(y_axis_data_k_07, f'Dense Graph (k={DEFAULT_GRAPH_DENSITIES[1]})', dsa_rl_stats_k07, dsa_rl_dcop_k07, required_dcops)
    
    print("\nExperiment completed successfully!")