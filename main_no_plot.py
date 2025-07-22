#!/usr/bin/env python3
"""
Main experiment runner without matplotlib plotting - just shows results
"""

from src.problems import *

# Function to create DCOP instance based on the algorithm and parameters
def create_selected_dcop(i,algorithm, k, p=None, **kwargs):
    A = 30  # Number of agents
    D = 10  # size of domain
    if algorithm == Algorithm.DSA:
        dcop_name = f"DSA_{i}"
        return DCOP_DSA(i,A,D,dcop_name,algorithm, k, p)
    if algorithm == Algorithm.MGM:
        dcop_name = f"MGM_{i}"
        return DCOP_MGM(i,A,D,dcop_name,algorithm, k)
    if algorithm == Algorithm.DSA_RL:
        dcop_name = f"DSA_RL_{i}"
        # Extract RL-specific parameters
        p0 = kwargs.get('p0', 0.5)
        learning_rate = kwargs.get('learning_rate', 0.01)
        baseline_decay = kwargs.get('baseline_decay', 0.9)
        episode_length = kwargs.get('episode_length', 70)
        return DCOP_DSA_RL(i,A,D,dcop_name,algorithm, k, p0, learning_rate, baseline_decay, episode_length)

# Function to solve DCOPs and calculate average global cost
def solve_dcops(dcops, return_stats=False):
    total = [0.0] * 100 # Initialize list to store total costs for each iteration
    all_stats = []

    for dcop in dcops:
        global_cost = dcop.execute()
        for i in range(min(len(global_cost), 100)):
            total[i] += global_cost[i]
        
        # Collect learning statistics if DSA-RL
        if hasattr(dcop, 'get_learning_statistics'):
            stats = dcop.get_learning_statistics()
            all_stats.append(stats)
    
    avg_global_cost = [val / repetitions for val in total]
    
    if return_stats:
        return avg_global_cost, all_stats
    return avg_global_cost

# Function to display results as text
def display_results(y_axis_data, title, dsa_rl_stats=None):
    print(f"\n{title}")
    print("="*50)
    
    labels = ['DSA: p=0.2', 'DSA: p=0.7', 'DSA: p=1', 'MGM', 'DSA-RL']
    
    for i, (data, label) in enumerate(zip(y_axis_data, labels)):
        if i < len(y_axis_data):
            initial_cost = data[0]
            final_cost = data[-1]
            improvement = initial_cost - final_cost
            print(f"{label:12} | Initial: {initial_cost:6.1f} | Final: {final_cost:6.1f} | Improvement: {improvement:6.1f}")
    
    # Display DSA-RL agent probabilities if available
    if dsa_rl_stats:
        print(f"\nDSA-RL Learned Probabilities ({title}):")
        print("-" * 60)
        display_agent_probabilities(dsa_rl_stats)

def display_agent_probabilities(all_stats):
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
    
    # Calculate and display averages
    print("Agent | Final Probability | Theta   | Baseline")
    print("------|-------------------|---------|----------")
    
    for agent_id in sorted(avg_probs.keys()):
        avg_p = sum(avg_probs[agent_id]) / len(avg_probs[agent_id])
        avg_theta = sum(avg_thetas[agent_id]) / len(avg_thetas[agent_id])
        avg_baseline = sum(avg_baselines[agent_id]) / len(avg_baselines[agent_id])
        
        print(f"{agent_id:5} | {avg_p:16.3f} | {avg_theta:7.3f} | {avg_baseline:8.1f}")
    
    # Summary statistics
    all_probs = [p for probs in avg_probs.values() for p in probs]
    if all_probs:
        min_p = min(all_probs)
        max_p = max(all_probs)
        avg_p = sum(all_probs) / len(all_probs)
        print(f"\nSummary: Min p={min_p:.3f}, Max p={max_p:.3f}, Avg p={avg_p:.3f}")
        print(f"Learning spread: {max_p - min_p:.3f} (higher = more differentiation)")

# Main execution starts here
if __name__ == '__main__':
    print("DSA-REINFORCE Experiment Runner")
    print("===============================")
    
    # Define DCOP configurations
    Dcop1= {'algorithm': Algorithm.DSA, 'p': 0.2}
    Dcop2={'algorithm': Algorithm.DSA, 'p': 0.7}
    Dcop3={'algorithm': Algorithm.DSA, 'p': 1}
    Dcop4={'algorithm': Algorithm.MGM}
    Dcop5={'algorithm': Algorithm.DSA_RL, 'p0': 0.5, 'learning_rate': 0.01, 'baseline_decay': 0.9, 'episode_length': 20}

    required_dcops = [Dcop1, Dcop2, Dcop3, Dcop4, Dcop5] # List of required DCOPs

    y_axis_data_k_02=[]
    y_axis_data_k_07=[]

    print(f"Running experiments with {repetitions} repetitions and {incomplete_iterations} iterations each...")

    # Solve and collect results for k=0.2 (sparse graph)
    print("\nProcessing sparse graphs (k=0.2)...")
    dsa_rl_stats_k02 = None
    for dcop in required_dcops:
        initialized_dcops=[]
        for i in range(repetitions):
            if dcop['algorithm'] == Algorithm.DSA:
                initialized_dcops.append(create_selected_dcop(i,dcop['algorithm'], 0.2, dcop['p']))
            elif dcop['algorithm'] == Algorithm.DSA_RL:
                # Pass RL-specific parameters
                rl_params = {k: v for k, v in dcop.items() if k != 'algorithm'}
                initialized_dcops.append(create_selected_dcop(i,dcop['algorithm'], 0.2, **rl_params))
            else:
                initialized_dcops.append(create_selected_dcop(i,dcop['algorithm'], 0.2))
        
        # Collect stats for DSA-RL
        if dcop['algorithm'] == Algorithm.DSA_RL:
            avg_global_cost, stats = solve_dcops(initialized_dcops, return_stats=True)
            dsa_rl_stats_k02 = stats
        else:
            avg_global_cost = solve_dcops(initialized_dcops)
        
        y_axis_data_k_02.append(avg_global_cost)
        print(f"  {dcop['algorithm'].name} completed")

    # Solve and collect results for k=0.7 (dense graph)
    print("\nProcessing dense graphs (k=0.7)...")
    dsa_rl_stats_k07 = None
    for dcop in required_dcops:
        initialized_dcops=[]
        for i in range(repetitions):
            if dcop['algorithm'] == Algorithm.DSA:
                initialized_dcops.append(create_selected_dcop(i,dcop['algorithm'], 0.7, dcop['p']))
            elif dcop['algorithm'] == Algorithm.DSA_RL:
                # Pass RL-specific parameters
                rl_params = {k: v for k, v in dcop.items() if k != 'algorithm'}
                initialized_dcops.append(create_selected_dcop(i,dcop['algorithm'], 0.7, **rl_params))
            else:
                initialized_dcops.append(create_selected_dcop(i,dcop['algorithm'], 0.7))
        
        # Collect stats for DSA-RL
        if dcop['algorithm'] == Algorithm.DSA_RL:
            avg_global_cost, stats = solve_dcops(initialized_dcops, return_stats=True)
            dsa_rl_stats_k07 = stats
        else:
            avg_global_cost = solve_dcops(initialized_dcops)
        
        y_axis_data_k_07.append(avg_global_cost)
        print(f"  {dcop['algorithm'].name} completed")

    # Display the results
    display_results(y_axis_data_k_02, 'Sparse Graph (k=0.2)', dsa_rl_stats_k02)
    display_results(y_axis_data_k_07, 'Dense Graph (k=0.7)', dsa_rl_stats_k07)
    
    print("\nExperiment completed successfully!")