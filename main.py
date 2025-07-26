#!/usr/bin/env python3
"""
DCOP Algorithm Comparison Main Runner

A unified command-line interface for running DCOP algorithm experiments
with support for DSA, MGM, and DSA-RL algorithms.
"""

import argparse
import sys
import time
from typing import List, Dict, Any, Optional
import json

# Conditional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will be disabled.")

from src.problems import create_dcop_problem, create_learned_policy_dcop, solve_synchronized_experiment
from src.topology import SharedGraphTopology
from src.global_map import (
    Algorithm, get_master_config, print_master_config,
    STANDARD_EXPERIMENT, MINIMAL_EXPERIMENT, FULL_EXPERIMENT
)
from src.validation import validate_dcop_config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DCOP Algorithm Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --algorithms DSA MGM --graphs 0.2 0.7 --repetitions 10
  %(prog)s --preset minimal --no-plot
  %(prog)s --algorithms DSA_RL --learning-episodes 30 --verbose
  %(prog)s --synchronized --output results.json
        """
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithms', 
        nargs='+', 
        choices=['DSA', 'MGM', 'DSA_RL'],
        default=['DSA', 'MGM', 'DSA_RL'],
        help='Algorithms to run (default: all)'
    )
    
    # Problem configuration
    parser.add_argument(
        '--agents', 
        type=int, 
        help='Number of agents (default: from global config)'
    )
    
    parser.add_argument(
        '--domain-size', 
        type=int, 
        help='Domain size (default: from global config)'
    )
    
    parser.add_argument(
        '--graphs', 
        nargs='+', 
        type=float,
        help='Graph densities to test (default: from global config)'
    )
    
    parser.add_argument(
        '--repetitions', 
        type=int, 
        help='Number of repetitions (default: from global config)'
    )
    
    parser.add_argument(
        '--iterations', 
        type=int, 
        help='Iterations per experiment (default: from global config)'
    )
    
    # DSA-specific parameters
    parser.add_argument(
        '--dsa-probabilities', 
        nargs='+', 
        type=float,
        default=[0.2, 0.7, 1.0],
        help='DSA probabilities to test (default: 0.2, 0.7, 1.0)'
    )
    
    # DSA-RL specific parameters
    parser.add_argument(
        '--learning-episodes', 
        type=int, 
        help='Number of learning episodes for DSA-RL (default: from global config)'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        help='Learning rate for DSA-RL (default: from global config)'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--preset', 
        choices=['standard', 'minimal', 'full'],
        help='Use preset experiment configuration'
    )
    
    parser.add_argument(
        '--priority-variant', 
        choices=['uniform', 'hierarchical', 'manual', 'stratified'],
        help='Agent priority configuration (default: from global config)'
    )
    
    parser.add_argument(
        '--synchronized', 
        action='store_true',
        help='Use synchronized topology for fair comparison (recommended)'
    )
    
    # Output options
    parser.add_argument(
        '--output', 
        type=str, 
        help='Output file for results (JSON format)'
    )
    
    parser.add_argument(
        '--no-plot', 
        action='store_true',
        help='Disable plotting (useful for headless environments)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--config', 
        action='store_true',
        help='Print current configuration and exit'
    )
    
    return parser.parse_args()


def create_experiment_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Create experiment configurations based on arguments."""
    configs = []
    config = get_master_config()
    
    # Get priority configuration
    if args.priority_variant:
        priority_config = config['priority_variant'][args.priority_variant]
    else:
        priority_config = config['priority_variant']['stratified']
    
    # Create algorithm configurations
    for algorithm_name in args.algorithms:
        algorithm = Algorithm[algorithm_name]
        
        if algorithm == Algorithm.DSA:
            # Create DSA configs with different probabilities
            for p in args.dsa_probabilities:
                configs.append({
                    'algorithm': algorithm,
                    'name': f'DSA_p{int(p*100):02d}',
                    'probability': p,
                    'agent_mu_config': priority_config
                })
        
        elif algorithm == Algorithm.MGM:
            configs.append({
                'algorithm': algorithm,
                'name': 'MGM',
                'agent_mu_config': priority_config
            })
        
        elif algorithm == Algorithm.DSA_RL:
            dsa_rl_config = {
                'algorithm': algorithm,
                'name': 'DSA_RL',
                'agent_mu_config': priority_config
            }
            
            # Add DSA-RL specific parameters if provided
            if args.learning_episodes:
                dsa_rl_config['num_episodes'] = args.learning_episodes
            if args.learning_rate:
                dsa_rl_config['learning_rate'] = args.learning_rate
            
            configs.append(dsa_rl_config)
    
    return configs


def run_single_graph_experiment(
    configs: List[Dict[str, Any]], 
    graph_density: float, 
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Run experiment for a single graph density."""
    config = get_master_config()
    
    # Get experiment parameters
    num_agents = args.agents or config['agents']
    domain_size = args.domain_size or config['domain_size']
    repetitions = args.repetitions or config['repetitions']
    
    if args.verbose:
        print(f"Running experiment for graph density k={graph_density}")
        print(f"  Agents: {num_agents}, Domain: {domain_size}, Repetitions: {repetitions}")
    
    results = {}
    
    if args.synchronized:
        # Use synchronized experiments
        if args.verbose:
            print("  Using synchronized topology for fair comparison")
        
        # Get priority config from first algorithm
        priority_config = configs[0]['agent_mu_config']
        
        # Create shared topology
        shared_topology = SharedGraphTopology(
            num_agents=num_agents,
            domain_size=domain_size,
            edge_probability=graph_density,
            agent_priority_config=priority_config,
            base_seed=42,
            mode="comparison"  # Use comparison mode for fair evaluation
        )
        
        # Run synchronized experiment
        experiment_results, dsa_rl_stats, dsa_rl_dcop = solve_synchronized_experiment(
            configs, graph_density, shared_topology
        )
        
        results.update(experiment_results)
        if dsa_rl_stats:
            results['dsa_rl_statistics'] = dsa_rl_stats
    
    else:
        # Run separate experiments (legacy mode)
        if args.verbose:
            print("  Using separate topologies (legacy mode)")
        
        for config_dict in configs:
            algorithm = config_dict['algorithm']
            name = config_dict['name']
            
            if args.verbose:
                print(f"    Running {name}...")
            
            # Create multiple problem instances
            total_costs = None
            max_iterations = args.iterations or config['iterations']
            
            for rep in range(repetitions):
                # Create problem instance
                problem_kwargs = {k: v for k, v in config_dict.items() 
                                if k not in ['algorithm', 'name', 'agent_mu_config']}
                
                dcop = create_dcop_problem(
                    algorithm=algorithm,
                    problem_id=rep,
                    num_agents=num_agents,
                    domain_size=domain_size,
                    edge_probability=graph_density,
                    agent_priority_config=config_dict['agent_mu_config'],
                    **problem_kwargs
                )
                
                # Execute problem
                cost_history = dcop.execute()
                
                # Accumulate results
                if total_costs is None:
                    total_costs = [0.0] * len(cost_history)
                
                for i, cost in enumerate(cost_history):
                    if i < len(total_costs):
                        total_costs[i] += cost
            
            # Calculate average costs
            if total_costs:
                avg_costs = [cost / repetitions for cost in total_costs]
                results[name] = avg_costs
    
    return results


def display_results(
    results_by_density: Dict[float, Dict[str, Any]], 
    args: argparse.Namespace
) -> None:
    """Display experiment results."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS")
    print("="*80)
    
    for density, results in results_by_density.items():
        print(f"\nGraph Density k={density}")
        print("-" * 60)
        
        # Display numerical results
        for name, cost_history in results.items():
            if name == 'dsa_rl_statistics':
                continue
                
            if isinstance(cost_history, list) and cost_history:
                initial_cost = cost_history[0]
                final_cost = cost_history[-1]
                improvement = initial_cost - final_cost
                print(f"{name:12}: Initial={initial_cost:7.1f}, Final={final_cost:7.1f}, "
                      f"Improvement={improvement:7.1f}")
        
        # Display DSA-RL statistics if available
        if 'dsa_rl_statistics' in results:
            print(f"\nDSA-RL Agent Statistics for k={density}:")
            stats = results['dsa_rl_statistics']
            if stats:
                print("Agent | Final Probability | Baseline")
                print("------|-------------------|----------")
                for agent_id in sorted(stats.keys()):
                    agent_stats = stats[agent_id]
                    prob = agent_stats.get('probability', 0.5)
                    baseline = agent_stats.get('baseline', 0.0)
                    print(f"{agent_id:5} | {prob:16.3f} | {baseline:8.1f}")
    
    # Plot results if matplotlib is available and plotting is enabled
    if MATPLOTLIB_AVAILABLE and not args.no_plot:
        plot_results(results_by_density, args)


def plot_results(results_by_density: Dict[float, Dict[str, Any]], args: argparse.Namespace) -> None:
    """Create plots of the results."""
    try:
        num_densities = len(results_by_density)
        fig, axes = plt.subplots(1, num_densities, figsize=(6*num_densities, 6))
        
        if num_densities == 1:
            axes = [axes]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, (density, results) in enumerate(results_by_density.items()):
            ax = axes[idx]
            
            color_idx = 0
            for name, cost_history in results.items():
                if name == 'dsa_rl_statistics' or not isinstance(cost_history, list):
                    continue
                
                if cost_history and color_idx < len(colors):
                    x_axis = list(range(len(cost_history)))
                    ax.plot(x_axis, cost_history, 
                           color=colors[color_idx], label=name, linewidth=2)
                    color_idx += 1
            
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Global Cost')
            ax.set_title(f'Graph Density k={density}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")


def save_results(results: Dict[str, Any], filename: str, args: argparse.Namespace) -> None:
    """Save results to JSON file."""
    try:
        # Prepare results for JSON serialization
        serializable_results = {}
        
        for density, density_results in results.items():
            serializable_results[str(density)] = {}
            
            for name, data in density_results.items():
                if isinstance(data, list):
                    serializable_results[str(density)][name] = data
                elif isinstance(data, dict):
                    # Convert numpy arrays and complex structures to lists/basic types
                    serializable_results[str(density)][name] = convert_to_serializable(data)
        
        # Add metadata
        metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'algorithms': args.algorithms,
                'synchronized': args.synchronized,
                'agents': args.agents,
                'domain_size': args.domain_size,
                'repetitions': args.repetitions,
                'iterations': args.iterations
            }
        }
        
        output_data = {
            'metadata': metadata,
            'results': serializable_results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        
    except Exception as e:
        print(f"Warning: Could not save results to {filename}: {e}")


def convert_to_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Handle config display
    if args.config:
        print_master_config()
        return
    
    # Validate configuration
    try:
        config = get_master_config()
    except Exception as e:
        print(f"Error: Invalid configuration: {e}")
        sys.exit(1)
    
    if args.verbose:
        print("DCOP Algorithm Comparison Tool")
        print("=" * 50)
        print(f"Algorithms: {', '.join(args.algorithms)}")
        print(f"Synchronized: {args.synchronized}")
    
    # Create experiment configurations
    try:
        experiment_configs = create_experiment_configs(args)
        if args.verbose:
            print(f"Created {len(experiment_configs)} algorithm configurations")
    except Exception as e:
        print(f"Error: Could not create experiment configurations: {e}")
        sys.exit(1)
    
    # Get graph densities to test
    graph_densities = args.graphs or config['graph_densities']
    if args.verbose:
        print(f"Graph densities: {graph_densities}")
    
    # Run experiments
    results_by_density = {}
    
    try:
        for density in graph_densities:
            start_time = time.time()
            
            results = run_single_graph_experiment(experiment_configs, density, args)
            results_by_density[density] = results
            
            elapsed = time.time() - start_time
            if args.verbose:
                print(f"  Completed in {elapsed:.1f} seconds")
        
        # Display results
        display_results(results_by_density, args)
        
        # Save results if requested
        if args.output:
            save_results(results_by_density, args.output, args)
        
        print(f"\nExperiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during experiment: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()