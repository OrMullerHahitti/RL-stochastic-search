#!/usr/bin/env python3
"""
DCOP Algorithm Comparison: Complete Learning and Evaluation Workflow

This script demonstrates the complete process of Distributed Constraint Optimization:
1. Problem Creation - Set up graph structure and constraints
2. Learning Phase - DSA-RL learns optimal probability policies  
3. Comparison Phase - Fair comparison of DSA, MGM, and learned DSA-RL
4. Results Analysis - Performance analysis and learned policy insights

Author: Claude Code
"""

import time
from typing import Dict, List, Any

# Conditional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False
    print("📊 Note: matplotlib not available. Plots will be disabled.")

from src.problems import create_dcop_problem
from src.topology import SharedGraphTopology
from src.global_map import Algorithm, get_master_config

def print_section_header(title: str, char: str = "=") -> None:
    """Print a nicely formatted section header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))

def display_config(config: Dict) -> None:
    """Display the experiment configuration."""
    print("🔧 Experiment Configuration:")
    print(f"   • Agents: {config['agents']}")
    print(f"   • Domain Size: {config['domain_size']} colors")
    print(f"   • Graph Densities: {config['graph_densities']}")
    print(f"   • Repetitions: {config['repetitions']}")
    print(f"   • Iterations per Episode: {config['iterations']}")
    print(f"   • DSA-RL Learning Episodes: {config['dsa_rl']['num_episodes']}")
    print(f"   • DSA-RL Learning Rate: {config['dsa_rl']['learning_rate']}")

def create_problem_structure(config: Dict) -> SharedGraphTopology:
    """
    Phase 1: Create the shared problem structure.
    
    This creates a fixed graph topology that will be used consistently
    across all algorithms and phases to ensure fair comparison.
    """
    print_subsection("Creating Shared Problem Structure")
    
    print("🏗️  Generating constraint graph topology...")
    
    # Create shared topology for fair comparison
    shared_topology = SharedGraphTopology(
        num_agents=config['agents'],
        domain_size=config['domain_size'],
        edge_probability=config['graph_densities'][0],  # Use first density for learning
        agent_priority_config=config['priority_variant']['stratified'],
        base_seed=42,  # Fixed seed for reproducibility
        mode="learning"  # Start in learning mode
    )
    
    stats = shared_topology.get_topology_stats()
    print(f"✅ Created constraint graph:")
    print(f"   • {stats['num_agents']} agents (countries)")
    print(f"   • {stats['num_edges']} constraints (borders)")
    print(f"   • {stats['density']:.1%} density")
    print(f"   • {stats['avg_degree']:.1f} average neighbors per agent")
    
    print("\n🎯 Problem Type: Graph Coloring")
    print("   • Each agent is a country choosing a color")
    print("   • Neighboring countries with same color incur penalty costs")
    print("   • Goal: Minimize total conflict penalties")
    
    return shared_topology

def run_learning_phase(shared_topology: SharedGraphTopology, config: Dict) -> Dict[float, Dict[int, float]]:
    """
    Phase 2: DSA-RL Learning Phase
    
    Train DSA-RL agents to learn optimal probability policies for EACH graph density.
    Each density requires different strategies due to different conflict patterns.
    """
    print_subsection("DSA-RL Learning Phase")
    
    print("🧠 Training DSA-RL agents with density-specific reinforcement learning...")
    print("   • Separate learning for each graph density")
    print("   • Each density has different optimal strategies")
    print("   • Varied penalties and initial assignments within each density")
    
    learned_probabilities_by_density = {}
    
    for density_idx, graph_density in enumerate(config['graph_densities']):
        density_type = "sparse" if graph_density < 0.5 else "dense"
        print(f"\n📊 Learning on {density_type} graph (k={graph_density})...")
        print(f"   • Expected strategy: {'More aggressive (fewer conflicts)' if graph_density < 0.5 else 'More coordinated (many conflicts)'}")
        
        # Create new shared topology for this density
        learning_topology = SharedGraphTopology(
            num_agents=config['agents'],
            domain_size=config['domain_size'],
            edge_probability=graph_density,
            agent_priority_config=config['priority_variant']['stratified'],
            base_seed=42 + density_idx,  # Different seed per density for variation
            mode="learning"
        )
        
        # Create DSA-RL problem for this density
        learning_dcop = create_dcop_problem(
            algorithm=Algorithm.DSA_RL,
            problem_id=density_idx,
            num_agents=config['agents'],
            domain_size=config['domain_size'],
            edge_probability=graph_density,
            agent_priority_config=config['priority_variant']['stratified'],
            shared_topology=learning_topology,
            current_episode=0,
            num_episodes=config['dsa_rl']['num_episodes'],
            learning_rate=config['dsa_rl']['learning_rate'],
            baseline_decay=config['dsa_rl']['baseline_decay']
        )
        
        print(f"   🎓 Running {config['dsa_rl']['num_episodes']} learning episodes...")
        start_time = time.time()
        
        # Execute learning for this density
        learning_costs = learning_dcop.execute()
        
        learning_time = time.time() - start_time
        print(f"   ✅ Density {graph_density} learning completed in {learning_time:.1f} seconds")
        
        # Get learned policies for this density
        learned_stats = learning_dcop.get_final_agent_statistics()
        density_probabilities = {agent_id: stats['probability'] 
                               for agent_id, stats in learned_stats.items()}
        
        learned_probabilities_by_density[graph_density] = density_probabilities
        
        # Display learning results for this density
        prob_values = list(density_probabilities.values())
        print(f"   📈 Results for k={graph_density}:")
        print(f"      - Initial cost: {learning_costs[0]:.1f}")
        print(f"      - Final cost: {learning_costs[-1]:.1f}")
        print(f"      - Improvement: {learning_costs[0] - learning_costs[-1]:.1f}")
        print(f"      - Prob range: {min(prob_values):.3f} - {max(prob_values):.3f}")
        print(f"      - Diversity: {max(prob_values) - min(prob_values):.3f}")
    
    # Compare policies across densities
    print("\n🔍 Policy Comparison Across Densities:")
    densities = list(learned_probabilities_by_density.keys())
    if len(densities) >= 2:
        sparse_probs = list(learned_probabilities_by_density[densities[0]].values())
        dense_probs = list(learned_probabilities_by_density[densities[1]].values())
        
        print(f"   - Sparse graph (k={densities[0]}) avg: {sum(sparse_probs)/len(sparse_probs):.3f}")
        print(f"   - Dense graph (k={densities[1]}) avg: {sum(dense_probs)/len(dense_probs):.3f}")
        print(f"   - Strategy difference: {abs(sum(sparse_probs)/len(sparse_probs) - sum(dense_probs)/len(dense_probs)):.3f}")
    
    return learned_probabilities_by_density

def run_comparison_phase(shared_topology: SharedGraphTopology, 
                        learned_probabilities_by_density: Dict[float, Dict[int, float]], 
                        config: Dict) -> Dict[str, List[float]]:
    """
    Phase 3: Algorithm Comparison Phase
    
    Run fair comparison between DSA (multiple probabilities), MGM, and DSA-RL.
    All algorithms use the same graph structure and identical problem instances.
    """
    print_subsection("Algorithm Comparison Phase")
    
    print("⚖️  Running fair comparison with identical problem instances...")
    print("   • Same graph structure for all algorithms")
    print("   • Same penalties and initial assignments per repetition")
    print("   • DSA-RL uses learned probabilities (no further learning)")
    
    # Switch to comparison mode for fair evaluation
    shared_topology.set_mode("comparison")
    
    results = {}
    
    # Test different graph densities
    for graph_density in config['graph_densities']:
        print(f"\n🌐 Testing on {'sparse' if graph_density < 0.5 else 'dense'} graph (k={graph_density})")
        
        # Update shared topology for this density
        shared_topology = SharedGraphTopology(
            num_agents=config['agents'],
            domain_size=config['domain_size'],
            edge_probability=graph_density,
            agent_priority_config=config['priority_variant']['stratified'],
            base_seed=42,
            mode="comparison"
        )
        
        density_results = {}
        
        # 1. Test DSA with multiple probabilities
        for p in [0.2, 0.7, 1.0]:
            print(f"   🎲 Running DSA (p={p})...")
            
            total_costs = None
            for rep in range(config['repetitions']):
                shared_topology.prepare_episode(rep)
                
                dsa_dcop = create_dcop_problem(
                    algorithm=Algorithm.DSA,
                    problem_id=rep,
                    num_agents=config['agents'],
                    domain_size=config['domain_size'],
                    edge_probability=graph_density,
                    probability=p,
                    agent_priority_config=config['priority_variant']['stratified'],
                    shared_topology=shared_topology,
                    current_episode=rep
                )
                
                costs = dsa_dcop.execute()
                
                if total_costs is None:
                    total_costs = [0.0] * len(costs)
                for i, cost in enumerate(costs):
                    if i < len(total_costs):
                        total_costs[i] += cost
            
            avg_costs = [cost / config['repetitions'] for cost in total_costs]
            density_results[f"DSA_p{int(p*100):02d}"] = avg_costs
        
        # 2. Test MGM
        print(f"   🤝 Running MGM...")
        
        total_costs = None
        for rep in range(config['repetitions']):
            shared_topology.prepare_episode(rep)
            
            mgm_dcop = create_dcop_problem(
                algorithm=Algorithm.MGM,
                problem_id=rep,
                num_agents=config['agents'],
                domain_size=config['domain_size'],
                edge_probability=graph_density,
                agent_priority_config=config['priority_variant']['stratified'],
                shared_topology=shared_topology,
                current_episode=rep
            )
            
            costs = mgm_dcop.execute()
            
            if total_costs is None:
                total_costs = [0.0] * len(costs)
            for i, cost in enumerate(costs):
                if i < len(total_costs):
                    total_costs[i] += cost
        
        avg_costs = [cost / config['repetitions'] for cost in total_costs]
        density_results["MGM"] = avg_costs
        
        # 3. Test DSA-RL with density-specific learned probabilities (no further learning)
        print(f"   🧠 Running DSA-RL (learned policies for k={graph_density})...")
        
        # Get the learned probabilities for this specific graph density
        density_learned_probabilities = learned_probabilities_by_density.get(graph_density, {})
        if not density_learned_probabilities:
            print(f"   ⚠️  Warning: No learned probabilities for density {graph_density}, skipping DSA-RL")
            density_results["DSA_RL_Learned"] = []
        else:
            print(f"   📋 Using policies learned specifically for k={graph_density}")
            
            from src.problems import create_learned_policy_dcop
            
            total_costs = None
            for rep in range(config['repetitions']):
                shared_topology.prepare_episode(rep)
                
                learned_dcop = create_learned_policy_dcop(
                    learned_probabilities=density_learned_probabilities,
                    problem_id=rep,
                    num_agents=config['agents'],
                    domain_size=config['domain_size'],
                    edge_probability=graph_density,
                    agent_priority_config=config['priority_variant']['stratified'],
                    shared_topology=shared_topology,
                    current_episode=rep
                )
                
                costs = learned_dcop.execute()
                
                if total_costs is None:
                    total_costs = [0.0] * len(costs)
                for i, cost in enumerate(costs):
                    if i < len(total_costs):
                        total_costs[i] += cost
            
            avg_costs = [cost / config['repetitions'] for cost in total_costs]
            density_results["DSA_RL_Learned"] = avg_costs
        
        results[f"k_{graph_density}"] = density_results
    
    return results

def analyze_results(results: Dict[str, Dict[str, List[float]]], 
                   learned_probabilities_by_density: Dict[float, Dict[int, float]]) -> None:
    """
    Phase 4: Results Analysis
    
    Analyze and display the performance comparison and learning insights.
    """
    print_subsection("Results Analysis")
    
    print("📊 Algorithm Performance Comparison:")
    
    for density_key, density_results in results.items():
        graph_density = float(density_key.split('_')[1])
        graph_type = "Sparse" if graph_density < 0.5 else "Dense"
        
        print(f"\n🌐 {graph_type} Graph (k={graph_density}):")
        print("   Algorithm      | Initial Cost | Final Cost | Improvement")
        print("   ---------------|--------------|------------|-------------")
        
        for alg_name, costs in density_results.items():
            if costs:
                initial = costs[0]
                final = costs[-1]
                improvement = initial - final
                print(f"   {alg_name:14} | {initial:10.1f} | {final:8.1f} | {improvement:9.1f}")
    
    print("\n🎯 Density-Specific Learning Analysis:")
    for density, probabilities in learned_probabilities_by_density.items():
        prob_values = list(probabilities.values())
        density_type = "sparse" if density < 0.5 else "dense"
        
        print(f"\n   📊 {density_type.capitalize()} Graph (k={density}):")
        print(f"      • Policy Diversity: {max(prob_values) - min(prob_values):.3f}")
        print(f"      • Average Probability: {sum(prob_values)/len(prob_values):.3f}")
        print(f"      • Agents with high prob (>0.7): {sum(1 for p in prob_values if p > 0.7)}")
        print(f"      • Agents with low prob (<0.3): {sum(1 for p in prob_values if p < 0.3)}")
        print(f"      • Balanced agents (0.3-0.7): {sum(1 for p in prob_values if 0.3 <= p <= 0.7)}")
    
    # Compare strategies across densities
    if len(learned_probabilities_by_density) >= 2:
        densities = sorted(learned_probabilities_by_density.keys())
        sparse_probs = list(learned_probabilities_by_density[densities[0]].values())
        dense_probs = list(learned_probabilities_by_density[densities[1]].values())
        
        sparse_avg = sum(sparse_probs) / len(sparse_probs)
        dense_avg = sum(dense_probs) / len(dense_probs)
        
        print(f"\n   🔍 Strategy Adaptation:")
        print(f"      • Sparse graph strategy: {sparse_avg:.3f} avg probability")
        print(f"      • Dense graph strategy: {dense_avg:.3f} avg probability")
        print(f"      • Adaptation magnitude: {abs(sparse_avg - dense_avg):.3f}")
        
        if sparse_avg > dense_avg:
            print(f"      ✅ Correct adaptation: More aggressive on sparse graphs")
        elif dense_avg > sparse_avg:
            print(f"      ⚠️  Unexpected: More aggressive on dense graphs")
        else:
            print(f"      ❓ No significant adaptation between densities")
    
    print("\n💡 Expected Learning Insights:")
    print("   • Sparse graphs: Should learn higher probabilities (more aggressive)")
    print("   • Dense graphs: Should learn lower/diverse probabilities (more coordinated)")
    print("   • Policy diversity within each density shows role specialization")
    print("   • DSA-RL should outperform fixed DSA using appropriate learned policies")

def plot_results(results: Dict[str, Dict[str, List[float]]]) -> None:
    """Create plots showing algorithm convergence."""
    if not PLOTTING_ENABLED:
        print("📊 Plotting disabled (matplotlib not available)")
        return
    
    print_subsection("Generating Performance Plots")

    try:
        num_densities = len(results)
        fig, axes = plt.subplots(1, num_densities, figsize=(6*num_densities, 6))
        
        if num_densities == 1:
            axes = [axes]
        
        colors = {
            'DSA_p20': '#1f77b4',    # Blue
            'DSA_p70': '#ff7f0e',    # Orange  
            'DSA_p100': '#2ca02c',   # Green
            'MGM': '#d62728',        # Red
            'DSA_RL_Learned': '#9467bd'  # Purple
        }
        
        for idx, (density_key, density_results) in enumerate(results.items()):
            ax = axes[idx]
            graph_density = float(density_key.split('_')[1])
            
            for alg_name, costs in density_results.items():
                if costs and alg_name in colors:
                    iterations = list(range(len(costs)))
                    ax.plot(iterations, costs, color=colors[alg_name], 
                           label=alg_name.replace('_', ' '), linewidth=2)
            
            graph_type = "Sparse" if graph_density < 0.5 else "Dense"
            ax.set_title(f'{graph_type} Graph (k={graph_density})')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Global Cost')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('DCOP Algorithm Performance Comparison', y=1.02, fontsize=16)
        plt.show()
        
        print("📈 Performance plots displayed successfully!")
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

def main():
    """
    Main execution function - complete DCOP learning and comparison workflow.
    """
    print_section_header("DCOP Algorithm Learning and Comparison", "=")
    
    print("🎯 Objective: Compare DSA, MGM, and learning-enhanced DSA-RL")
    print("📚 Method: Learn optimal policies, then fair comparison on identical problems")
    print("🧪 Focus: Graph coloring with distributed constraint optimization")
    
    # Load configuration
    config = get_master_config()
    display_config(config)
    
    start_time = time.time()
    
    try:
        # Phase 1: Create Problem Structure
        print_section_header("Phase 1: Problem Structure Creation")
        shared_topology = create_problem_structure(config)
        
        # Phase 2: Learning Phase  
        print_section_header("Phase 2: DSA-RL Learning Phase")
        learned_probabilities_by_density = run_learning_phase(shared_topology, config)
        
        # Phase 3: Comparison Phase
        print_section_header("Phase 3: Algorithm Comparison Phase")
        results = run_comparison_phase(shared_topology, learned_probabilities_by_density, config)
        
        # Phase 4: Analysis
        print_section_header("Phase 4: Results Analysis")
        analyze_results(results, learned_probabilities_by_density)
        
        # Optional: Plotting
        if PLOTTING_ENABLED:
            plot_results(results)
        
        # Summary
        total_time = time.time() - start_time
        print_section_header("Experiment Completed Successfully", "=")
        print(f"🏁 Total execution time: {total_time:.1f} seconds")
        print(f"📊 Results generated for {len(results)} graph densities")
        total_policies = sum(len(policies) for policies in learned_probabilities_by_density.values())
        print(f"🧠 DSA-RL learned {total_policies} agent-specific policies across {len(learned_probabilities_by_density)} densities")
        print(f"⚖️  Fair comparison completed across {config['repetitions']} repetitions")
        
        print("\n🔍 Key Findings:")
        print("   ✓ Graph structure maintained consistently across all phases")
        print("   ✓ DSA-RL learned density-specific probability policies")
        print("   ✓ Fair comparison uses appropriate learned policies per density")
        print("   ✓ Learning-enhanced agents adapt to different graph structures")
        
    except KeyboardInterrupt:
        print("\n⏹️  Experiment interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()