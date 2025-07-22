import matplotlib.pyplot as plt
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
        episode_length = kwargs.get('episode_length', 20)
        return DCOP_DSA_RL(i,A,D,dcop_name,algorithm, k, p0, learning_rate, baseline_decay, episode_length)

# Function to solve DCOPs and calculate average global cost
# Doesn't guarantee optimal solution (incomplete solver-run for incomplete_iterations)
def solve_dcops(dcops):
    total = [0.0] * 100 # Initialize list to store total costs for each iteration

    for dcop in dcops:
        global_cost=dcop.execute()
        for i in range(incomplete_iterations):
            total[i] += global_cost[i]
    avg_global_cost = [val / repetitions for val in total]  # Calculate the average global cost for each iteration
    return avg_global_cost

# Function to display the graphs for the results
def display_graph(y_axis_data, title):

    x_axis = list(range(100))  # X-axis represents iterations

    plt.figure(figsize=(10, 6))
    plt.xlabel('Iterations')  # Label for x-axis
    plt.ylabel('Global Cost')  # Label for y-axis
    plt.title(title)  # Title for the graph

    # Plot the points with labels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    labels = ['DSA: p=0.2', 'DSA: p=0.7', 'DSA: p=1', 'MGM', 'DSA-RL']
    
    for i, (data, label, color) in enumerate(zip(y_axis_data, labels, colors)):
        if i < len(y_axis_data):
            plt.plot(x_axis, data, color=color, label=label)

    plt.legend()  # Add legend to the graph
    plt.show()

# Main execution starts here
if __name__ == '__main__':
    # Define DCOP configurations
    Dcop1= {'algorithm': Algorithm.DSA, 'p': 0.2}
    Dcop2={'algorithm': Algorithm.DSA, 'p': 0.7}
    Dcop3={'algorithm': Algorithm.DSA, 'p': 1}
    Dcop4={'algorithm': Algorithm.MGM}
    Dcop5={'algorithm': Algorithm.DSA_RL, 'p0': 0.5, 'learning_rate': 0.01, 'baseline_decay': 0.9, 'episode_length': 20}

    required_dcops = [Dcop1, Dcop2, Dcop3, Dcop4, Dcop5] # List of required DCOPs

    y_axis_data_k_02=[]
    y_axis_data_k_07=[]

    # Solve and collect results for k=0.2 (sparse graph)
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
        avg_global_cost=solve_dcops(initialized_dcops)
        y_axis_data_k_02.append(avg_global_cost)

    # Solve and collect results for k=0.7 (dense graph)
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
        avg_global_cost=solve_dcops(initialized_dcops)
        y_axis_data_k_07.append(avg_global_cost)

    # Display the graphs for the results
    display_graph(y_axis_data_k_02, 'Sparse Graph (k=0.2)')
    display_graph(y_axis_data_k_07, 'Dense Graph (k=0.7)')

