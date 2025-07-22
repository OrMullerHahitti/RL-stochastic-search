import matplotlib.pyplot as plt
from problems import *


# Function to create DCOP instance based on the algorithm and parameters
def create_selected_dcop(i,algorithm, k, p=None):
    A = 30  # Number of agents
    D = 10  # size of domain
    if algorithm == Algorithm.DSA:
        dcop_name = f"DSA_{i}"
        return DCOP_DSA(i,A,D,dcop_name,algorithm, k, p)
    if algorithm == Algorithm.MGM:
        dcop_name = f"MGM_{i}"
        return DCOP_MGM(i,A,D,dcop_name,algorithm, k)

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

    y_axis1 = y_axis_data[0]
    y_axis2 = y_axis_data[1]
    y_axis3 = y_axis_data[2]
    y_axis4 = y_axis_data[3]

    plt.xlabel('Iterations')  # Label for x-axis
    plt.ylabel('Global Cost')  # Label for y-axis
    plt.title(title)  # Title for the graph

    # Plot the points with labels
    plt.plot(x_axis, y_axis1, color='#1f77b4', label='DSA: p=0.2')
    plt.plot(x_axis, y_axis2, color='#ff7f0e', label='DSA: p=0.7')
    plt.plot(x_axis, y_axis3, color='#2ca02c', label='DSA: p=1')
    plt.plot(x_axis, y_axis4, color='#d62728', label='MGM')

    plt.legend()  # Add legend to the graph
    plt.show()

# Main execution starts here
if __name__ == '__main__':
    # Define DCOP configurations
    Dcop1= {'algorithm': Algorithm.DSA, 'p': 0.2}
    Dcop2={'algorithm': Algorithm.DSA, 'p': 0.7}
    Dcop3={'algorithm': Algorithm.DSA, 'p': 1}
    Dcop4={'algorithm': Algorithm.MGM}

    required_dcops = [Dcop1, Dcop2, Dcop3, Dcop4] # List of required DCOPs

    y_axis_data_k_02=[]
    y_axis_data_k_07=[]

    # Solve and collect results for k=0.2 (sparse graph)
    for dcop in required_dcops:
        initialized_dcops=[]
        for i in range(repetitions):
            if dcop['algorithm'] == Algorithm.DSA:
                initialized_dcops.append(create_selected_dcop(i,dcop['algorithm'], 0.2, dcop['p']))
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
            else:
                initialized_dcops.append(create_selected_dcop(i,dcop['algorithm'], 0.7))
        avg_global_cost=solve_dcops(initialized_dcops)
        y_axis_data_k_07.append(avg_global_cost)

    # Display the graphs for the results
    display_graph(y_axis_data_k_02, 'Sparse Graph (k=0.2)')
    display_graph(y_axis_data_k_07, 'Dense Graph (k=0.7)')

