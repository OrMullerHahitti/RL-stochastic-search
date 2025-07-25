import math


def sigmoid(x):
    """
    Compute the sigmoid function with numerical stability.

    Args:
        x (float): Input value

    Returns:
        float: Sigmoid of x
    """
    # Clamp x to prevent overflow - sigmoid approaches 0 or 1 beyond these bounds
    x = max(-500, min(500, x))
    
    # Use numerically stable computation
    if x >= 0:
        exp_neg_x = math.exp(-x)
        return 1 / (1 + exp_neg_x)
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)
def compute_advantage(reward, baseline):
    """
    Compute advantage for REINFORCE with baseline.
    
    Args:
        reward (float): Reward received
        baseline (float): Current baseline value
        
    Returns:
        float: Advantage (reward - baseline)
    """
    return reward - baseline

def update_exponential_moving_average(current_value, new_value, beta):
    """
    Update exponential moving average baseline.
    
    Args:
        current_value (float): Current EMA value
        new_value (float): New observation
        beta (float): Decay factor (typically 0.9)
        
    Returns:
        float: Updated EMA value
    """
    return beta * current_value + (1 - beta) * new_value

def distribute_rewards_proportionally(changers, local_gains, global_improvement):
    """
    Distribute global improvement among changers proportionally to their local gains.
    
    Args:
        changers (list): List of agent IDs that made changes
        local_gains (dict): Dictionary mapping agent_id -> local_gain
        global_improvement (float): Total global improvement to distribute
        
    Returns:
        dict: Dictionary mapping agent_id -> reward
    """
    if not changers or global_improvement == 0:
        return {agent_id: 0 for agent_id in changers}
    
    # Get total positive local gains for proportional distribution
    total_positive_gain = sum(max(0, local_gains.get(agent_id, 0)) for agent_id in changers)
    
    rewards = {}
    
    if total_positive_gain > 0 and global_improvement > 0:
        # Distribute positive improvement proportionally to positive local gains
        for agent_id in changers:
            local_gain = max(0, local_gains.get(agent_id, 0))
            proportion = local_gain / total_positive_gain
            rewards[agent_id] = proportion * global_improvement
    elif global_improvement < 0:
        # Distribute negative improvement equally among changers
        reward_per_agent = global_improvement / len(changers)
        for agent_id in changers:
            rewards[agent_id] = reward_per_agent
    else:
        # No improvement or no positive gains - give zero rewards
        for agent_id in changers:
            rewards[agent_id] = 0
            
    return rewards

