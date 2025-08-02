import math


def sigmoid(x):
    """
    Compute the sigmoid function with numerical stability.

    Args:
        x (float): Input value

    Returns:
        float: Sigmoid of x
    """
    # Clamp x to prevent overflow
    x = max(-500, min(500, x))
    
    # Use numerically stable computation
    if x >= 0:
        exp_neg_x = math.exp(-x)
        return 1 / (1 + exp_neg_x)
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)

