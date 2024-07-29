from micrograd.engine.value import Value

def linear_activation(x):
    """
    Linear activation function, useful for e.g. linear regression
    """
    return x

# logistic activation (1 / (1 + e^-f(x)))
def sigmoid_activation(x):
    """
    Note: For classification we need the logistic activation to ensure our output is mapped between 0 and 1. 
    This allows us to apply log function without domain errors (required for our Binary Cross entropy loss function).
    """
    # print(f"Test: {x}")
    return Value(1.0) / (Value(1.0) + (-x).exp())

def relu_activation(x):
    """
    ReLU activation function, useful for e.g. neural networks
    """
    return x.relu()