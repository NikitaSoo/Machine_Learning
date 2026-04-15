import numpy as np
import matplotlib.pyplot as plt

def perceptron(inputs, weights, bias):
    """Calculates the output of a single perceptron utilizing a step activation function."""
    assert len(inputs) == len(weights), "Dimensionality mismatch between inputs and weights."
    
    # Calculate the weighted sum using the dot product
    weighted_sum = np.dot(inputs, weights) + bias

    # --- YOUR CODE HERE --- #
    # Apply the step activation function:
    # If weighted_sum is greater than or equal to 0, output 1
    # Otherwise, output 0
    # --- END YOUR CODE --- #

# Test functionality
test_inputs =[1, 0]
test_weights = [0.5, 0.5]
test_bias = -0.7
print(f"Test Output: {perceptron(test_inputs, test_weights, test_bias)}") 