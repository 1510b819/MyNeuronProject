Neuron Class in Python

This repository contains a simple implementation of a Neuron class in Python, utilizing NumPy for numerical operations. The class features basic functionality for a single neuron, including weights, bias, and an activation function.
Table of Contents

    Installation
    Usage
    Code Overview
    License

Installation

To use the Neuron class, ensure you have Python and NumPy installed. You can install NumPy using pip:

bash

pip install numpy

Usage

You can use the Neuron class as follows:

python

import numpy as np

# Define weights and bias
weights = [0.2, 0.8]
bias = 0.1

# Create a Neuron instance
neuron = Neuron(weights, bias)

# Define inputs
inputs = [0.5, 1.5]

# Compute the output
output = neuron.forward(inputs)

print(f"Neuron output: {output}")

Code Overview
Neuron Class

    Initialization (__init__): Initializes the neuron with given weights and bias.

    Activation Function (activation): Applies the ReLU activation function. It returns the maximum of 0 and the input value.

    Forward Pass (forward): Computes the neuron's output by performing a dot product of weights and inputs, adding the bias, and applying the activation function. It raises a ValueError if the number of inputs does not match the number of weights.

Example

The if __name__ == "__main__": block provides a basic example of how to create a Neuron instance and compute its output given inputs.
License

This project is licensed under the MIT License - see the LICENSE file for details.
