import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def activation(self, z):
        return max(0, z)

    def forward(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        return self.activation(z)

if __name__ == "__main__":
    print("Script is running...")  # Check if the script runs
    weights = [0.2, 0.8]
    bias = 0.1
    neuron = Neuron(weights, bias)
    
    inputs = [0.5, 1.5]
    output = neuron.forward(inputs)
    
    print(f"Neuron output: {output}")  # This line should print the output
