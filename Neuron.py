import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        """
        Initializes the Neuron with weights and bias.

        Args:
            weights (list or np.ndarray): The weights for the neuron.
            bias (float): The bias for the neuron.
        """
        self.weights = np.array(weights)
        self.bias = bias

    def activation(self, z):
        """
        Applies the activation function (ReLU).

        Args:
            z (float): The input to the activation function.

        Returns:
            float: The activated output.
        """
        return max(0, z)

    def forward(self, inputs):
        """
        Computes the output of the neuron given the inputs.

        Args:
            inputs (list or np.ndarray): The inputs to the neuron.

        Returns:
            float: The output of the neuron.

        Raises:
            ValueError: If the number of inputs does not match the number of weights.
        """
        if len(inputs) != len(self.weights):
            raise ValueError("The number of inputs must match the number of weights.")

        z = np.dot(self.weights, inputs) + self.bias
        return self.activation(z)

if __name__ == "__main__":
    print("Script is running...") 
    weights = [0.2, 0.8]
    bias = 0.1
    neuron = Neuron(weights, bias)
    
    inputs = [0.5, 1.5]
    try:
        output = neuron.forward(inputs)
        print(f"Neuron output: {output}")
    except ValueError as e:
        print(f"Error: {e}")
