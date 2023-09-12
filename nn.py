import numpy as np
from layer import Layer
from node import Node

class NN:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, key=1):
        layer = Layer(input_size, output_size, key)
        self.layers.append(layer)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data
    def backward(self, input_data):
        dypred = -2 * (input_data - self.forward(input_data))
        multilist = np.array([dypred])
        for layer in self.layers:
            pass
        pass
            
            
            
    


if __name__ == "__main__":
    # Initialize a simple NN
    print("aaaaaaaaa")
    neural_net = NN()
    neural_net.add_layer(input_size=3, output_size=5)  # Hidden layer
    neural_net.add_layer(input_size=5, output_size=1)  # Output layer

    # Test forward pass
    sample_input = np.array([0.5, 0.6, 0.1])
    prediction = neural_net.forward(sample_input)
    print("Prediction:", prediction)
    