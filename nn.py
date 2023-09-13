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
    def MSE(self, prediction, target):
        return (1/prediction.size) * (prediction-target)
    
    def backpropagate(self, target):
      
        prediction = self.layers[-1].get_values()
        dloss = prediction - target

    
        for i in reversed(range(1, len(self.layers))):
            dloss = self.layers[i].backward(dloss, self.layers[i-1])
            
    def update_weights(self, learning_rate):
        for layer in self.layers[1:]:
            for node in layer.nodes:
                for i in range(len(node.weights)):
                    node.weights[i] -= learning_rate * node.weights[i]

    def train(self, data, target, learning_rate):
        prediction = self.forward(data)
        self.backpropagate(target)
        self.update_weights(learning_rate)
        return self.compute_loss(prediction, target)
            
            
            
    

