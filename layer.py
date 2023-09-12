from node import Node
import numpy as np

class Layer:
    def __init__(self, input_size, output_size, key=1):
        self.input_size = input_size
        self.output_size = output_size
        self.key = key
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.random.randn(1,output_size)*0.001
        self.nodes = []
        for i in range(output_size):
            self.nodes.append(Node(np.random.random_sample(), self.key))


    def forward(self, input_data):
        #input data is a row
        print(input_data.shape)
        print(self.weights.shape)
        z = np.dot(input_data, self.weights) + self.biases
        z = np.transpose(z)
        output = np.array([])
        for i in range(self.output_size):
            output = np.append(output, self.nodes[i].activation(z[i]))
        return output
            
