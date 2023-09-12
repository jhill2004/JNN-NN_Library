import numpy as np
import activation as ac

class Node:
    def __init__(self, value=0, key=1):
        self.value = value
        self.input = 0
        self.gradient = 0
        self.key=1
    
    def activation(self, z):
        self.input = z
        if self.key == 1:
            self.value = ac.relu(z)
        elif self.key == 2:
            self.value = ac.lrelu(z)
        elif self.key == 3:
            self.value = ac.sigmoid(z)
        
        return self.value
    
    def derivative(self):
        if self.key == 1:
            if self.value > 0:
                return 1
            else:
                return 0
        elif self.key == 2:
            if self.value > 0:
                return 1
            else:
                return 0.01
        elif self.key == 3:
            # can use self.value instead of ac.sigmoid(self.value) because the activate function should be called before 
            # the derivative function, and there self.value is modified into ac.sigmoid(self.value)
            return self.value * (1-self.value)
        
        return self.value
    





    
