import pandas as pd 
import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
        self.output = None


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
