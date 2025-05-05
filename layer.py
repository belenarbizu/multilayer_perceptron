import pandas as pd 
import numpy as np
from toolkit import DataParser

class Layer:
    def __init__(self, n_inputs, n_neurons, activation='relu'):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.bias = np.zeros((1, n_neurons))
        self.activation = activation
        self.output = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.bias
        if self.activation == 'relu':
            self.output = DataParser.relu(self.output)
        
        return self.output

    def backward(self, dinputs):
        self.dinputs = np.dot(dinputs, self.weights.T)
        self.dweights = np.dot(self.inputs.T, dinputs)
        self.dbiases = np.sum(dinputs, axis=0, keepdims=True)
        return self.dinputs
