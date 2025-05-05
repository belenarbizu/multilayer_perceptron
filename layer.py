import pandas as pd 
import numpy as np
from toolkit import DataParser

class Layer:
    '''
    Represents a neural network layer.

    Attributes:
    weights (np.ndarray): Weight matrix of shape (n_inputs, n_neurons), initialized using He initialization.
    bias (np.ndarray): Bias vector of shape (1, n_neurons), initialized to zeros.
    activation (str): Activation function to use.
    output (np.ndarray): Output of the layer after activation.
    dweights (np.ndarray): Gradient of the loss with respect to weights.
    dbiases (np.ndarray): Gradient of the loss with respect to biases.
    dinputs (np.ndarray): Gradient of the loss with respect to inputs.
    inputs (np.ndarray): Inputs of the layer.
    '''
    def __init__(self, n_inputs, n_neurons, activation='relu'):
        '''
        Initializes the layer with random weights and zero biases.

        Parameters:
        n_inputs (int): Number of input features.
        n_neurons (int): Number of neurons in the layer.
        activation (str): Activation function to use.
        '''
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.bias = np.zeros((1, n_neurons))
        self.activation = activation
        self.output = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None
        self.inputs = None

    def forward(self, inputs):
        '''
        Performs the forward pass through the layer.

        Parameters:
        inputs (np.ndarray): Input data.

        Returns:
        np.ndarray: Output of the layer after applying weights, biases, and activation.
        '''
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.bias
        if self.activation == 'relu':
            self.output = DataParser.relu(self.output)
        
        return self.output

    def backward(self, dinputs):
        '''
        Performs the backward pass, computing gradients for weights, biases, and inputs.

        Parameters:
        dinputs (np.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        np.ndarray: Gradient of the loss with respect to the input of this layer.
        '''
        self.dinputs = np.dot(dinputs, self.weights.T)
        self.dweights = np.dot(self.inputs.T, dinputs)
        self.dbiases = np.sum(dinputs, axis=0, keepdims=True)
        return self.dinputs
