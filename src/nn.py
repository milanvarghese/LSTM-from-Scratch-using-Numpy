import numpy as np
from .ActivationFunctions import sigmoid, tanh, relu  # Import activation functions

class InputLayer:
    """Passes input data without modification."""
    def __init__(self, input_shape):
        self.input_shape = input_shape  # e.g., (batch_size, input_dim)
        self.output_shape = input_shape

    def forward(self, inputs):
        self.output = inputs
        return self.output

class FullyConnectedLayer:
    """Dense/fully connected layer with activation."""
    def __init__(self, input_dim, output_dim, activation='sigmoid'):
        self.weights = np.random.randn(output_dim, input_dim) * 0.01  # Xavier initialization
        self.biases = np.zeros((output_dim, 1))
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        # Compute pre-activation
        self.z = np.dot(self.weights, self.inputs.T).T + self.biases.T
        # Apply activation
        if self.activation == 'sigmoid':
            self.output = sigmoid(self.z)
        elif self.activation == 'tanh':
            self.output = tanh(self.z)
        elif self.activation == 'relu':
            self.output = relu(self.z)
        else:
            self.output = self.z  # Linear activation
        return self.output

    def backward(self, grad_output, learning_rate):
        # Compute gradients (to be used during LSTM backprop)
        if self.activation == 'sigmoid':
            grad_activation = self.output * (1 - self.output)
        elif self.activation == 'tanh':
            grad_activation = 1 - self.output**2
        elif self.activation == 'relu':
            grad_activation = (self.output > 0).astype(float)
        else:
            grad_activation = 1  # Linear

        self.grad_weights = np.dot(grad_output.T, self.inputs)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True).T
        grad_input = np.dot(grad_output, self.weights)
        
        # Update parameters
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases
        return grad_input * grad_activation
