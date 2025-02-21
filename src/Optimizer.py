import numpy as np

class AdamOptimizer:
    """Adam optimizer implementation."""
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers  # List of network layers (e.g., [FullyConnectedLayer])
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestep
        
        # Initialize moment estimates for weights/biases in all layers
        self.m_weights = [np.zeros_like(layer.weights) for layer in layers if hasattr(layer, 'weights')]
        self.v_weights = [np.zeros_like(layer.weights) for layer in layers if hasattr(layer, 'weights')]
        self.m_biases = [np.zeros_like(layer.biases) for layer in layers if hasattr(layer, 'biases')]
        self.v_biases = [np.zeros_like(layer.biases) for layer in layers if hasattr(layer, 'biases')]

    def step(self):
        """Update parameters using Adam."""
        self.t += 1
        layer_idx = 0
        
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'grad_weights'):
                # Update weights
                self.m_weights[layer_idx] = self.beta1 * self.m_weights[layer_idx] + (1 - self.beta1) * layer.grad_weights
                self.v_weights[layer_idx] = self.beta2 * self.v_weights[layer_idx] + (1 - self.beta2) * (layer.grad_weights ** 2)
                
                # Bias correction
                m_hat_w = self.m_weights[layer_idx] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v_weights[layer_idx] / (1 - self.beta2 ** self.t)
                
                layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

                # Update biases
                self.m_biases[layer_idx] = self.beta1 * self.m_biases[layer_idx] + (1 - self.beta1) * layer.grad_biases
                self.v_biases[layer_idx] = self.beta2 * self.v_biases[layer_idx] + (1 - self.beta2) * (layer.grad_biases ** 2)
                
                m_hat_b = self.m_biases[layer_idx] / (1 - self.beta1 ** self.t)
                v_hat_b = self.v_biases[layer_idx] / (1 - self.beta2 ** self.t)
                
                layer.biases -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
                
                layer_idx += 1
