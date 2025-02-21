import numpy as np

class Sigmoid:
    """Sigmoid activation (for loss function integration)."""
    @staticmethod
    def function(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        s = Sigmoid.function(x)
        return s * (1 - s)

class Tanh:
    """Tanh activation (for loss function integration)."""
    @staticmethod
    def function(x):
        return np.tanh(x)
    
    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x)**2
