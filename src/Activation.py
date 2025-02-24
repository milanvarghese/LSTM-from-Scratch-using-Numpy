import numpy as np
from .Layer import Layer

class sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):

        self.setPrevIn(dataIn)

        result = 1 / (1 + np.exp(-dataIn)) 
        
        self.setPrevOut(result)

        return result

    def gradient(self):
        out = self.getPrevOut()
        return out * (1 - out)

    def backward(self, gradIn):
        return gradIn * self.gradient2()
    
    

class tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):

        self.setPrevIn(dataIn)

        result = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))

        self.setPrevOut(result)
 
        return result

    def gradient(self):
        return 1 - np.square(self.getPrevOut())
    
    def backward(self, gradIn):
        return gradIn * self.gradient2()
    
        
    
class relu(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):

        self.setPrevIn(dataIn)
        
        result = np.maximum(0, dataIn)

        self.setPrevOut(result)

        return result

    def gradient(self):
        grad = (self.getPrevIn() > 0).astype(float)
        return np.array([np.diag(grad[i]) for i in range(grad.shape[0])])
    
    def backward(self, gradIn):
        return super().backward(gradIn)
    
class softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):

        self.setPrevIn(dataIn)

        # Avoiding Underflow
        exp_values = np.exp(dataIn - np.max(dataIn, axis=1, keepdims=True))  
        result = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.setPrevOut(result)
        
        return result

    def gradient(self):
        
        softmax_out = self.getPrevOut()
        batch_size, D = softmax_out.shape
        jacobians = np.zeros((batch_size, D, D))

        for i in range(batch_size):
            s = softmax_out[i].reshape(-1, 1)
            jacobians[i] = np.diag(s.flatten()) - np.dot(s, s.T)

        return jacobians
    
    def backward(self, gradIn):
        return super().backward(gradIn)
