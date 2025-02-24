import numpy as np
from .Layer import Layer
from .Activation import sigmoid, tanh, relu 

class InputLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0, keepdims=True)
        self.stdX = np.std(dataIn, axis=0, keepdims=True, ddof=1)
        # For numeric stability
        self.stdX[self.stdX == 0] = 1

    def forward(self, dataIn):
        zscore_data = (dataIn - self.meanX) / self.stdX
        self.setPrevIn(dataIn)
        self.setPrevOut(zscore_data)
        return zscore_data


class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        self.W = np.random.uniform(-1e-4, 1e-4, (sizeIn, sizeOut))
        self.b = np.random.uniform(-1e-4, 1e-4, (1, sizeOut))

    def getWeights(self):
        return self.W

    def setWeights(self, W):
        self.W = W

    def getBiases(self):
        return self.b

    def setBiases(self, b):
        self.b = b

    def forward(self, dataIn):

        self.setPrevIn(dataIn)
        result = np.dot(dataIn, self.W) + self.b
        self.setPrevOut(result)
        return result

    def gradient(self):
        return np.tile(self.W.T, (self.getPrevIn().shape[0], 1, 1))
    
    def backward(self, gradIn):
        return np.dot(gradIn, self.W.T)
    
    def updateWeights(self,gradIn, eta):

        dJdb = np.sum(gradIn, axis = 0)/gradIn.shape[0]
        dJdW = (self.getPrevIn().T @ gradIn)/gradIn.shape[0]

        self.W -= eta*dJdW 
        self.b -= eta*dJdb