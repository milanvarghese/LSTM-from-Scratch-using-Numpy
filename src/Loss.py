import numpy as np

class LogLoss:
    def eval(self, Y, Yhat):
        eps = 1e-7
        Yhat = np.clip(Yhat, eps, 1 - eps)
        return -np.mean(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))
    
    def gradient(self, Y, Yhat):
        eps = 1e-7
        Yhat = np.clip(Yhat, eps, 1 - eps)
        return np.atleast_2d((1 - Y) / (1 - Yhat) - Y / Yhat)
    

class CrossEntropy:
    def eval(self, Y, Yhat):
        eps = 1e-7
        Yhat = np.clip(Yhat, eps, 1 - eps)
        return -np.mean(np.sum(Y * np.log(Yhat), axis=1))
    
    def gradient(self, Y, Yhat):
        eps = 1e-7
        Yhat = np.clip(Yhat, eps, 1 - eps)
        return -Y / Yhat