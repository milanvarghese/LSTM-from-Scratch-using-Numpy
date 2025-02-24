import numpy as np

def adam_optimize_objective(objectiveFunction, objectiveGrad, x, w_init, lr=5, epochs=100, rho1=0.9, rho2=0.999, delta=1e-8):
    """
    Adam optimizer for a scalar objective function.
    
    Parameters:
      objectiveFunction : Function that takes (x, w) and returns the scalar objective J.
      objectiveGrad     : Function that takes (x, w) and returns the gradient dJ/dw.
      x                 : The input value (for example, x = 1).
      w_init            : Initial value for the parameter w.
      lr                : Learning rate.
      epochs            : Number of epochs (iterations).
      rho1              : Exponential decay rate for the first moment estimate (beta1).
      rho2              : Exponential decay rate for the second moment estimate (beta2).
      delta             : A small constant for numerical stability (epsilon).
      
    Returns:
      w         : Optimized parameter.
      JValues   : List of objective function values at each epoch.
    """
    w = w_init
    s = 0.0
    r = 0.0
    JValues = []
    
    for t in range(1, epochs + 1):
        try:
            J = objectiveFunction(x, w)
            JValues.append(J)
            grad = objectiveGrad(x, w)
            
            # Update biased first moment estimate.
            s = rho1 * s + (1 - rho1) * grad
            # Update biased second moment estimate.
            r = rho2 * r + (1 - rho2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate.
            s_hat = s / (1 - rho1 ** t)
            # Compute bias-corrected second moment estimate.
            r_hat = r / (1 - rho2 ** t)
            
            # Update parameter.
            w = w - lr * (s_hat / (np.sqrt(r_hat) + delta))
            
        except OverflowError:
            print(f"Overflow encountered at epoch {t}. Stopping early.")
            break
    
    return w, JValues
