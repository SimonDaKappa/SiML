import numpy as np
from cvxopt import matrix, solvers
from cvxopt.solvers import qp
from numpy.core.multiarray import array as array
from _base import LinearModel
from ..metrics.kernel import *

class SupportVectorMachine(LinearModel):
  """ SupportVectorMachine with kernel trick

  Args:
      LinearModel (_type_): _description_
  """
  
  def __init__(self, 
               kernel: Kernel,
               lambda_regularize: float = 1.0,
               max_iter: int = 1000):
    super().__init__()
    self.lambda_regularize = lambda_regularize
    self.kernel = Kernel
    
    
  
  def fit(self, X: np.array, y: np.array):
    """
    fits the model to the input data, using dual formulation
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
        y (_type_): np.array of shape (n, 1), the target data
    """
    m = X.shape[0]
    gram_matrix = self._gram_matrix(X, y)
    G = []
    h = []
    
    for i in range(m):
      zero_constraint = [0 for j in range(m)]
      zero_constraint[i] = -1
      regularization_constraint = [0 for j in range(m)]
      regularization_constraint[i] = 1
      G.extend([zero_constraint, regularization_constraint])
      h.extend([0, self.lambda_regularize])
    
    ones = [1 for i in range(m)]
    A = np.array([[y[i]] for i in range(m)]).T
    G = matrix(np.array(G).astype(np.double))
    h = matrix(np.array(h).astype(np.double))
    A = matrix(A.astype(np.double))
    b = matrix(np.array([0]).astype(np.double))
    ones = matrix(-1*np.array(ones).astype(np.double))
    
    alpha = qp(P=gram_matrix, q=ones, G=G, h=h, A=A, b=b)
    s_idx = -1
    for i in range(len(alpha['x'])):
      if alpha['x'][i] > 0 and alpha['x'][i] < self.C:
        s_idx = i
        break

    # recover primal b_star = y[s] - sum_{alpha_i > 0}^N alpha_i y_i K(x_i, x_s)
    self.b_star = y[s_idx] - sum([alpha['x'][i]*y[i]*self.kernel(X[i], X[s_idx])
                                 for i in range(len(alpha['x']))])
    # recover primal hypothesis(x_input) = sign(sum_{alpha_i > 0}^N alpha_i y_i K(x_i, x_input) + b_star)
    self.hypothesis = lambda x: np.sign(sum(
        [alpha['x'][i]*y[i]*self.kernel(X[i], x) for i in range(len(alpha['x']))]) + self.b_star)
    
    
  def predict(self, X: np.array) -> np.array:
    """
    predicts the class of the input data
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
    
    Returns:
        np.array(_type_): the predicted class of the input data
    """
    return np.array([self.hypothesis(x) for x in X]
  
  
  def _gram_matrix(self, X: np.array, y: np.array) -> matrix:
    """
    computes the gram matrix of the input data
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
    
    Returns:
        np.array(_type_): the gram matrix of the input data
    """
    gram = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
      for j in range(X.shape[0]):
        gram[i, j] = y[i]*y[j]*self.kernel(X[i], X[j])
    return matrix(gram)
    
    
    