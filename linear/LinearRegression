import numpy as np
from _base import LinearModel

class LinearRegression(LinearModel):
  
  
  def __init__(self, W: np.array = None, b: np.array = None, fit_intercept: bool = True):
    """
    Ordinary Least Squares linear regression model
    
    Args:
        W (_type_): np.array of shape (m, k), the weights of the model
        b (_type_): np.array of shape (k, 1), the bias of the model
    """
    self.fit_intercept = fit_intercept
    super().__init__(W, b)
    
    
  def fit(self, X: np.array, y: np.array):
    """
    fits the model by ordinary least squares
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
        y (_type_): np.array of shape (n, 1), the target data
    """
    if self.fit_intercept:
      # add a column of ones to the input data for the bias term
      X = np.hstack((np.ones((X.shape[0], 1)), X))
      self.W = np.linalg.pinv(X) @ X.T @ y
      self.b = self.W[0]
    else:
      self.W = np.linalg.pinv(X) @ X.T @ y
      self.b = 0
      
  
  def predict(self, X: np.array) -> np.array:
    """
    predicts the output W.T @ X of the input data
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
    
    Returns:
        np.array(_type_): the predicted class of the input data
    """
    if self.fit_intercept:
      X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X @ self.W + self.b
    
    