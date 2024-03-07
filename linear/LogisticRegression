import numpy as np
from _base import LinearModel
from ..utils import sigmoid
from ..metrics import mean_squared_error, binary_cross_entropy


class LogisticRegression(LinearModel):
  
  
  def __init__(self, 
               W: np.array = None, 
               b: np.array = None, 
               training_iterations: int = 1000,
               learning_rate: float = 0.0001,
               fit_intercept: bool = True,
               gradient_descent_type: str = "batch"):
    """
    logistic regression linear model
    
    Args:
        W (_type_): np.array of shape (m, k), the weights of the model
        b (_type_): np.array of shape (k, 1), the bias of the model
        fit_intercept (_type_): bool, whether to fit the intercept
        gradient_descent_type (_type_): str, the type of gradient descent to use, can be 
            "batch", "stochastic", or TODO: "mini-batch"
    """
    self.fit_intercept = fit_intercept
    self.training_iterations = training_iterations
    self.learning_rate = learning_rate
    super().__init__(W, b)
    
  
  def fit(self, X: np.array, y: np.array):
    """
    fits the model by gradient descent
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
        y (_type_): np.array of shape (n, 1), the target data
    """
    if self.fit_intercept:
      # add a column of ones to the input data for the bias term
      X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    if self.W is None:
      self.W = np.random.rand(X.shape[1], 1)
      self.b = np.random.rand(1)
    
    batched_data = np.zeros((self.batch_size, X.shape[1]))
    
    for _ in range(self.training_iterations):
      
      
      # compute the gradient of the loss function for current batch
      grad = self.logistic_gradient(X, y)
      
      # update the weights and bias
      self.W -= self.learning_rate * grad[0]
  
  
  def predict(self, X: np.array) -> np.array:
    """
    predicts the class of the input data
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
    
    Returns:
        np.array(_type_): the predicted class of the input data
    """
    if self.fit_intercept:
      X = np.hstack((np.ones((X.shape[0], 1)), X))
    return sigmoid(X @ self.W + self.b) 
  
  
  def logistic_gradient(self, X: np.array, y: np.array) -> np.array:
    """
    computes the gradient of the logistic loss function, for some batch of the input data.
    The batch
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
        y (_type_): np.array of shape (n, 1), the target data
        
    Returns:
        np.array(_type_): the gradient of the logistic loss function
    """
    
    pass