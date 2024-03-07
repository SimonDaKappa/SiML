import numpy as np
from ..base_model import (
  BaseClassifier, 
  BaseRegressor
)


class LinearClassifier(BaseClassifier):
  algorithm = None
  W = None
  b = None
  
  
  """
  Base CLass for Linear Classifier models
  
  Args:
      BaseClassifier (BaseModel): the BaseClassifier superclass
  """
  def set_training_algorithm(self, algorithm: str):
    """
    sets the training algorithm for the model
    
    Args:
        algorithm (_type_): str, the training algorithm to use
    """
    try:
      self.training_algorithm = getattr(self, algorithm, None)
    except AttributeError:
      raise ValueError(
        "algorithm must be a valid training algorithm\n"
        + "Algorithms supported: PLA, LinearRegressoion, LogisticRegression"
      )
  
  
  def fit(self, X: np.array, y: np.array):
    """
    fits the model to the input data
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
        y (_type_): np.array of shape (n, 1), the target data
    """
    self.training_algorithm(X, y)
    
    
  def predict(self, X: np.array) -> np.array:
    """
    predicts the class of the input data
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
    
    Returns:
        np.array(_type_): the predicted class of the input data
    """
    
    
  def set_weights(self, W: np.array, b: np.array = None):
    """
    sets the weights and bias of the model
    
    Args:
        W (_type_): np.array of shape (m, k), the weights of the model
        b (_type_): np.array of shape (k, 1), the bias of the model
    """
    self.W = W
    self.b = self.b if b is None else b
  
  
  def set_bias(self, b: np.array):
    """
    sets the bias of the model
    
    Args:
        b (_type_): np.array of shape (k, 1), the bias of the model
    """
    self.b = b
  
  
 