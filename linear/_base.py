import numpy as np
from scipy import linalg, optimize
from ..base_model import BaseModel

class LinearModel(BaseModel):
  
  
  def __init__(self, W: np.array = None, b: np.array = None):
    """
    Base CLass for Linear Classifier models
    
    Args:
        W (_type_): np.array of shape (m, k), the weights of the model
        b (_type_): np.array of shape (k, 1), the bias of the model
    """
    if W is None:
      self.W = self.construct_default_weights()
    else:
      self.W = W
    self.b = np.zeros((1, 1)) if b is None else b
    
    self.super().__init__()

  
  def fit(self, X: np.array, y: np.array):
    """
    fits the model to the input data
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
        y (_type_): np.array of shape (n, 1), the target data
    """
    
    
  def predict(self, X: np.array) -> np.array:
    """
    predicts the class of the input data
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
    
    Returns:
        np.array(_type_): the predicted class of the input data
    """
    
    
  def score(self, X: np.array, y: np.array) -> float:
    """
    returns the accuracy of the model
    
    Args:
        X (_type_): np.array of shape (n, m), the input data
        y (_type_): np.array of shape (n, 1), the target data
    
    Returns:
        float(_type_): the accuracy of the model
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
  
  
  def construct_default_weights(self, shape: tuple = [2, 1]):
    """
    Creates the default weights for the model

    Args:
        shape (tuple) : [a, b], a tuple of integers representing the length of each layer
        (i.e, the number of features and the number of classes)
    """
    self.W = np.random.rand(*shape)
    

  def __str__(self):
    return f"LinearModel(W={self.W}, b={self.b})"
  

