import copy 
import numpy as np
from metrics import loss

class BaseModel:
  """ 
  The very base level class for all models
  """
  
  
  def __init__(self):
    self.loss_fn = None
  
  
  def copy(self):
    """
    returns a deep copy of the model
    """
    return copy.deepcopy(self)

  
  def cross_validate(self, X: np.array, y: np.array, k: int = 5):
    """
    performs k-fold cross validation on the model
    """
    if not self.loss_fn:
      raise ValueError("Loss function not defined")
    
