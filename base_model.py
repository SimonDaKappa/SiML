import copy 
import numpy as np

class BaseModel:
  """ 
  The very base level class for all models
  """
  
  def copy(self):
    """
    returns a deep copy of the model
    """
    return copy.deepcopy(self)

  
  def cross_validate(self, X: np.array, y: np.array, k: int = 5):
    """
    performs k-fold cross validation on the model
    """
