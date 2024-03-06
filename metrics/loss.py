# ---------------------------------------------------------------------
#                        Loss Functions
#                Regression and Classification 

# Separated loss functions for regression and classification tasks
# ---------------------------------------------------------------------


import numpy as np
from ..utils import ulist

# ---------------------------------------------------------------------
#                        Regression Losses

# Loss functions for regression tasks
# ---------------------------------------------------------------------


def mean_ppow_error(y_true: np.array, y_pred: np.array, p: float) -> np.array:
  """
  computes the mean p-power error between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values
      p (_type_): float, the power to raise the error to

  Returns:
      float(_type_): the mean p-power error between the true and predicted values
  """

  if y_true.shape[1] == 1 and y_pred.shape[1] == 1:
    return np.mean(np.abs(y_true - y_pred)**p) ** (1/p)
  elif y_true.shape == y_pred.shape: 
    return np.mean(np.sum(np.abs(y_true - y_pred)**p, axis=1) ** (1/p))
  else:
    raise ValueError("y_true and y_pred must have the same shape")
  
  
def mean_squared_error(y_true: np.array, y_pred: np.array) -> np.array:
  """
  computes the mean squared error between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values

  Returns:
      np.array(_type_): the mean squared error between the true and predicted values
  """
  return mean_ppow_error(y_true, y_pred, 2)


def mean_absolute_error(y_true: np.array, y_pred: np.array) -> np.array:
  """
  computes the mean absolute error between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values

  Returns:
      np.array(_type_): the mean absolute error between the true and predicted values
  """
  return mean_ppow_error(y_true, y_pred, 1)


# ---------------------------------------------------------------------
#                        Classification Losses
#
# Loss functions for classification tasks
# ---------------------------------------------------------------------


def cross_entropy(y_true: np.array, y_pred: np.array, classes: list[int] = [0, 1]) -> np.array:
  """
  computes the log loss between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values
      classes (_type_): list[int], the classes to compute the cross entropy
      for, default is binary classification [0, 1], for multi-class [0, 1, ..., n-1]
      duplicates are reduced to unique values

  Returns:
      np.array(_type_): the log loss between the true and predicted values
  """
  
  if not ulist.increasing_by_one(classes, idx = 0):
    raise ValueError("classes must be increasing by one starting at 0")
  if y_true.shape[1] != y_pred.shape:
    raise ValueError("y_true and y_pred must have the same shape")
  if y_true.shape[1] < 1:
    raise ValueError("y_true and y_pred must have at least one feature")
  if y_true.shape[1] == 1:
    return binary_cross_entropy(y_true, y_pred)
  if y_true.shape[1] > 1:
    return n_cross_entropy(y_true, y_pred)
    

def binary_cross_entropy(y_true: np.array, y_pred: np.array) -> np.array:
  """
  computes the binary cross entropy between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values

  Returns:
      np.array(_type_): the binary cross entropy between the true and predicted values
  """

  if y_true.shape != y_pred.shape:
    raise ValueError("y_true and y_pred must have the same shape")
  if y_true.shape[1] != 1:
    raise ValueError("y_true and y_pred must have exactly one output feature")
  elif y_true.shape[1] == 1:
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
  else: 
    return -np.mean(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))  


def n_cross_entropy(y_true: np.array, y_pred: np.array, n: int) -> np.array:
  """
    computes the n-class cross entropy loss, classes indexed as [0, ..., n-1
  
  Args:
      y_true (np.array): A (n, m) array of true values, 
      m features, with eage feature having the same n classes
      y_pred (np.array): _description_
      n (int): _description_

  

  Returns:
      np.array: _description_
  """
  

def hinge(y_true: np.array, y_pred: np.array) -> np.array:
  """
  computes the hinge loss between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values

  Returns:
      np.array(_type_): the hinge loss between the true and predicted values
  """
  if y_true.shape != y_pred.shape:
    raise ValueError("y_true and y_pred must have the same shape")
  if y_true.shape[1] < 1:
    raise ValueError("y_true and y_pred must have at least one feature")
  if y_true.shape[1] == 1:
    return binary_hinge(y_true, y_pred)
  if y_true.shape[1] > 1:
    return n_hinge(y_true, y_pred)


def hinge_ppow(y_true: np.array, y_pred: np.array, p: float) -> np.array:
  """
  computes the pth-power of hinge loss between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values
      p (_type_): float, the power to raise the hinge loss to
    
  Returns:
      np.array(_type_): the p-power hinge loss between the true and predicted values
  """
  if y_true.shape != y_pred.shape:
    raise ValueError("y_true and y_pred must have the same shape")
  if y_true.shape[1] < 1:
    raise ValueError("y_true and y_pred must have at least one feature")
  if y_true.shape[1] == 1:
    return binary_hinge_ppow(y_true, y_pred, p)
  if y_true.shape[1] > 1:
    return n_hinge_ppow(y_true, y_pred, p)
  


def binary_hinge(y_true: np.array, y_pred: np.array) -> np.array:
  """
  computes the hinge loss between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values

  Returns:
      np.array(_type_): the hinge loss between the true and predicted values
  """
  if y_true.shape[1] == 1 and y_pred.shape[1] == 1:
    return np.mean(np.maximum(0, 1 - y_true * y_pred))
  elif y_true.shape == y_pred.shape: 
    return np.mean(np.sum(np.maximum(0, 1 - y_true * y_pred), axis=1))
  raise ValueError("y_true and y_pred must have the same shape")


def binary_hinge_ppow(y_true: np.array, y_pred: np.array, p: float) -> np.array:
  """
  computes the pth-power of hinge loss between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values
      p (_type_): float, the power to raise the hinge loss to
    
  Returns:
      np.array(_type_): the p-power hinge loss between the true and predicted values
  """
  if y_true.shape[1] == 1 and y_pred.shape[1] == 1:
    return (np.maximum(0, 1 - y_true * y_pred)) ** p
  elif y_true.shape == y_pred.shape:
    return (np.sum(np.maximum(0, 1 - y_true * y_pred), axis=1)) ** p
  
  
def n_hinge(y_true: np.array, y_pred: np.array) -> np.array:
  """
  computes the m-feature hinge loss between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values

  Returns:
      np.array(_type_): the m-feature hinge loss between the true and predicted values
  """
  if y_true.shape[1] == 1 and y_pred.shape[1] == 1:
    return np.mean(np.maximum(0, 1 - y_true * y_pred))
  elif y_true.shape == y_pred.shape: 
    return np.mean(np.sum(np.maximum(0, 1 - y_true * y_pred), axis=1))
  raise ValueError("y_true and y_pred must have the same shape")


def n_hinge_ppow(y_true: np.array, y_pred: np.array, p: float) -> np.array:
  """
  computes the pth-power of m-feature hinge loss between the true and predicted values
  
  Args:
      y_true (_type_): np.array of shape (n, m), the true values
      y_pred (_type_): np.array of shape (n, m), the predicted values
      p (_type_): float, the power to raise the hinge loss to
    
  Returns:
      np.array(_type_): the pth-power m-feature hinge loss between the true and predicted values
  """
  if y_true.shape[1] == 1 and y_pred.shape[1] == 1:
    return (np.maximum(0, 1 - y_true * y_pred)) ** p
  elif y_true.shape == y_pred.shape:
    return (np.sum(np.maximum(0, 1 - y_true * y_pred), axis=1)) ** p