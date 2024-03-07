import numpy as np


def relu(X: np.array) -> np.array:
  """
  computes the rectified linear unit of the input
  Args:
      X (_type_): np.array of shape (n, m), the input to the activation function
  Returns:
      np.array(_type_): the rectified linear unit of the input
  """
  return np.maximum(0, X)


def softmax(X: np.array) -> np.array:
  """
  computes the softmax of the input
  Args:
      X (_type_): np.array of shape (n, m), the input to the activation function
  Returns:
      np.array(_type_): the softmax of the input
  """
  exp_X = np.exp(X)
  return exp_X / np.sum(exp_X, axis=1, keepdims=True)


def sigmoid(X: np.array) -> np.array:
  """
  computes the sigmoid of the input
  Args:
      X (_type_): np.array of shape (n, m), the input to the activation function
  Returns:
      np.array(_type_): the sigmoid of the input
  """
  return 1 / (1 + np.exp(-X))


def tanh(X: np.array) -> np.array:
  """
  computes the hyperbolic tangent of the input
  Args:
      X (_type_): np.array of shape (n, m), the input to the activation function
  Returns:
      np.array(_type_): the hyperbolic tangent of the input
  """
  return np.tanh(X)
