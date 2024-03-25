import numpy as np

def heaviside(X: np.array) -> np.array:
    """
    computes the Heaviside step function of the input
    Args:
        X (_type_): np.array of shape (n, m), the input to the activation function
    Returns:
        np.array(_type_): Outputs in range {0, 1}
    """
    return np.heaviside(X, 0)


def relu(X: np.array) -> np.array:
  """
  computes the rectified linear unit of the input
  Args:
      X (_type_): np.array of shape (n, m), the input to the activation function
  Returns:
      np.array(_type_): Outputs in range [0, inf)
  """
  return np.maximum(0, X)


def leaky_relu(X: np.array, alpha: float = 0.01) -> np.array:
    """
    computes the leaky rectified linear unit of the input
    Args:
        X (np.array): np.array of shape (n, m), the input to the activation function
        alpha (float, optional): scaling parameter. Defaults to 0.01.

    Returns:
        np.array: Outputs in range (-inf, inf)
    """
    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    
    return np.where(X > 0, X, X * alpha)
    

def softmax(X: np.array) -> np.array:
  """
  computes the softmax of the input
  Args:
      X (_type_): np.array of shape (n, m), the input to the activation function
  Returns:
      np.array(_type_): Outputs in range [0, 1]
  """
  exp_X = np.exp(X)
  return exp_X / np.sum(exp_X, axis=1, keepdims=True)


def sigmoid(X: np.array) -> np.array:
  """
  computes the sigmoid of the input
  Args:
      X (_type_): np.array of shape (n, m), the input to the activation function
  Returns:
      np.array(_type_): Ouputs in range [-e, inf)
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


def softplus(X: np.array) -> np.array:
  """
  computes the softplus of the input
  Args:
      X (_type_): np.array of shape (n, m), the input to the activation function
  Returns:
      np.array(_type_): Ouputs in range [0, inf)
  """
  return np.log(1 + np.exp(X))