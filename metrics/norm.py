import numpy as np

def l1_norm(X: np.array):
  """
  computes the l1 norm of a vector
  
  Args:
      X (_type_): np.array of shape (n, m), calculating the norm of each row 

  Returns:
      np.array(_type_): 1-norm of each row in X 
  """
  return lp_norm(X, 1)


def l2_norm(X: np.array):
  """
  computes the euclidean distance norm in n-dimensions
  
  Args:
      X (_type_): np.array of shape (n, m), calculating the norm of each row 

  Returns:
      np.array(_type_): 2-norm of each row in X 
  """
  return lp_norm(X, 2) 


def l3_p_norm(X: np.array):
  """
  computes the l3 norm of a vector
  
  Args:
      X (_type_): np.array of shape (n, m), calculating the norm of each row 

  Returns:
      np.array(_type_): 3-norm of each row in X 
  """
  return lp_norm(X, 3)


def lp_norm(X: np.array, p: float) -> np.array:
  """
  returns a function that computes the l_p norm of a vector
  
  Args:
      p (_type_): int, the norm to compute
      X (_type_): np.array of shape (n, m), calculating the norm of each row
  Returns:
      callable(_type_): function that computes the l_p norm of a vector
  """
  
  if X.shape[1] == 1:
    return np.sum(np.abs(X)**p) ** (1/p)
  else:
    return np.sum(np.abs(X)**p, aXis=1) ** (1/p)


def frobenius_norm(X: np.array):
  """
  computes the frobenius norm of a matriX
  
  Args:
      X (_type_): np.array of shape (n, m), calculating the square root of
      matriX trace of (XX*), where H* is the conjugate transpose. 

  Returns:
      np.array(_type_): frobenius norm of matriX X 
  """
  return np.sqrt(np.trace(X @ X.H))

