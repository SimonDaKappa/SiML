import numpy as np


def increasing_by_one(lst: list, idx: int = 0) -> bool:
    """
    checks if a list is increasing by one, with optional starting point for index 0
    stronger version of sorted_from
    
    Args:
        lst (_type_): list[int], the non-null list to check
        idxt (_type_): int, the starting index to check from, default list start  
    Returns:
        bool(_type_): True if the list is increasing by one, False otherwise,
        
    """
    if lst[0] != idx:
      return False
    return all(lst[i] == lst[i-1] + 1 for i in range(1, len(lst)))
  
  
def sorted_from(lst: list, idx: int = 0) -> bool:
  """
  checks if a list is sorted from the optionally chosen index
  
  Args:
      lst (_type_): list[int], the non-null list to check 
      idx (_type_): int, the starting index to check from, default list start
      
  Returns:
      bool(_type_): True if the list is sorted from zero, False otherwise,
      stronger version 
  """
  if idx < 0:
    raise ValueError("idx must be a non-negative integer")
  return all(lst[i] == lst[i-1] + 1 for i in range(idx, len(lst)))