import copy 
from collections import defaultdict
import numpy as np

class BaseRegressor:
  def score(self, X, y):
    y_pred = self.predict(X)
    return self._score(y, y_pred)

class BaseClassifier:
  def score(self, X, y):
    y_pred = self.predict(X)
    return self._score(y, y_pred)

class BaseCluster:
  pass
