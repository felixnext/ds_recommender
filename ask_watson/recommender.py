'''Base Class for a recommender (derived from sklearn estimators).'''


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.multioutput import MultiOutputClassifier


class BaseRecommender(BaseEstimator, ClassifierMixin):
  '''Defines the basic structure for an Recommender.

  TODO: describe how the recommender interacts with data
  '''
  def __init__(self):
    pass

  def fit(self, X, y, sample_weight=None):
    return self

  def predict(self, X):
    pass

  def predict_proba(self, X):
    pass

  def predict_log_proba(self, X):
    pass

  def score(self, X, y, sample_weight=None):
    pass


class SimilarityRecommender(MultiOutputClassifier):
  '''Recommender for similar items based on the `SimilarityTransformer`.

  Args:
    num_items (int): Number of items that should be estimated
  '''
  def __init__(self, num_items, ascending=False):
    self.num_items = num_items if num_items is not None else -1
    self.ascending = ascending

  def fit(self, X, y=None, sample_weight=None):
    # store the similarity matrix internally
    self.sim_mat = X
    return self

  def predict(self, X):
    '''Predicts the desired number of outputs.

    Returns:
      Array of shape (X.shape[0], num_items)
    '''
    res = []
    for item in X:
      try:
        pred = self.sim_mat.loc[item].sort_values(ascending=self.ascending)
        pred = pred.drop(labels=[item])
        res.append(pred.index[:self.num_items])
      except:
        res.append(np.full(self.num_items, np.nan))
    return np.array(res)

  def predict_proba(self, X):
    '''Predicts the desired number of outputs and returns normalized scores with them.

    '''
    res = []
    for item in X:
      try:
        pred = self.sim_mat.loc[item].sort_values(ascending=self.ascending)
        pred = pred.drop(labels=[item])
        res.append(list(zip(pred.index[:self.num_items], pred.values[:self.num_items])))
      except:
        res.append(np.full(self.num_items, (np.nan, np.nan)))
    return np.array(res)
