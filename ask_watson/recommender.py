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

  The goal of this recommender is to recommend similar items to the items that are already given (item-item recommender).

  Args:
    num_items (int): Number of items that should be estimated (if None output all items)
  '''
  def __init__(self, num_items=None, ascending=False):
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


class CrossSimilarityRecommender(MultiOutputClassifier):
  '''Recommender that uses similarities along one dimension and uses them to recommend items along a different dimension.

  The goal of this recommender is to recommend items through the similarity of users (collaborative-filtering)

  Args:
    num_items (int): The number of items to recommend
    similarity (Recommender): Recommender that is used to generate the similarity scores between different users (should be a function with no input values)
  '''
  def __init__(self, num_items, similarity=SimilarityRecommender):
    self.num_items = num_items
    self.estimator = similarity()

  def fit(self, X, y=None, sample_weight=None):
    '''Stores structure of the dataset in the recommender.

    Args:
      X (tuple): Tuple of data items in form (user-item-matrix, user-user-similarity-matrix).
    '''
    # fit the internal estimator
    self.estimator.fit(X[1], y, sample_weight)
    self.user_item = X[0]
    return self

  def predict(self, X):
    '''Predicts list of `num_items` for the given user_ids.'''
    res = []
    # retrieve similar users for each input users
    sim_users = self.estimator.predict(X)
    for x, susers in zip(X, sim_users):
      # retrieve items the user already interacted with
      seen = self.user_item.loc[x]
      # TODO: highlight relevant items (how to select which the user has interacted with)

      # iterate through all similar users and find relevant items
      user_rec = []
      for user in susers:
        # TODO: find items relevant for this user

        # TODO: remove items already seen by current users
        cur_rec = []

        # TODO: combine items check exit condition
        if len(user_rec) + len(cur_rec) >= self.num_items:
          len_rem = self.num_items - len(user_rec)
          cur_rec = cur_rec[:len_rem]
          user_rec += np.sample
          break
        else:
          user_rec += cur_rec

      res.append(user_rec)
    return np.array(res)

  def predict_proba(self, X):
    pass
