'''Various Estimators for Recommender related values.'''


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin


class FunkSVDEstimator(BaseEstimator, RegressorMixin):
  '''Uses FunkSVD to interact with .

  Args:
    epochs (int): Number of epochs for the fit phase
    lr (float): Learning Rate used
    latent_features (int): number of latent features to learn through FunkSVD
    early_stop (bool): Defines if early stopping is enabled
    early_stop_thres (float): Threshold for early stopping (default: `0.0001`)
  '''
  def __init__(self, epochs=25, lr=0.005, latent_features=15, early_stop=False, early_stop_thres=0.0001):
    # set relevant values
    self.lr = lr
    self.epochs = epochs
    self.latent = latent_features
    self.early_stop = early_stop
    self.early_thres = early_stop_thres
    # generate random start matrix
    self.user_mat = None
    self.item_mat = None
    # TODO: retrieve list of IDs to match against items

  def fit(self, X, y=None):
    '''Fits the SVD matricies to the given values.

    Args:
      X (np.array): 2D Numpy Array that contains the user-item matrix
    '''
    # TODO: error checks

    # init with random data
    self.user_mat = np.random.rand(X.shape[0], self.latent)
    self.item_mat = np.random.rand(self.latent, X.shape[1])
    accum_sse = np.INF

    # iterate through all epochs
    for ep in range(self.epochs):
      # setup vals
      old_sse = accum_sse
      accum_sse = 0

      # For each user-movie pair
      for u_idx in range(X.shape[0]):
        for m_idx in range(X.shape[1]):
          # if the rating exists
          if not np.isnan(X[u_idx, m_idx]):
            # compute the error as the actual minus the dot product of the user and movie latent features
            act = X[u_idx, m_idx]
            pu = self.user_mat[u_idx, :]
            pm = self.movie_mat[:, m_idx]
            pred = np.sum(np.dot(pu, pm))
            accum_sse += (pred - act) ** 2

            # update the values in each matrix in the direction of the gradient
            pu_new = pu + self.lr * 2 * (act-pred) * pm
            pm_new = pm + self.lr * 2 * (act-pred) * pu_new
            self.user_mat[u_idx, :]  = pu_new
            self.movie_mat[:, m_idx] = pm_new

      # check for early stopping
      if self.early_stop == True and old_sse - accum_sse < self.early_thres:
        break

    return self

  def predict(self, X):
    '''Predicts the ratings for the given parings.

    Args:
      X (np.array): Array of data items to predict. Either single value tuple in form `(user_id, item_id)` or list of tuples

    Returns:
      Array of predicted values
    '''
    # check for init
    if self.user_mat is None or self.movie_mat is None:
      raise RuntimeError("FunkSVDEstimator has not been fitted!")

    # retrieve the relevant items to predict (if possible)
    if isinstance(X, tuple) or len(X.shape) == 1:
      X = [X]
    # TODO: additional error / type checking

    # iterate through predicitons
    res = []
    for row in X:
      # retrieve data
      user_id = np.where(self.user_idxs == row[0])[0]
      item_id = np.where(self.item_idxs == row[1])[0]
      # cannot predict if not in training
      if len(user_id) == 0 or len(item_id) == 0:
        res.append(np.nan)
      # predict value
      pred = np.dot(self.user_mat[user_id[0], :], self.item_mat[:, item_id[0]])
      res.append(pred)

    return np.array(res)

  def score(self, X, y=None):
    '''Provides the RMSE score for the given items.

    Args:
      X (np.array): set of validation data-points to check against - should be in format `user | item | value`

    Returns:
      RMSE score for all predictable items as a float
    '''
    # setup values
    rmse = 0
    n_items = 0

    # iterate through all values
    for row in X.iterrows():
      # cannot calc if no rating
      if np.isnan(row[2]): continue

      # retrieve data
      user_id = np.where(self.user_idxs == row[0])[0]
      item_id = np.where(self.item_idxs == row[1])[0]

      # cannot predict if not in training data
      if len(user_id) == 0 or len(item_id) == 0: continue

      # predict data
      pred = np.dot(self.user_mat[user_id[0], :], self.item_mat[:, item_id[0]])
      rmse += (act - pred) ** 2
      n_items += 1

    # check if
    if n_items == 0: return np.INF

    # calculate and return
    return np.sqrt(rmse/n_items)
