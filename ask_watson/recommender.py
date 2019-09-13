'''Base Class for a recommender (derived from sklearn estimators).'''


from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin, MultiOutputMixin


class BaseRecommender(BaseEstimator, ClassifierMixin, MultiOutputMixin):
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
