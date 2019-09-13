'''Base Class for a recommender (derived from sklearn estimators).'''

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin

class BaseRecommender(BaseEstimator, ClassifierMixin):
  '''Defines the basic structure for an Recommender.

  TODO: describe how the recommender interacts with data
  '''
  pass

# TODO: define additional transformers (popular movies, etc.)
# TODO: use these transformers in predefined pipelines for the other recommenders e.g. Pipeline([FunkSVDTransformer, RatingEstimator])

class RatingEstimator(BaseEstimator, RegressorMixin):
  '''Uses FunkSVD to interact with .'''
  pass
