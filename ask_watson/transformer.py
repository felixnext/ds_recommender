'''Various transformers to prepare data for the estimators.'''


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, MultiOutputMixin


class UserItemTransformer(BaseEstimator, TransformerMixin):
  '''Transformer that constructs a user-item matrix.

  Args:
    user_col (str): Name of the column used for the user side
    item_col (str): Name of the column used for the itme side
    value_col (str): Name of the column that contains the matrix values
    agg_fct (str): Aggregation function that is used in case of duplicates
  '''
  def __init__(self, user_col, item_col, value_col, agg_fct='max'):
    self.user_col = user_col
    self.item_col = item_col
    self.value_col = value_col
    self.agg = agg_fct

  def fit(self, x, y=None):
    return self

  def transform(self, X):
    '''Transform the given input data into a user-item matrix.'''
    # safty: check if pandas dataframe
    if not isinstance(X, pd.DataFrame):
      print("Warning: Input is not a pandas.DataFrame, column selection might not be possible (due to missing names)...")
      X = pd.DataFrame(X)
    # perform transformation
    mat = X.groupby([self.user_col, self.item_col])[self.value_col].agg(self.agg).unstack()
    return mat
