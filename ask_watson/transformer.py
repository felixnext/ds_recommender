'''Various transformers to prepare data for the estimators.'''


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
  '''Allows to select a subset of columns from a pandas dataframe.

  Args:
    cols (list): str or list of str that indicates the columns to select
  '''
  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return X[self.cols]


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


class SimilarityTransformer(BaseEstimator, TransformerMixin):
  '''Transformer that constructs a similarity matrix.

  Args:
    cols (list): List of columns (int or str) used as id - if tuple take the indexed ids from first to second element
    preserve_ids (bool): Defines if the original index should be preserved (as index and columns)
    index_col (str): Alternatively provide the name of a column to use as index
    remove_duplicates (bool): Defines if an additional duplicate removal should be done
  '''
  def __init__(self, cols=None, preserve_idx=True, index_col=None, remove_duplicates=False):
    self.cols = cols
    self.preserve_idx = preserve_idx
    self.index_col = index_col
    self.dedup = remove_duplicates

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    '''Performs the dot product on the input matrix to create a similarity score.'''
    # TODO: additional error checks
    # retrieve the relevant data
    mat = X
    if self.cols is not None:
      if isinstance(self.cols, tuple):
        mat = X.iloc[:, self.cols[0]:self.cols[1]]
      elif self.cols:
        mat = X.iloc[:, self.cols]
      else:
        mat = X.loc[:, self.cols]

    # optional remove duplicates
    if self.dedup:
      mat = mat.remove_duplicates()

    # perform the actual transformation
    mat = mat.dot(np.transpose(mat))

    # check for index update
    if self.index_col is not None:
      idx = X[self.index_col]
      mat.index = idx
      mat.columns = idx
    elif self.preserve_idx == True:
      idx = np.array(X.index)
      mat.index = idx
      mat.columns = idx

    return mat


class RankingTransformer(BaseEstimator, TransformerMixin):
  '''Ranks the items according to the provided criteria, allowing a successive recommender to filter them and output recommendations.

  Args:
    ranking_cols (list): List of str of the columns that should be used for rating
    min_count (int): Minimal number of rating elements to consider an element (if None consider all elements)
    id_col (str): Column to used for the id (if `None` use the index)
    ascending (bool): Defines if the elements should be order ascending (might also be an array of same length as rating cols)
  '''
  def __init__(self, ranking_cols, min_count=None, id_col=None, ascending=False):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    '''Transforms the given list of user interaction datapoints into a ranked list of items.

    Args:
      X (pd.DataFrame): Dataframe that contains the ids of the elements and the ratings to be considered

    Returns:
      DataFrame of Ranked items with column item_id, score
    '''
    pass
