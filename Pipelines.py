import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from scipy.stats import yeojohnson
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import BinaryEncoder
import time



all_cols = ['Country', 'Year', 'Status', 'Adult Mortality', 'infant deaths',
           'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
           'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
           ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
           ' thinness 5-9 years', 'Income composition of resources', 'Schooling']


class ColumnSelector(TransformerMixin, BaseEstimator):
    """
    Transformer that selects only the specified columns from a data frame.

    Parameters:
    ----------
    columns : list or array-like, default=None
        List of column names to select from the input data frame. If None, all columns are selected.
    """
    def __init__(self, columns=all_cols):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X[self.columns]
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

    
    
class AddColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class to add two columns in a Pandas DataFrame and drop the original columns.

    Parameters
    ----------
    col1 : str
        The name of the first column to add.
    col2 : str
        The name of the second column to add.
    new_col_name : str
        Name for the new column (defult = 'New column')

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the added column and the original columns dropped.
    """
    def __init__(self, col1, col2, new_col_name='New column'):
        self.col1 = col1
        self.col2 = col2
        self.new_col_name = new_col_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        X_new[self.new_col_name] = X_new[self.col1] + X_new[self.col2]
        return X_new


    
class DataFrameImputer(TransformerMixin, BaseEstimator):
    """
    A class to impute missing values in a Pandas DataFrame using a combination of median, knn, and most frequent
    imputers on specified columns.
    
    Parameters:
    -----------
    median_cols : list of str, optional (default = None)
        Columns to impute missing values using the median imputer.
    knn_cols : list of str, optional (default = None)
        Columns to impute missing values using the KNN imputer.
    freq_cols : list of str, optional (default = None)
        Columns to impute missing values using the most frequent imputer.
    
    Returns:
    --------
    X_imputed : Pandas DataFrame
        A DataFrame with imputed missing values.
    """
    def __init__(self, median_cols=None, knn_cols=None, freq_cols=None):
        self.median_cols = median_cols
        self.knn_cols = knn_cols
        self.freq_cols = freq_cols
    
    def fit(self, X, y=None):
        self.median_imputer = SimpleImputer(strategy='median')
        self.knn_imputer = KNNImputer()
        self.freq_imputer = SimpleImputer(strategy='most_frequent')
        self.median_imputer.fit(X[self.median_cols])
        self.knn_imputer.fit(X[self.knn_cols])
        self.freq_imputer.fit(X[self.freq_cols])
        return self
    
    def transform(self, X):
        X_median = pd.DataFrame(self.median_imputer.transform(X[self.median_cols]), 
                                columns=self.median_cols, index=X.index)
        X_knn = pd.DataFrame(self.knn_imputer.transform(X[self.knn_cols]), 
                             columns=self.knn_cols, index=X.index)
        X_freq = pd.DataFrame(self.freq_imputer.transform(X[self.freq_cols]), 
                              columns=self.freq_cols, index=X.index)
        X_imputed = pd.concat([X_median, X_knn, X_freq], axis=1)
        X_imputed = X_imputed.reindex(X.columns, axis=1)
        return X_imputed
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that drops specified columns from a DataFrame.

    Parameters
    ----------
    cols : list
        A list of column names to be dropped.
    return
    ------
        dataframe with dropped columns
    """
    def __init__(self, cols=None):
        self.cols = cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.cols is None:
            return X
        else:
            return X.drop(self.cols,axis=1)
        
        
class WinsorizationImpute(BaseEstimator, TransformerMixin):
    """
    A transformer that performs winsorization imputation on specified columns in a Pandas DataFrame.

    Parameters:
    -----------
    p : float, default=0.05
        The percentile value representing the lower bound for winsorization.
    q : float, default=0.95
        The percentile value representing the upper bound for winsorization.
    random_state : int, default=42
        Seed for the random number generator used for imputing missing values.
    cols : list, default=None
        The list of names of columns to be winsorized.

    Returns:
    --------
    A new Pandas DataFrame with the specified columns winsorized.

    """
    def __init__(self, p=0.05, q=0.95, random_state=42, cols=None):
        self.p = p
        self.q = q
        self.random_state = random_state
        self.cols = cols
        
    def fit(self, X, y=None):
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.cols:
            lower_bound = np.percentile(X[col], self.p * 100)
            upper_bound = np.percentile(X[col], self.q * 100)
            self.lower_bounds_[col] = lower_bound
            self.upper_bounds_[col] = upper_bound
        return self
    
    def transform(self, X):
        X_winsorized = X.copy()
        for col in self.cols:
            lower_bound = self.lower_bounds_[col]
            upper_bound = self.upper_bounds_[col]
            outliers_mask = (X_winsorized[col] < lower_bound) | (X_winsorized[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                random_values = np.random.normal(loc=X_winsorized[col].mean(), scale=X_winsorized[col].std(), size=outliers_count)
                random_values = np.clip(random_values, lower_bound, upper_bound)
                X_winsorized.loc[outliers_mask, col] = random_values
        return X_winsorized

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class LogTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply a log transform to a specified column in a Pandas DataFrame.

    Parameters
    ----------
    col : str
        The name of the column to apply the log transform to.
    domain_shift : float
        The value to be added to the column before applying the log transform.
        
    return
    ------
        transformed feature
    """
    def __init__(self, col, domain_shift=0):
        self.col = col
        self.domain_shift = domain_shift

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.col] = np.log(X_copy[self.col] + self.domain_shift)
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)
    


class YeoJohnsonTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply a Yeo-Johnson transform to a specified column in a Pandas DataFrame.

    Parameters
    ----------
    col : str
        The name of the column to apply the transform to.
    domain_shift : float
        The value to be added to the column before applying the transform.
        
    return
    ------
        transformed feature
    """
    def __init__(self, col, domain_shift=0):
        self.col = col
        self.domain_shift = domain_shift
        self.transformer_ = None

    def fit(self, X, y=None):
        self.transformer_ = PowerTransformer(method='yeo-johnson', standardize=False)
        self.transformer_.fit(X.loc[:, self.col].values.reshape(-1, 1) + self.domain_shift)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.col] = self.transformer_.transform(X.loc[:, self.col].values.reshape(-1, 1) + self.domain_shift)
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    
class LabelEncodeColumns(BaseEstimator, TransformerMixin):
    """
    A transformer class to encode categorical columns using LabelEncoder.

    Parameters
    ----------
    cols : list of str
        The names of the columns to be encoded.

    return
    ------
        encoded feature
    """
    def __init__(self, cols):
        self.cols = cols
        self.encoders_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            encoder = LabelEncoder()
            encoder.fit(X[col])
            self.encoders_[col] = encoder
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders_.items():
            X_copy[col] = encoder.transform(X_copy[col])
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    
class BinaryEncodeColumns(BaseEstimator, TransformerMixin):
    """
    A transformer class to encode categorical columns using BinaryEncoder.

    Parameters
    ----------
    cols : list of str
        The names of the columns to be encoded.

    return
    ------
        encoded feature
    """
    def __init__(self, cols):
        self.cols = cols
        self.encoders_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            encoder = BinaryEncoder()
            encoder.fit(X[col])
            self.encoders_[col] = encoder
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders_.items():
            encoded_cols = encoder.transform(X_copy[col])
            encoded_cols.columns = [f'{col}_bin_{i}' for i in range(encoded_cols.shape[1])]
            X_copy = pd.concat([X_copy, encoded_cols], axis=1)
            X_copy = X_copy.drop(col, axis=1)
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

class StandardScaleTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply standard scaling to specified columns in a Pandas DataFrame.

    Parameters
    ----------
    cols : list of str
        The names of the columns to apply standard scaling to.
    """
    def __init__(self, cols):
        self.cols = cols
        self.scaler_ = None

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.cols] = self.scaler_.transform(X_copy.loc[:, self.cols])
        return X_copy

    def fit_transform(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
        return self.transform(X)
    
    
    
class FullPipeline1:
    """
    1 - add_cols: ['Hepatitis B', 'Polio']
    2 - imputer: ['Year','Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
                  'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
                  ' HIV/AIDS', 'GDP', ' thinness 1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                  'vaccinations', 'Population', 'Status', 'Country']
    3 - drop_cols: ['infant deaths', 'percentage expenditure', ' thinness 5-9 years',
                    'Schooling', 'Diphtheria ', 'Year', 'Hepatitis B', 'Polio', 'Diphtheria ']
    4 - winsorization: ['Income composition of resources']
    5 - transform_vaccinations: ['vaccinations']
    6 - transform_adult_mortality: ['Adult Mortality']
    7 - transform_thinness_1_19: ['thinness 1-19 years']
    8 - transform_gdp: ['GDP]'
    9 - transform_hiv_aids: ['HIV/AIDS']
    10 - transform_population: ['Population']
    11 - transform_measles: ['Measles']
    12 - transform_under_five: ['under-five deaths ']
    13 - label_encode: ['Status', 'Country']
    14 - scale: ['Country', 'Status', 'Adult Mortality', 'Alcohol', 'Measles ', ' BMI ',
                 'under-five deaths ', 'Total expenditure', ' HIV/AIDS', 'GDP', 'Population',
                 ' thinness 1-19 years', 'Income composition of resources', 'vaccinations']
    """
    
    def __init__(self):
        self.add_cols = ['Hepatitis B', 'Polio']
        self.median_cols = ['Year','Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
                            'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS',
                            'GDP', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                            'vaccinations']
        self.knn_cols = ['Population']
        self.freq_cols = ['Status','Country']
        self.drop_cols = ['infant deaths', 'percentage expenditure', ' thinness 5-9 years', 'Schooling', 'Diphtheria ',
                          'Year', 'Hepatitis B', 'Polio', 'Diphtheria ']
        self.winsorization_cols = ['Income composition of resources']
        self.encode_cols = ['Status','Country']
        self.scale_cols = ['Country', 'Status', 'Adult Mortality', 'Alcohol', 'Measles ', ' BMI ', 'under-five deaths ',
                           'Total expenditure', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
                           'Income composition of resources', 'vaccinations']
        
        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=all_cols)),
            ('add_cols', AddColumnsTransformer(col1='Hepatitis B', col2='Polio', new_col_name='vaccinations')),
            ('imputer', DataFrameImputer(median_cols=self.median_cols, knn_cols=self.knn_cols, freq_cols=self.freq_cols)),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols)),
            ('winsorization', WinsorizationImpute(cols=self.winsorization_cols)),
            ('transform_vaccinations', YeoJohnsonTransform(col='vaccinations', domain_shift=100)),
            ('transform_adult_mortality', LogTransform(col='Adult Mortality', domain_shift=20)),
            ('transform_thinness_1_19', LogTransform(col=' thinness  1-19 years', domain_shift=1)),
            ('transform_gdp', LogTransform(col='GDP', domain_shift=0)),
            ('transform_hiv_aids', LogTransform(col=' HIV/AIDS', domain_shift=0)),
            ('transform_population', LogTransform(col='Population', domain_shift=80000)),
            ('transform_measles', LogTransform(col='Measles ', domain_shift=0.1)),
            ('transform_under_five', LogTransform(col='under-five deaths ', domain_shift=0.1)),
            ('label_encode', LabelEncodeColumns(cols=self.encode_cols)),
            ('scale', StandardScaleTransform(self.scale_cols))
        ])
    
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['Life expectancy ']))
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test
    

    
class FullPipeline2:
    """
    1 - add_cols: ['Hepatitis B', 'Polio']
    2 - imputer: ['Year','Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
                  'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
                  ' HIV/AIDS', 'GDP', ' thinness 1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                  'vaccinations', 'Population', 'Status', 'Country']
    3 - drop_cols: ['infant deaths', 'percentage expenditure', ' thinness 5-9 years',
                    'Schooling', 'Diphtheria ', 'Hepatitis B', 'Polio', 'Diphtheria ']
    4 - winsorization: ['Income composition of resources']
    5 - transform_vaccinations: ['vaccinations']
    6 - transform_adult_mortality: ['Adult Mortality']
    7 - transform_thinness_1_19: ['thinness 1-19 years']
    8 - transform_gdp: ['GDP]'
    9 - transform_hiv_aids: ['HIV/AIDS']
    10 - transform_population: ['Population']
    11 - transform_measles: ['Measles']
    12 - transform_under_five: ['under-five deaths ']
    13 - label_encode: ['Status', 'Country']
    14 - scale: ['Country', 'Status', 'Adult Mortality', 'Alcohol', 'Measles ', ' BMI ',
                 'under-five deaths ', 'Total expenditure', ' HIV/AIDS', 'GDP', 'Population',
                 ' thinness 1-19 years', 'Income composition of resources', 'vaccinations']
    """
    
    def __init__(self):
        self.add_cols = ['Hepatitis B', 'Polio']
        self.median_cols = ['Year','Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
                            'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS',
                            'GDP', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                            'vaccinations']
        self.knn_cols = ['Population']
        self.freq_cols = ['Status','Country']
        self.drop_cols = ['infant deaths', 'percentage expenditure', ' thinness 5-9 years', 'Schooling', 'Diphtheria ',
                          'Hepatitis B', 'Polio', 'Diphtheria ']
        self.winsorization_cols = ['Income composition of resources']
        self.encode_cols = ['Status','Country']
        self.scale_cols = ['Country', 'Status', 'Adult Mortality', 'Alcohol', 'Measles ', ' BMI ', 'under-five deaths ',
                           'Total expenditure', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
                           'Income composition of resources', 'vaccinations','Year']
        
        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=all_cols)),
            ('add_cols', AddColumnsTransformer(col1='Hepatitis B', col2='Polio', new_col_name='vaccinations')),
            ('imputer', DataFrameImputer(median_cols=self.median_cols, knn_cols=self.knn_cols, freq_cols=self.freq_cols)),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols)),
            ('winsorization', WinsorizationImpute(cols=self.winsorization_cols)),
            ('transform_vaccinations', YeoJohnsonTransform(col='vaccinations', domain_shift=100)),
            ('transform_adult_mortality', LogTransform(col='Adult Mortality', domain_shift=20)),
            ('transform_thinness_1_19', LogTransform(col=' thinness  1-19 years', domain_shift=1)),
            ('transform_gdp', LogTransform(col='GDP', domain_shift=0)),
            ('transform_hiv_aids', LogTransform(col=' HIV/AIDS', domain_shift=0)),
            ('transform_population', LogTransform(col='Population', domain_shift=80000)),
            ('transform_measles', LogTransform(col='Measles ', domain_shift=0.1)),
            ('transform_under_five', LogTransform(col='under-five deaths ', domain_shift=0.1)),
            ('label_encode', LabelEncodeColumns(cols=self.encode_cols)),
            ('scale', StandardScaleTransform(self.scale_cols))
        ])
        
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['Life expectancy '])),
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test
    

class FullPipeline3:
    """
    1 - add_cols: ['Hepatitis B', 'Polio']
    2 - imputer: ['Year','Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
                  'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
                  ' HIV/AIDS', 'GDP', ' thinness 1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                  'vaccinations', 'Population', 'Status', 'Country']
    3 - drop_cols: ['infant deaths', 'percentage expenditure', ' thinness 5-9 years',
                    'Schooling', 'Diphtheria ', 'Year', 'Hepatitis B', 'Polio', 'Diphtheria ']
    4 - winsorization: ['Income composition of resources']
    5 - transform_vaccinations: ['vaccinations']
    6 - transform_adult_mortality: ['Adult Mortality']
    7 - transform_thinness_1_19: ['thinness 1-19 years']
    8 - transform_gdp: ['GDP]'
    9 - transform_hiv_aids: ['HIV/AIDS']
    10 - transform_population: ['Population']
    11 - transform_measles: ['Measles']
    12 - transform_under_five: ['under-five deaths ']
    13 - winsorization: ['Adult Mortality','Total expenditure','GDP']
    14 - label_encode: ['Status', 'Country']
    15 - scale: ['Country', 'Status', 'Adult Mortality', 'Alcohol', 'Measles ', ' BMI ',
                 'under-five deaths ', 'Total expenditure', ' HIV/AIDS', 'GDP', 'Population',
                 ' thinness 1-19 years', 'Income composition of resources', 'vaccinations']
    """
    
    def __init__(self):
        self.add_cols = ['Hepatitis B', 'Polio']
        self.median_cols = ['Year','Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
                            'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS',
                            'GDP', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                            'vaccinations']
        self.knn_cols = ['Population']
        self.freq_cols = ['Status','Country']
        self.drop_cols = ['infant deaths', 'percentage expenditure', ' thinness 5-9 years', 'Schooling', 'Diphtheria ',
                          'Year', 'Hepatitis B', 'Polio', 'Diphtheria ']
        self.winsorization_cols1 = ['Income composition of resources']
        self.encode_cols = ['Status','Country']
        self.scale_cols = ['Country', 'Status', 'Adult Mortality', 'Alcohol', 'Measles ', ' BMI ', 'under-five deaths ',
                           'Total expenditure', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
                           'Income composition of resources', 'vaccinations']
        self.winsorization_cols2 = ['Adult Mortality','Total expenditure','GDP']
        
        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=all_cols)),
            ('add_cols', AddColumnsTransformer(col1='Hepatitis B', col2='Polio', new_col_name='vaccinations')),
            ('imputer', DataFrameImputer(median_cols=self.median_cols, knn_cols=self.knn_cols, freq_cols=self.freq_cols)),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols)),
            ('winsorization1', WinsorizationImpute(cols=self.winsorization_cols1)),
            ('transform_vaccinations', YeoJohnsonTransform(col='vaccinations', domain_shift=100)),
            ('transform_adult_mortality', LogTransform(col='Adult Mortality', domain_shift=20)),
            ('transform_thinness_1_19', LogTransform(col=' thinness  1-19 years', domain_shift=1)),
            ('transform_gdp', LogTransform(col='GDP', domain_shift=0)),
            ('transform_hiv_aids', LogTransform(col=' HIV/AIDS', domain_shift=0)),
            ('transform_population', LogTransform(col='Population', domain_shift=80000)),
            ('transform_measles', LogTransform(col='Measles ', domain_shift=0.1)),
            ('transform_under_five', LogTransform(col='under-five deaths ', domain_shift=0.1)),
            ('winsorization2', WinsorizationImpute(cols=self.winsorization_cols2)),
            ('label_encode', LabelEncodeColumns(cols=self.encode_cols)),
            ('scale', StandardScaleTransform(self.scale_cols))
        ])
    
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['Life expectancy '])),
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test

    

class FullPipeline4:
    """
    1 - add_cols: ['Hepatitis B', 'Polio']
    2 - imputer: ['Year','Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
                  'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
                  ' HIV/AIDS', 'GDP', ' thinness 1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                  'vaccinations', 'Population', 'Status', 'Country']
    3 - drop_cols: ['infant deaths', 'percentage expenditure', ' thinness 5-9 years',
                    'Schooling', 'Hepatitis B', 'Polio', 'Diphtheria ']
    4 - winsorization: ['Income composition of resources']
    5 - transform_vaccinations: ['vaccinations']
    6 - transform : ['Diphtheria']
    7 - winsorization: ['Diphtheria']
    8 - transform_adult_mortality: ['Adult Mortality']
    9 - transform_thinness_1_19: ['thinness 1-19 years']
    10 - transform_gdp: ['GDP]'
    11 - transform_hiv_aids: ['HIV/AIDS']
    12 - transform_population: ['Population']
    13 - transform_measles: ['Measles']
    14 - transform_under_five: ['under-five deaths ']
    15 - label_encode: ['Status', 'Country']
    16 - scale: ['Country', 'Status', 'Adult Mortality', 'Alcohol', 'Measles ', ' BMI ',
                 'under-five deaths ', 'Total expenditure', ' HIV/AIDS', 'GDP', 'Population',
                 ' thinness 1-19 years', 'Income composition of resources', 'vaccinations']
    """
    
    def __init__(self):
        self.add_cols = ['Hepatitis B', 'Polio']
        self.median_cols = ['Year','Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
                            'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS',
                            'GDP', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                            'vaccinations']
        self.knn_cols = ['Population']
        self.freq_cols = ['Status','Country']
        self.drop_cols = ['infant deaths', 'percentage expenditure', ' thinness 5-9 years', 'Schooling',
                          'Hepatitis B', 'Polio']
        self.winsorization_cols1 = ['Income composition of resources']
        self.encode_cols = ['Status','Country']
        self.scale_cols = ['Country', 'Status', 'Adult Mortality', 'Alcohol', 'Measles ', ' BMI ', 'under-five deaths ',
                           'Total expenditure', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
                           'Income composition of resources', 'vaccinations', 'Year', 'Diphtheria ']
        self.winsorization_cols2 = ['Diphtheria ']
        
        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=all_cols)),
            ('add_cols', AddColumnsTransformer(col1='Hepatitis B', col2='Polio', new_col_name='vaccinations')),
            ('imputer', DataFrameImputer(median_cols=self.median_cols, knn_cols=self.knn_cols, freq_cols=self.freq_cols)),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols)),
            ('winsorization1', WinsorizationImpute(cols=self.winsorization_cols1)),
            ('transform_vaccinations', YeoJohnsonTransform(col='vaccinations', domain_shift=100)),
            ('transform_Diphtheria', YeoJohnsonTransform(col='Diphtheria ', domain_shift=100)),
            ('winsorization2', WinsorizationImpute(cols=self.winsorization_cols2)),
            ('transform_adult_mortality', LogTransform(col='Adult Mortality', domain_shift=20)),
            ('transform_thinness_1_19', LogTransform(col=' thinness  1-19 years', domain_shift=1)),
            ('transform_gdp', LogTransform(col='GDP', domain_shift=0)),
            ('transform_hiv_aids', LogTransform(col=' HIV/AIDS', domain_shift=0)),
            ('transform_population', LogTransform(col='Population', domain_shift=80000)),
            ('transform_measles', LogTransform(col='Measles ', domain_shift=0.1)),
            ('transform_under_five', LogTransform(col='under-five deaths ', domain_shift=0.1)),
            ('label_encode', LabelEncodeColumns(cols=self.encode_cols)),
            ('scale', StandardScaleTransform(self.scale_cols))
        ])
        
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['Life expectancy '])),
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test
    
    

class FullPipeline5:
    """
    1 - add_cols: ['Hepatitis B', 'Polio']
    2 - imputer: ['Year','Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
                  'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
                  ' HIV/AIDS', 'GDP', ' thinness 1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                  'vaccinations', 'Population', 'Status', 'Country']
    3 - drop_cols: ['infant deaths', 'percentage expenditure', ' thinness 5-9 years',
                    'Schooling', 'Diphtheria ', 'Hepatitis B', 'Polio', 'Diphtheria ']
    4 - winsorization: ['Income composition of resources']
    5 - transform_vaccinations: ['vaccinations']
    6 - transform_adult_mortality: ['Adult Mortality']
    7 - transform_thinness_1_19: ['thinness 1-19 years']
    8 - transform_gdp: ['GDP]'
    9 - transform_hiv_aids: ['HIV/AIDS']
    10 - transform_population: ['Population']
    11 - transform_measles: ['Measles']
    12 - transform_under_five: ['under-five deaths ']
    13 - label_encode: ['Status']
    14 - binary_encode: ['Country']
    14 - scale: ['Status', 'Adult Mortality', 'Alcohol', 'Measles ', ' BMI ',
                 'under-five deaths ', 'Total expenditure', ' HIV/AIDS', 'GDP', 'Population',
                 ' thinness 1-19 years', 'Income composition of resources', 'vaccinations']
    """
    
    def __init__(self):
        self.add_cols = ['Hepatitis B', 'Polio']
        self.median_cols = ['Year','Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
                            'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS',
                            'GDP', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                            'vaccinations']
        self.knn_cols = ['Population']
        self.freq_cols = ['Status','Country']
        self.drop_cols = ['infant deaths', 'percentage expenditure', ' thinness 5-9 years', 'Schooling', 'Diphtheria ',
                          'Hepatitis B', 'Polio', 'Diphtheria ']
        self.winsorization_cols = ['Income composition of resources']
        self.label_encode_cols = ['Status']
        self.binary_encode_cols = ['Country']
        self.scale_cols = ['Status', 'Adult Mortality', 'Alcohol', 'Measles ', ' BMI ', 'under-five deaths ',
                           'Total expenditure', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
                           'Income composition of resources', 'vaccinations','Year']
        
        self.full_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=all_cols)),
            ('add_cols', AddColumnsTransformer(col1='Hepatitis B', col2='Polio', new_col_name='vaccinations')),
            ('imputer', DataFrameImputer(median_cols=self.median_cols, knn_cols=self.knn_cols, freq_cols=self.freq_cols)),
            ('drop_cols', DropColumnsTransformer(cols=self.drop_cols)),
            ('winsorization', WinsorizationImpute(cols=self.winsorization_cols)),
            ('transform_vaccinations', YeoJohnsonTransform(col='vaccinations', domain_shift=100)),
            ('transform_adult_mortality', LogTransform(col='Adult Mortality', domain_shift=20)),
            ('transform_thinness_1_19', LogTransform(col=' thinness  1-19 years', domain_shift=1)),
            ('transform_gdp', LogTransform(col='GDP', domain_shift=0)),
            ('transform_hiv_aids', LogTransform(col=' HIV/AIDS', domain_shift=0)),
            ('transform_population', LogTransform(col='Population', domain_shift=80000)),
            ('transform_measles', LogTransform(col='Measles ', domain_shift=0.1)),
            ('transform_under_five', LogTransform(col='under-five deaths ', domain_shift=0.1)),
            ('label_encode', LabelEncodeColumns(cols=self.label_encode_cols)),
            ('binary_encode', BinaryEncodeColumns(cols=self.binary_encode_cols)),
            ('scale', StandardScaleTransform(self.scale_cols))
        ])
        
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=['Life expectancy '])),
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.full_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.full_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test
    
    

class ModelEvaluate:
    """
    A class that takes a list of regression models and evaluates their performance
    using cross-validation.

    Parameters
    ----------
    models : list
        A list of regression models to evaluate.

    Methods
    -------
    fit(X_train, y_train)
        Fits the regression models on the training data using cross-validation and
        stores the evaluation results in the 'results' attribute.

    get_results()
        Returns the evaluation results as a pandas DataFrame.
    """
    def __init__(self, models):
        self.models = models

    def fit(self, X_train, y_train):
        self.results = []
        for model in self.models:
            start = time.time()
            scores = cross_validate(model, X_train, y_train, cv=5,
                                    scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
                                    return_train_score=False)
            end = time.time()
            results_dict = {}
            results_dict['model'] = model.__class__.__name__
            results_dict['mean_mae'] = -np.mean(scores['test_neg_mean_absolute_error'])
            results_dict['mean_rmse'] = np.sqrt(-np.mean(scores['test_neg_mean_squared_error']))
            results_dict['mean_r2'] = np.mean(scores['test_r2'])
            results_dict['time'] = end - start
            self.results.append(results_dict)

    def get_results(self):
        return pd.DataFrame(self.results)    



class RegressionPlot:
    """A class for creating a set of plots to visualize the performance of a regression model.

    Parameters
    ----------
    y_test : pandas.DataFrame
        The actual target values for the test set.
    y_pred : array-like
        The predicted target values for the test set.
    color : str, optional
        The color to use for the plot markers and histograms.

    Methods
    -------
    plot()
        Creates a set of three plots to visualize the performance of the regression model.

    """

    def __init__(self, y_test, y_pred, color='b'):
        self.y_test = y_test
        self.y_pred = y_pred
        self.color = color
    
    def plot(self):
        """Creates a set of three plots to visualize the performance of the regression model.

        The three plots are: a scatter plot with regression line, a histogram of errors, and a residual plot.
        """

        # Create subplots
        fig, axs = plt.subplots(ncols=3, figsize=(15,5))

        # Plot scatter plot with regression line
        axs[0].scatter(self.y_test[self.y_test.columns[0]], self.y_pred, color=self.color)
        axs[0].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=3, color='k')
        axs[0].set_xlabel('Actual Values')
        axs[0].set_ylabel('Predicted Values')
        axs[0].set_title('Scatter Plot with Regression Line')

        # Plot histogram of errors
        errors = self.y_test[self.y_test.columns[0]] - self.y_pred
        axs[1].hist(errors, bins=50, color=self.color)
        axs[1].axvline(x=errors.median(), color='k', linestyle='--', lw=3)
        axs[1].set_xlabel('Errors')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Histogram of Errors')

        # Plot residual plot
        axs[2].scatter(self.y_pred, errors, color=self.color)
        axs[2].axhline(y=0, color='k', linestyle='-', lw=3)
        axs[2].set_xlabel('Predicted Values')
        axs[2].set_ylabel('Errors')
        axs[2].set_title('Residual Plot')

        # Show the plots
        plt.tight_layout()
        plt.show()