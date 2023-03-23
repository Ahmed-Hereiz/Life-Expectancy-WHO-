import colorsys
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def create_sequential_palette(num_colors, hue=0.5, saturation=0.8, value_start=0.4):
    """
    Creates a sequential color palette with the specified number of colors,
    using a base color with the specified hue, saturation, and value_start.
    """
    colors = [colorsys.hsv_to_rgb(hue, saturation, value_start + (i/num_colors)*(1-value_start)) for i in range(num_colors)]
    return sns.color_palette(colors)

def get_color():
    """
    Given a list of colors, returns the last color in the list.
    """
    palette = create_sequential_palette(5)
    return palette[-2]

def continuous_data_distribution(df, continuous_data):
    """
    A function to visualize continuous data in a dataframe.

    Parameters:
    df (pandas dataframe): The dataframe containing the data.
    continuous_data (list): A list of column names containing the continuous data.

    Returns:
    None
    """
    for i in continuous_data:
        if len(df[i].unique()) > 20:
            
            plt.figure()
            rcParams['figure.figsize'] = (20,3) 
            fig, (ax_box, ax_kde) = plt.subplots(ncols=2, sharex=True)
            plt.gca().set(xlabel= i,ylabel='Density')    

            # Create a boxplot and kde plot
            sns.boxplot(df[i], ax=ax_box , linewidth= 1.0, color=get_color())
            sns.kdeplot(df[i], ax=ax_kde , fill=True, color=get_color(),data=df)

            plt.show()

        else:
            
            plt.figure()
            rcParams['figure.figsize'] = (20,4) 
            fig, (ax_count) = plt.subplots(ncols=1, sharex=True)
            plt.gca().set(xlabel= i,ylabel='Density')    

            # Create a count plot
            sns.countplot(df[i], ax=ax_count , linewidth= 1.0,
                          palette=create_sequential_palette(len(df[i].unique())))

            plt.show()

    rcParams['figure.figsize'] = (20,10)
    
    
def countinous_data_scatterplot(df,target='Life expectancy '):
    
    fig = plt.figure(figsize=(20,15))
    counter = 0

    for i in df.columns:
        if df[i].dtype != 'object':
            sub = fig.add_subplot(5,4,counter+1)
            g = sns.scatterplot(x=i,y=target,data=df,color=get_color())
            counter = counter + 1

    plt.tight_layout()

    

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        self.unknown_value = -1
    
    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            le = self.encoders.get(col, None)
            if le is None:
                X_encoded[col] = self.unknown_value
            else:
                X_encoded[col] = le.transform(X[col])
        return X_encoded
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)



class WinsorizationImpute(BaseEstimator, TransformerMixin):
    def __init__(self, p=0.05, q=0.95, random_state=None):
        self.p = p
        self.q = q
        self.random_state = random_state
        
    def fit(self, X, y=None):
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in X.columns:
            lower_bound = np.percentile(X[col], self.p * 100)
            upper_bound = np.percentile(X[col], self.q * 100)
            self.lower_bounds_[col] = lower_bound
            self.upper_bounds_[col] = upper_bound
        return self
    
    def transform(self, X):
        X_winsorized = X.copy()
        for col in X.columns:
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
