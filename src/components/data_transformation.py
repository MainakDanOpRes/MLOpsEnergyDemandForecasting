import sys
import os
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.exception import CustomException
from src.utils.logger import logging

class DatetimeIndexer(BaseEstimator, TransformerMixin):
    """Converts Date and Time columns into a Datetime Index."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['Datetime'] = pd.to_datetime(X_transformed['Date'] + ' ' + X_transformed['Time'], format='%d/%m/%Y %H:%M:%S')
        X_transformed.drop(['Date', 'Time'], axis=1, inplace=True)
        X_transformed.set_index('Datetime', inplace=True)
        return X_transformed
    
class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values using time-based interpolation."""
    def __init__(self, method = 'time'):
        self.method = method

    def fit(self, X, y=None):
        # Imputers don't need to "learn" anything from the training data 
        # for interpolation, so we just return self.
        return self
    
    def transormation(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.interpolate(method=self.method)
        X_transformed.bfill(inplace=True)

        return X_transformed

class TimeSeriesResampler(BaseEstimator, TransformerMixin):
    """Resamples the data to hourly averages."""
    def __init__(self, frequency = 'h'):
        self.frequency = frequency
    
    def fit(self, X, y=None):
        # Imputers don't need to "learn" anything from the training data 
        # for interpolation, so we just return self.
        return self
    
    def transformation(self, X):
        freq_data = pd.DataFrame()
        freq_data['Global_active_power'] = X['Global_active_power'].resample(self.frequency).mean()

        return freq_data


@dataclass
class DataTransformConfig:
    transformed_train_file_path: str = os.path.join('artifacts', 'train_transformed.csv')
    transformed_test_file_path: str = os.path.join('artifacts', 'test_transformed.csv')
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns the scikit-learn Pipeline containing all 
        transformation steps in sequential order.
        """
        logging.info("Initializing the data transformation pipeline.")
        try:
            timeSeries_pipeline = Pipeline(
                steps = [
                    ("indexer", DatetimeIndexer()),
                    ("imputer", TimeSeriesImputer(method='time')),
                    ("resampler", TimeSeriesResampler(frequency='h'))
                    ]
            )
            logging.info('Time series transformation completed!')
            return timeSeries_pipeline
        except Exception as e:
            raise CustomException(e, sys)

