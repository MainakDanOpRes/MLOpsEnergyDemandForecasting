import sys
import os
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.utils import save_object

class DatetimeIndexer(BaseEstimator, TransformerMixin):
    """Converts Date and Time columns into a Datetime Index."""
    def __init__(self, date_col, time_col):
        self.date_col = date_col
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['Datetime'] = pd.to_datetime(X_transformed[self.date_col] + ' ' + \
                                                    X_transformed[self.time_col], format='%d/%m/%Y %H:%M:%S')
        X_transformed.drop([self.date_col, self.time_col], axis=1, inplace=True)
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
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.interpolate(method=self.method)
        X_transformed.bfill(inplace=True)

        return X_transformed

class TimeSeriesResampler(BaseEstimator, TransformerMixin):
    """Resamples the data to hourly averages."""
    def __init__(self, target_col, frequency = 'h'):
        self.target_col = target_col
        self.frequency = frequency
    
    def fit(self, X, y=None):
        # Imputers don't need to "learn" anything from the training data 
        # for interpolation, so we just return self.
        return self
    
    def transform(self, X):
        freq_data = pd.DataFrame()
        freq_data[self.target_col] = X[self.target_col].resample(self.frequency).mean()

        return freq_data


@dataclass
class DataTransformConfig:
    transformed_train_file_path: str = os.path.join('artifacts', 'train_transformed.csv')
    transformed_test_file_path: str = os.path.join('artifacts', 'test_transformed.csv')
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

    date_col: str = 'Date'
    time_col: str = 'Time'
    target_col: str = 'Global_active_power'

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
                    ("indexer", DatetimeIndexer(
                        date_col=self.data_transformation_config.date_col,
                        time_col=self.data_transformation_config.time_col
                    )),
                    ("imputer", TimeSeriesImputer(method='time')),
                    ("resampler", TimeSeriesResampler(
                        target_col=self.data_transformation_config.target_col,
                        frequency='h'))
                    ]
            )
            logging.info('Time series transformation completed!')
            return timeSeries_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path, low_memory=False)
            test_df = pd.read_csv(test_path, low_memory=False)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing pipeline on training dataframe and testing dataframe")
            train_processed = preprocessing_obj.fit_transform(train_df)
            test_processed = preprocessing_obj.fit_transform(test_df)

            train_processed.to_csv(self.data_transformation_config.transformed_train_file_path,
                                   index = True, header = True)
            test_processed.to_csv(self.data_transformation_config.transformed_test_file_path,
                                  index = True, header = True)
            
            logging.info('Saved transformed datasets to artifacts.')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info('Saved preprocessing pipeline object.')

            return (
                self.data_transformation_config.transformed_train_file_path,
                self.data_transformation_config.transformed_test_file_path,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     obj = DataTransformation()
#     obj.initiate_data_tranformation(train_path='artifacts/train.csv',
#                                     test_path='artifacts/test.csv')