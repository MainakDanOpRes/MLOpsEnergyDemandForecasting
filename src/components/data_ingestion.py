import sys
import os
from src.utils.exception import CustomException
from src.utils.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/dataset/household_power_consumption.txt', sep=';', na_values=['?'], low_memory=False)
            logging.info('Read the dataset as dataframe')
            
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
            df.drop(['Date', 'Time'], axis=1, inplace=True)
            df.set_index('Datetime', inplace=True)
            logging.info('Setting date and time as index')
            
            df = df.interpolate(method='time')
            df.bfill(inplace=True) 
            logging.info('Managing the missing value with time based interpolation')

            hourly_data = pd.DataFrame()
            hourly_data['Global_active_power'] = df['Global_active_power'].resample('h').mean()
            logging.info('Resampling only global active power every hour')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            hourly_data.to_csv(self.ingestion_config.raw_data_path, index=True, header=True)
            logging.info('Raw data imported!')
            
            # initiating train and test data split. for timeseries datam shuffle must be set to False
            logging.info('Train test split inititated')
            train_set, test_set = train_test_split(hourly_data, test_size=0.2, shuffle=False)
            train_set.to_csv(self.ingestion_config.train_data_path, index=True, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=True, header=True)
            logging.info('Data ingestion completed!')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
# if __name__ == "__main__":
#     obj = DataIngestion()
#     obj.initiate_data_ingestion()