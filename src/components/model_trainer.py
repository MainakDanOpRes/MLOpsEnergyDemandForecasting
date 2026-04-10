import os
import sys
import pandas as pd
import numpy as np
import itertools
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from prophet import Prophet
import xgboost as xgb
import pmdarima as pm

from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.utils import save_object # Assuming you have this from earlier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    target_col: str = 'Global_active_power'
    params: dict = None

    def __post__(self):
        self.params = {
            "XGBoost": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            "Prophet": {
                'changepoint_prior_scale': [0.01, 0.1, 0.5],
                'seasonality_prior_scale': [0.1, 1, 10]
            },
            "ARIMA": {}
        }

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.train_df = None
        self.test_df = None
        self.target = self.model_trainer_config.target_col

    def evaluate_metrics(self, model_name, actual, predicted):
        """Helper to calculate standard performance metrics"""
        mae = mean_absolute_error(y_true=actual, y_pred=predicted)
        rmse = np.sqrt(mean_squared_error(y_true=actual, y_pred=predicted))
        r2 = r2_score(y_true=actual, y_pred=predicted)
        
        logging.info(f"{model_name} Evaluation - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")
        
        return {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    
    # --- model specific data preparation and training ---

    def _train_prophet(self, model_class, param_grid):
        logging.info('Training and Tuning Prophet...')
        
        train_df_prophet = self.train_df.reset_index().rename(
            columns={'Datetime': 'ds', self.target: 'y'})
        
        all_params = [dict(zip(param_grid.keys(), v)) 
                      for v in itertools.product(*param_grid.values())]

        best_r2 = float('-inf')
        best_model = None
        best_metrics = None

        for params in all_params:
            model = model_class(**params)
            model.fit(train_df_prophet)

            future = model.make_future_dataframe(periods=len(self.test_df), freq='h')
            forecast = model.predict(future)
            forecast_prophet = forecast.set_index('ds')['yhat'][-len(self.test_df):]
            metrics = self.evaluate_metrics("Prophet", self.test_df[self.target], 
                                            forecast_prophet)
            
            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                best_model = model
                best_metrics = metrics
                
        return best_model, best_metrics

    def _train_xgboost(self, train, test):
        pass
    def _train_arima(self, train, test):
        pass

    def initiate_model_trainer(self, train_path, test_path):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)