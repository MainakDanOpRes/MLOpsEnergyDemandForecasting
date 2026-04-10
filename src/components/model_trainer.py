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

        dataset_freq = pd.infer_freq(self.train_df.index)

        if dataset_freq is None:
            logging.warning("Could not infer frequency from index. Defaulting to 'h'.")
            dataset_freq = 'h'
        else:
            logging.info(f"Dynamically inferred dataset frequency: {dataset_freq}")
        
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

            future = model.make_future_dataframe(periods=len(self.test_df), freq=dataset_freq)
            forecast = model.predict(future)
            forecast_prophet = forecast.set_index('ds')['yhat'][-len(self.test_df):]
            metrics = self.evaluate_metrics("Prophet", self.test_df[self.target], 
                                            forecast_prophet)
            
            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                best_model = model
                best_metrics = metrics
        logging.info('Prophet training and tuning completed!')
        return best_model, best_metrics

    def _train_xgboost(self, base_model, param_grid):
        logging.info('Training and tuning XGBoost...')

        def create_xgb_features(df):
            df_xgb = df.copy()
            df_xgb['hour'] = df_xgb.index.hour
            df_xgb['day'] = df_xgb.index.day
            df_xgb['weekday'] = df_xgb.index.weekday
            df_xgb['month'] = df_xgb.index.month

            for i in range(24):
                df_xgb[f'lag_{i+1}'] = df_xgb[self.target].shift(i+1)
            df_xgb.dropna(inplace=True)
            return df_xgb

        full_data = pd.concat([self.train_df, self.test_df])
        xgb_data = create_xgb_features(full_data)

        train_len = len(self.train_df) - 24
        X = xgb_data.drop(self.target, axis=1)
        y = xgb_data[self.target]

        X_train, y_train = X.iloc[:train_len], y.iloc[:train_len]
        X_test, y_test = X.iloc[train_len:], y.iloc[train_len:]

        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        metrics = self.evaluate_metrics("XGBoost", y_test, y_pred)
        logging.info('XGBoost training and tuning completed!')

        return best_model, metrics


    def _train_arima(self, model_func, param_grid):
        logging.info('Training Auto-ARIMA')
        y_train = self.train_df[self.target]
        y_test = self.test_df[self.target]

        model = model_func(y_train, start_p=0, start_q=0, 
                           max_p=5, max_q=5, m=1, 
                           trace=False, error_action='ignore', 
                           suppress_warnings=True, 
                           stepwise=True)
        
        forecast = model.predict(n_periods=len(y_test))
        metrics = self.evaluate_metrics("ARIMA", y_test, forecast)

        logging.info('Training of Auto-ARIMA completed!')

        return model, metrics

    def initiate_model_trainer(self, train_path, test_path):
        try:
            logging.info('loading transformed data into class state.')

            self.train_df = pd.read_csv(train_path, index_col = 'Datetime', parse_dates=True)
            self.test_df = pd.read_csv(test_path, index_col = 'Datetime', parse_dates=True)
            
            models = {
                "Prophet": Prophet,
                "XGBoost": xgb.XGBRegressor(),
                "ARIMA": pm.auto_arima
            }

            model_report = {}

            for model_name, model_obj in models.items():
                logging.info(f'Initiating training for {model_name}')
                model_params = self.model_trainer_config.params[model_name]

                if model_name == "Prophet":
                    trained_model, metrics = self._train_prophet(model_obj, model_params)
                elif model_name == "XGBoost":
                    trained_model, metrics = self._train_xgboost(model_obj, model_params)
                elif model_name == "ARIMA":
                    trained_model, metrics = self._train_arima(model_obj, model_params)
                else:
                    logging.warning(f"No specific training logic defined for {model_name}")
                    continue
                model_report[model_name] = {
                    "model": trained_model, 
                    "metrics": metrics
                }
            
            best_model_name = max(model_report,
                                  key = lambda k: model_report[k]["metrics"]["R2"])
            best_model = model_report[best_model_name]["model"]
            best_metrics = model_report[best_model_name]["metrics"]

            logging.info(f"BEST MODEL FOUND: {best_model_name}")
            logging.info(f"Metrics: {best_metrics}")

            if best_metrics["R2"] < 0:
                logging.warning("No model performed better than a " \
                "baseline mean prediction (R2 < 0).")

            logging.info('Saving the best model object.')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = (best_model_name, best_model)
            )
            
            return best_model_name, best_metrics

        except Exception as e:
            raise CustomException(e, sys)