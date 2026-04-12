import os
import sys
import pandas as pd
from datetime import timedelta

from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, recent_data_path, steps_to_forecast=24):
        """
        Takes recent historical data, preprocesses it, and predicts 'n' steps into the future.
        """
        try:
            logging.info("Loading model and preprocessor artifacts.")

            preprocessor = load_object(file_path=self.preprocessor_path)
            model_artifact = load_object(file_path=self.model_path)

            self.model_name, self.model = model_artifact
            logging.info(f"Successfully loaded the {self.model_name} model.")

            logging.info("Preprocessing recent historical data.")
            raw_data = pd.read_csv(recent_data_path, sep=',',na_values=['?'],low_memory=False)
            
            logging.info("Applying the exact same transformations used during training.")
            # Apply the exact same transformations used during training
            processed_data = preprocessor.transform(raw_data)
            logging.info("Preprocessing completed.")
            # Infer frequency for generating future dates
            self.freq = pd.infer_freq(processed_data.index)
            if self.freq is None:
                self.freq = 'h' # Default to hourly if pandas can't infer it

            # 3. Route to the correct prediction logic
            predictions = None
            self.steps_to_forecast = steps_to_forecast

            if self.model_name == "Prophet":
                predictions = self._predict_prophet()
            
            elif self.model_name == "ARIMA":
                predictions = self._predict_arima(last_index=processed_data.index)
            
            elif self.model_name == "XGBoost":
                predictions = self._predict_xgboost(processed_data=processed_data)

            else:
                raise Exception(f"Unknown model type loaded: {self.model_name}")
            
            logging.info('Prediction completed successfully.')
            return predictions

        except Exception as e:
            raise CustomException(e, sys)
        

    def _predict_prophet(self):
        """Prophet generates its own future dataframe."""
        future = self.model.make_future_dataframe(periods=self.steps_to_forecast,
                                                  freq = self.freq )
        
        forecast = self.model.predict(future)
        future_forecast = forecast[['ds', 'yhat']].tail(self.steps_to_forecast)
        future_forecast.set_index('ds', inplace=True)
        future_forecast.rename(columns={'yhat': 'Forecast'}, inplace=True)
        return future_forecast

    def _predict_arima(self, last_index):
        """ARIMA simply predicts n_periods ahead."""
        forecast_values = self.model.predict(n_periods=self.steps_to_forecast)
        
        # Create a datetime index for the forecasted values
        last_date = last_index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=self.steps_to_forecast, 
                                     freq=self.freq)
        
        return pd.DataFrame({'Forecast': forecast_values}, index=future_dates)

    def _predict_xgboost(self, processed_data):
        """
        Generates recursive forecasts for XGBoost.
        Because XGBoost is a tabular model, to predict hour (t+1),
        it requires the prediction from hour (t) as a lag feature.
        """
        logging.info("Executing recursive forecasting for XGBoost.")
        target_col = 'Global_active_power'
        
        # 1. Validation: We must have at least 24 historical points to generate the initial 24 lags
        if len(processed_data) < 24:
            raise ValueError("XGBoost requires at least 24 hours of recent historical data to initialize its lag features.")
            
        # Extract the target sequence as a standard Python list so we can append to it
        history = processed_data[target_col].tolist()
        last_date = processed_data.index[-1]
        
        forecast_dates = []
        forecast_values = []
        
        # We use pd.to_timedelta to dynamically parse the inferred frequency
        # Note: If freq is 'h' or 'D', pandas handles this cleanly
        try:
            time_step = pd.to_timedelta(1, unit=self.freq)
        except:
            time_step = pd.Timedelta(hours=1) # Fallback to hourly
            
        current_date = last_date
        
        # 2. The Recursive Loop
        for _ in range(self.steps_to_forecast):
            # Advance the clock by one step
            current_date += time_step
            forecast_dates.append(current_date)
            
            # A. Build the time-based features for this specific future hour
            row_features = {
                'hour': current_date.hour,
                'day': current_date.day,
                'weekday': current_date.weekday(),
                'month': current_date.month
            }
            
            # Build the lag features dynamically from our history list
            # history[-1] is the most recent value (lag_1), history[-2] is lag_2, etc.
            for i in range(1, 25):
                row_features[f'lag_{i}'] = history[-i]
                
            # Convert the dictionary to a single-row DataFrame
            X_step = pd.DataFrame([row_features])
            
            # Predict the value for this single step
            # model.predict returns an array, so we slice [0] to get the float
            step_prediction = float(self.model.predict(X_step)[0])
            
            # The most important part: Append the prediction back into the history!
            # The next loop will now use this prediction as lag_1
            history.append(step_prediction)
            forecast_values.append(step_prediction)
            
        # 3. Format and return the final dataframe
        future_forecast = pd.DataFrame({'Forecast': forecast_values}, index=forecast_dates)
        
        return future_forecast
    

# if __name__ == "__main__":
#     pipeline = PredictPipeline()
#     forecast_df = pipeline.predict(
#     recent_data_path='artifacts/test.csv', # Or a newly uploaded CSV of last week's power usage
#     steps_to_forecast=24
# )

# print(forecast_df)