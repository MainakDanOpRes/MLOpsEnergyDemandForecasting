import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.utils.exception import CustomException
from src.utils.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def run_training_pipeline(self):
        """
        Executes the entire machine learning lifecycle sequentially.
        """
        try:
            logging.info("========== Starting the Training Pipeline ==========")

            # Data Ingestion
            logging.info("Initiating Data Ingestion")
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()

            # Data Transformation
            logging.info("Initiating Data Transformation")
            transformation = DataTransformation()
            train_arr_path, test_arr_path, _ = transformation.initiate_data_transformation(
                train_path=train_data_path, 
                test_path=test_data_path
            )

            # Model Training
            logging.info("Initiating Model Training")
            trainer = ModelTrainer()
            best_model_name, best_metrics = trainer.initiate_model_trainer(
                train_path=train_arr_path,
                test_path=test_arr_path
            )

            logging.info("========== Training Pipeline Completed ==========")
            print(f"\n✅ Pipeline Success! Best Model: {best_model_name}")
            print(f"📊 Metrics: {best_metrics}\n")
            
            return best_model_name, best_metrics

        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
    # pipeline = TrainPipeline()
    # pipeline.run_training_pipeline()