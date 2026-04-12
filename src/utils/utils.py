import os
import sys
import joblib
from src.utils.exception import CustomException
from src.utils.logger import logging

def save_object(file_path, obj):
    """Save an artifact to the file path."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Loads a saved artifact from the file path."""
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)