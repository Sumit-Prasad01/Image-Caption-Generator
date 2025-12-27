import numpy as np
import pandas as pd
import joblib
from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import *
from src.model_architecture import CaptionModel


logger = get_logger(__name__)

class Loader:

    @staticmethod
    def load_data(self, captions_path):
        
        try:
            logger.info("Loading captions data")
            data = pd.read_csv(captions_path)

            logger.info("Data loaded successfully.")

            return data
        
        except Exception as e:
            logger.error("Failed to load captions data.")
            raise CustomException("Error while loading captions data", e)
    
    @staticmethod
    def load_model(self):

        try:

            vocab_size = joblib.load(VOCAB_SIZE_PATH)
            max_length = joblib.load(MAX_LENGTH_PATH)

            model = CaptionModel(max_length, vocab_size)

            logger.info("Model loaded successfully")

            return model
        
        except Exception as e:
            logger.info("Failed to load model.")
            raise CustomException("Error while loading model", e)