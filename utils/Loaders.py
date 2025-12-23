import numpy as np
import pandas as pd
from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import *

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
    
