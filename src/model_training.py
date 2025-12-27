import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.Loaders import Loader



logger = get_logger(__name__)

class ModelTraining:

    def __init__(self,model_name ,train_generator_path, validation_generator_path):
        self.history = None
        self.model_name = model_name
        self.checkpoint = None
        self.earlystopping = None
        self.learning_rate_reduction = None
        self.model = None
        self.train_generator = train_generator_path
        self.validation_generator = validation_generator_path

        logger.info("Model training pipeline initiated.")


    def initialize_model_checkpoint(self):
        try:
            self.checkpoint = ModelCheckpoint(
                self.model_name, 
                monitor = 'val_loss',
                mode = 'min',
                save_best_only = True,
                save_weights_only = False,
                verbose = 1
            )

            self.earlystopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
            self.learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_loss', patience = 3, factor = 0.2, min_lr = 1e-8)

            logger.info("Model checkpoint initialized successfully.")
        
        except Exception as e:
            logger.error("Failed to initialize model checpoint.")
            raise CustomException("Error while initializing model checkpoint.")
        

    
    def train_model(self):
        try:

            self.model = Loader.load_model()

            self.history = self.model.fit(
                self.train_generator,
                epochs = 2,
                validation_data = self.validation_generator,
                callbacks = [self.checkpoint, self.earlystopping, self.learning_rate_reduction]
            )

            logger.info("Model training completed successfully.")

        except Exception as e:
            logger.error("Falied to train model.")
            raise CustomException("Error while training model.", e)
        
    
    def run(self):
        
        try:

            logger.info("Starting model training pipeline.")

            self.initialize_model_checkpoint()
            self.train_model()

        except Exception as e:
            logger.error("Failed to train model.")
            raise CustomException("Error while training model", e)


if __name__ == "__main__":

    trainer = ModelTraining(SAVED_MODEL_PATH, TRAIN_GENERTOR_PATH, VALIDATION_GENERTOR_PATH)
    trainer.run()