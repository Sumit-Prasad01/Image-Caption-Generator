import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.layers import Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from textwrap import wrap
import pickle
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from src.model_architecture import CaptionModel 



logger = get_logger(__name__)

class ModelTraining:

    def __init__(self):
        self.history = None
        self.model_name = "../artifacts/models/lstm_model.keras"
        self.checkpoint = None
        self.earlystopping = None
        self.learning_rate_reduction = None
        self.model = None

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
        
    
    def load_and_train_model(self):
        try:

            self.model = CaptionModel()

            self.history = caption_model.fit(
                train_generator,
                epochs = 2,
                validation_data = validation_generator,
                callbacks = [checkpoint, earlystopping, learning_rate_reduction]
            )


        