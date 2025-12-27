import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tqdm
import joblib
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
import seaborn as sns
from textwrap import wrap
import re
import pickle
from utils.Loaders import Loader
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from src.CustomDataGeneration import CustomDataGenerator


logger = get_logger(__name__)


class DataProcessing:

    def __init__(self, image_path : str, captions_path : str):

        self.image_path = image_path
        self.captions_path = captions_path
        self.data : pd.DataFrame = None
        self.captions = None
        self.train = None
        self.test = None
        self.vocab_size = None
        self.max_length = None
        self.fe = None
        self.tokenizer = None
        self.features = {}
        self.train_generator = None
        self.validation_generator = None


        logger.info("Data processing started.")


    def load_captions(self):
        try:
            logger.info("Loading captions data.")

            self.data = Loader.load_data(self.captions_path)

            logger.info("captions  data loaded successfully.")

        except Exception as e:
            logger.error("Failed to load captions data")
            raise CustomException("Error while loading captions data.", e)

    
    def readImage(self,img_size=224):
            
        try:
            logger.info("Reading images.")

            img = load_img(self.image_path,color_mode='rgb',target_size=(img_size,img_size))
            img = img_to_array(img)
            img = img/255.

            logger.info("Reading images completed.")
            
            return img
        
        except Exception as e:
            logger.error("Failed to read images.")
            raise CustomException("Error while reading images.", e)
        
    
    
    def text_preprocessing(self):

        try:
            logger.info("Captions text processing started")

            self.data['caption'] = self.data['caption'].apply(lambda x: x.lower())
            self.data['caption'] = self.data['caption'].apply(lambda x: x.replace("[^A-Za-z]",""))
            self.data['caption'] = self.data['caption'].apply(lambda x: re.sub(r"\s+", " ", x))
            self.data['caption'] = self.data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word)>1]))
            self.data['caption'] = "startseq "+self.data['caption']+" endseq"

            logger.info("Captions text completed successfully.")
        
        except Exception as e:
            logger.error("Failed to process captions text data.")
            raise CustomException("Error while processing captions text data.")

        

    def tokenize_and_save_and_split_data(self):
        try:
            logger.info("Starting tokenization and data splitting.")

            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(self.captions)
            self.vocab_size = len(self.tokenizer.word_index) + 1
            self.max_length = max(len(caption.split()) for caption in self.captions)

            images = self.data['image'].unique().tolist()
            nimages = len(images)

            split_index = round(0.85*nimages)
            train_images = images[:split_index]
            val_images = images[split_index:]

            self.train = self.data[self.data['image'].isin(train_images)]
            self.test = self.data[self.data['image'].isin(val_images)]

            self.train.reset_index(inplace=True,drop=True)
            self.test.reset_index(inplace=True,drop=True)

            with open(TOKENIZER_PATH, "wb") as f:
                pickle.dump(self.tokenizer, f)

        except Exception as e:
            logger.error("Failed to tokenize and split data.")
            raise CustomException("Error while tokenizing and splitting data.", e)
        

    def extract_and_save_image_features(self):

        try:
            logger.info("Extracting image features and saving it.")

            model = DenseNet201()
            self.fe = Model(inputs = model.input, outputs = model.layers[-2].output)
            img_size = 224

            self.features = {}

            for image in tqdm(self.data['image'].unique().tolist()):
                img = load_img(os.path.join(self.image_path, image), target_size = (img_size, img_size))
                img = img_to_array(img)
                img = img/255.
                img = np.expand_dims(img, axis = 0)
                feature = self.fe.predict(img, verbose = 0)
                self.features[image] = feature
            
            self.fe.save(FEATURE_EXTRACTED_PATH)

            logger.info(f"All features extracted and saved successfully and saved to {FEATURE_EXTRACTED_PATH}.")
        
        except Exception as e:
            logger.error("Failed to extract and save image features.")
            raise CustomException("Error while extracting and saving image features.", e)
        
   
    def generate_custom_data(self):

        try:
            self.train_generator = CustomDataGenerator(
                                        df = self.train,
                                        X_col = 'image',
                                        y_col = 'caption',
                                        batch_size = 64,
                                        directory = self.image_path,
                                        tokenizer = self.tokenizer,
                                        vocab_size = self.vocab_size,
                                        max_length = self.max_length,
                                        features = self.features
                                        )

            self.validation_generator = CustomDataGenerator(
                                        df = self.test,
                                        X_col = 'image',
                                        y_col = 'caption',
                                        batch_size = 64, 
                                        directory = self.image_path,
                                        tokenizer = self.tokenizer,
                                        vocab_size = self.vocab_size,
                                        max_length = self.max_length,
                                        features = self.features
                                        )
            
            joblib.dump(self.train_generator, TRAIN_LOAD_PATH)
            joblib.dump(self.validation_generator, VALIDATION_GENERTOR_PATH)
     
        except Exception as e:
            logger.error("Failed to generate custom data.")
            raise CustomException("Error while generating custom data.", e)
        
    def run(self):

        try:
            logger.info("Executing the data processing pipeline.")
            self.load_captions()
            self.readImage()
            self.text_preprocessing()
            self.tokenize_and_save_and_split_data()
            self.extract_and_save_image_features()
            self.generate_custom_data()

            # save vocab_size and max_len
            joblib.dump(self.vocab_size, VOCAB_SIZE_PATH)
            joblib.dump(self.max_length, MAX_LENGTH_PATH)

            logger.info("Data processing pipeline executed successfully.")

        except Exception as e:
            logger.error("Faled to execute data processing pipeline.")
            raise CustomException("Error while executing data processing pipeline.", e)


        

if __name__ == "__main__":

    processor = DataProcessing(IMAGE_PATH, CAPTIONS_PATH)
    processor.run()