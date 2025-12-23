import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.layers import Embedding, LSTM, add,  Reshape, concatenate
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *



logger = get_logger(__name__)

class CaptionModel:

    @staticmethod
    def Caption_Model(self, max_length, vocab_size):

        try:

            logger.info("Model architecture initialized")

            input1 = Input(shape = (1920, ))
            input2 = Input(shape = (max_length, ))

            img_features = Dense(256, activation = 'relu')(input1)
            img_features_reshaped = Reshape((1,256), input_shape = (256, ))(img_features)

            sentence_features = Embedding(vocab_size, 256, mask_zero = False)(input2)

            merged = concatenate([img_features_reshaped, sentence_features], axis = 1)
            sentence_features = LSTM(256)(merged)

            x = Dropout(0.5)(sentence_features)
            x = add([x, img_features])
            x = Dense(128, activation = 'relu')(x)
            x = Dropout(0.5)(x)
            output = Dense(vocab_size, activation = 'softmax')(x)

            caption_model = Model(inputs = [input1, input2], outputs = output)
            caption_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

            return caption_model
        
        except Exception as e:
            logger.error("Failed to initialize model architecture.")
            raise CustomException("Error while initializing model architecture.")
        