from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from config.data_ingestion_config import *
from config.paths_config import *

logger = get_logger(__name__)

class TrainingPipeline:

    def train(self):
        try:

            ingestion = DataIngestion(DATASET_NAME, TARGET_DIR)
            ingestion.run()

            processor = DataProcessing(IMAGE_PATH, CAPTIONS_PATH)
            processor.run()

            trainer = ModelTraining(SAVED_MODEL_PATH, TRAIN_GENERTOR_PATH, VALIDATION_GENERTOR_PATH)
            trainer.run()

        except Exception as e:
            logger.error(f"Pipeline crashed: {e}")


if __name__ == "__main__":

    training_pipeline = TrainingPipeline()
    training_pipeline.train()