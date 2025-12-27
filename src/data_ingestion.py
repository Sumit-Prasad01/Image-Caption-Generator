import os
import shutil
import zipfile
import kagglehub
from src.logger import get_logger
from src.custom_exception import CustomException
from config.data_ingestion_config import DATASET_NAME, TARGET_DIR

logger = get_logger(__name__)


class DataIngestion:
    """
    Downloads ANY Kaggle dataset and stores raw files in artifacts/raw
    """

    def __init__(self, dataset_name: str, target_dir: str):
        self.dataset_name = dataset_name
        self.target_dir = target_dir

    def create_raw_dir(self) -> str:
        raw_dir = os.path.join(self.target_dir, "raw")
        try:
            os.makedirs(raw_dir, exist_ok=True)
            logger.info(f"Raw directory ready at: {raw_dir}")
            return raw_dir
        except Exception as e:
            raise CustomException(f"Failed to create raw directory:",e)

    def _copy_directory(self, source_dir: str, raw_dir: str):
        """
        Copies entire dataset directory contents to artifacts/raw
        """
        for item in os.listdir(source_dir):
            src_path = os.path.join(source_dir, item)
            dst_path = os.path.join(raw_dir, item)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)

            logger.info(f"Copied: {src_path} → {dst_path}")

    def _extract_zip(self, zip_path: str, raw_dir: str):
        """
        Extracts ZIP dataset into artifacts/raw
        """
        logger.info("ZIP dataset detected — extracting all contents...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        logger.info("ZIP extraction completed.")

    def download_dataset(self, raw_dir: str):
        """
        Downloads dataset using KaggleHub and saves everything to artifacts/raw
        """
        try:
            logger.info(f"Downloading Kaggle dataset: {self.dataset_name}")
            dataset_path = kagglehub.dataset_download(self.dataset_name)

            if not dataset_path or not os.path.exists(dataset_path):
                raise CustomException("KaggleHub download failed")

            logger.info(f"Dataset downloaded to: {dataset_path}")

            # Case 1: ZIP file
            if dataset_path.endswith(".zip"):
                self._extract_zip(dataset_path, raw_dir)

            # Case 2: Directory
            elif os.path.isdir(dataset_path):
                self._copy_directory(dataset_path, raw_dir)

            else:
                raise CustomException("Unknown dataset format returned by KaggleHub",e)

        except Exception as e:
            logger.error(f"Dataset download failed: {e}")
            raise CustomException(f"Dataset download failed: ",e)

    def run(self):
        try:
            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)
            logger.info("Data ingestion completed successfully")
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise CustomException(f"Data ingestion failed: ",e)


if __name__ == "__main__":
    try:
        ingestion = DataIngestion(DATASET_NAME, TARGET_DIR)
        ingestion.run()
    except Exception as e:
        logger.error(f" Data Ingestion Pipeline crashed: {e}")
