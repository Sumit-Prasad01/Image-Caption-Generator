import os

CAPTIONS_PATH : str = "artifacts/raw/captions.txt"
IMAGE_PATH : str = "artifacts/raw/Images"

PROCESSED_DATA_PATH = "artifacts/processed/"

# Features Path
## Save 
TRAIN_PATH = os.path.join(PROCESSED_DATA_PATH, "X_train.pkl")
TEST_PATH = os.path.join(PROCESSED_DATA_PATH, "X_test.pkl")

# load path
TRAIN_LOAD_PATH = 'artifacts/processed/X_train.pkl'
TEST_LOAD_PATH = 'artifacts/processed/X_test.pkl'


# Model Training
MODEL_PATH = "artifacts/models"
os.makedirs(MODEL_PATH, exist_ok=True)
SAVE_MODEL_PATH = os.path.join(MODEL_PATH, "lstm_model.keras")
SAVED_MODEL_PATH = "artifacts/models/lstm_model.keras"
FEATURE_EXTRACTED_PATH = "artifacts/models/feature_extractor.keras"
TOKENIZER_PATH = "artifacts/models/tokenizer.pkl"

