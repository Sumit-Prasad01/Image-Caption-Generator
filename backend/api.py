from fastapi import FastAPI, UploadFile, File
import numpy as np
import pickle
import uvicorn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import shutil
import os
from config.paths_config import *


app = FastAPI(title="Image Captioning API")

# Load models once (IMPORTANT for performance)
MODEL_PATH = SAVED_MODEL_PATH
TOKENIZER_PATH = TOKENIZER_PATH
FEATURE_EXTRACTOR_PATH = FEATURE_EXTRACTED_PATH
MAX_LENGTH = 34
IMG_SIZE = 224

caption_model = load_model(MODEL_PATH)
feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


def generate_caption(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    image_features = feature_extractor.predict(img, verbose=0)

    in_text = "startseq"
    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)

        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat_index)
        if word is None:
            break

        in_text += " " + word
        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()


@app.post("/generate-caption")
async def caption_image(file: UploadFile = File(...)):
    image_path = f"temp_{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    caption = generate_caption(image_path)
    os.remove(image_path)

    return {
        "filename": file.filename,
        "caption": caption
    }



if __name__ == "__main__":
    uvicorn.run("api:app", host = "127.0.0.1", port = 8000, reload = True)
