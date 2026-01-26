
# 🖼️ Image Caption Generator  
**DenseNet201 + LSTM | FastAPI | Streamlit**

This project is a **production-ready Image Caption Generator** that automatically generates natural language descriptions for images using **Deep Learning**.  
It follows a **CNN–RNN architecture**, where **DenseNet201** extracts visual features from images and an **LSTM-based RNN** generates captions word by word.  
The trained model is exposed via a **FastAPI backend** and consumed through an interactive **Streamlit frontend**.

---

## 🚀 Key Highlights

- CNN–RNN based Image Captioning system
- DenseNet201 (pretrained) for robust image feature extraction
- LSTM RNN for sequential caption generation
- Complete training pipeline with configuration management
- RESTful inference API using FastAPI
- Interactive web interface built with Streamlit
- Modular, scalable, and industry-style project structure
- Model artifacts and checkpoints stored for reuse

---

## 🧠 Tech Stack

- **Python**
- **Deep Learning**: DenseNet201, LSTM (RNN)
- **Frameworks**: PyTorch / TensorFlow
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Utilities**: NumPy, Pandas

---

## 📁 Project Structure

```
.
├── artifacts/
│   └── models/              # Saved trained models & checkpoints
├── backend/                 # FastAPI backend for inference
├── frontend/                # Streamlit web application
├── config/                  # Model & training configurations
├── notebooks/               # Experiments and training setup
├── pipeline/                # End-to-end training pipeline
├── src/                     # Model architecture and core logic
├── utils/                   # Helper utilities
├── main.py                  # Training pipeline entry point
├── requirements.txt         # Project dependencies
├── setup.py                 # Package setup
├── pyproject.toml           # Build configuration
├── uv.lock                  # Dependency lock file
└── README.md                # Documentation
```

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone <https://github.com/Sumit-Prasad01/Image-Caption-Generator.git>
cd image-caption-generator
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate    # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Train the Model
```bash
python main.py
```

- Extracts image features using DenseNet201  
- Trains the LSTM decoder on image–caption pairs  
- Saves trained models in `artifacts/models/`

---

### 🔹 Run FastAPI Backend
```bash
uvicorn backend.app:app --reload
```

**API Endpoint**
```
POST /predict-caption
```

Input: Image file  
Output: Generated caption (JSON)

---

### 🔹 Run Streamlit Frontend
```bash
streamlit run frontend/app.py
```

- Upload an image
- Generate captions in real time using the trained model

---

## 🏗️ Model Architecture

- **Encoder (CNN)**: DenseNet201 (pretrained)
- **Feature Vector**: Extracted from global average pooling layer
- **Decoder (RNN)**: LSTM for word sequence prediction
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam

---

## 📊 Results

- Generates grammatically correct and context-aware captions
- Learns semantic relationships between visual objects and language
- Caption quality improves with training epochs and dataset size

---

## 🔮 Future Improvements

- Attention-based image captioning
- Transformer-based decoder
- BLEU, METEOR, CIDEr evaluation metrics
- Multi-language caption generation
- Dockerized deployment and CI/CD

---


