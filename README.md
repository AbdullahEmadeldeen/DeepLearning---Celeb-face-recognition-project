# 🎭 Celebrity Face Recognition

A powerful full-stack application that detects and recognizes celebrity faces using deep learning. This project combines **FastAPI** for the backend with a **FaceNet + ArcFace** model for high-accuracy face recognition, and a modern **HTML/JS** frontend for user interaction.

## ✨ Features

-   **Face Detection**: Uses MTCNN to detect faces in uploaded images.
-   **Face Recognition**: Identifies celebrities from a supported dataset of 105 classes.
-   **Confidence Scoring**: Displays recognition confidence and filters low-confidence predictions (< 85%).
-   **Top 3 Predictions**: Shows the top 3 most likely celebrity matches.
-   **Real-time API**: Fast and efficient REST API built with FastAPI.
-   **Interactive UI**: User-friendly interface to upload images, view results, and search available celebrities.

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI, Uvicorn
-   **Deep Learning**: PyTorch, FaceNet (InceptionResnetV1), ArcFace Loss
-   **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
-   **Image Processing**: Pillow (PIL), NumPy

## 📂 Project Structure

```
.
├── fastapi_1.py          # Main FastAPI backend application
├── improved_gui.html     # Main frontend interface
├── req.txt               # Python dependencies
├── script.js             # Frontend logic (if separate)
├── style.css             # Frontend styling (if separate)
└── ...
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies
Ensure you have Python installed (3.8+ recommended). Install the required packages:
```bash
pip install -r req.txt
```

### 3. Model & Dataset Setup
This application requires a pre-trained model and a dataset directory structure.
-   **Model**: Ensure `best_facenet_arcface.pth` is located at `D:\3rd first term\Deep Learning\` (or update `MODEL_PATH` in `fastapi_1.py`).
-   **Dataset**: Ensure the dataset is at `D:\3rd first term\Deep Learning\datasets\105_classes_pins_dataset` (or update `DATASET_PATH` in `fastapi_1.py`).

## 🏃‍♂️ Usage

### Start the Backend Server
Run the FastAPI server using Uvicorn:
```bash
python fastapi_1.py
# OR
uvicorn fastapi_1:app --reload --host 0.0.0.0 --port 8000
```
The API will start at `http://0.0.0.0:8000`.

### Launch the Frontend
Simply open `improved_gui.html` in your web browser.
-   Ensure the backend server is running first.
-   The UI will automatically check the API health status.

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root endpoint with API info. |
| `GET` | `/health` | Checks API health and model status. |
| `GET` | `/celebrities` | Returns a list of all supported celebrities. |
| `POST` | `/predict` | Upload an image to detect and recognize faces. |

## 🧠 Model Details

The system uses **FaceNet (InceptionResnetV1)** as the backbone feature extractor, trained with **ArcFace (Additive Angular Margin Loss)** to maximize class separability.
-   **Input Size**: 160x160 pixels
-   **Embedding Size**: 512 dimensions
-   **Classes**: 105 Celebrities

## DATA LOCATION 
https://drive.google.com/drive/folders/1i7PZRyXUerebsoi5IK6sA_9M4deiu8tV?usp=drive_link



