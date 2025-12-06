"""
Celebrity Face Recognition - FastAPI Application
=================================================

API endpoints for celebrity face recognition using FaceNet + ArcFace
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
from facenet_pytorch import InceptionResnetV1, MTCNN
import os
import io
import base64

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = r"D:\3rd first term\Deep Learning\best_facenet_arcface.pth"
DATASET_PATH = r"D:\3rd first term\Deep Learning\datasets\105_classes_pins_dataset"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 160
EMBEDDING_SIZE = 512
NUM_CLASSES = 105

# Celebrity class names (define them directly or load from a separate file)
# Option 1: Define the list directly
class_names_sorted = sorted(os.listdir(DATASET_PATH))
CLASS_NAMES = [name for name in class_names_sorted if os.path.isdir(os.path.join(DATASET_PATH, name))]



class PredictionResult(BaseModel):
    celebrity_name: str
    confidence: float

class FaceDetection(BaseModel):
    face_index: int
    bounding_box: List[float]
    detection_confidence: float
    prediction: PredictionResult
    top_3_predictions: List[PredictionResult]

class RecognitionResponse(BaseModel):
    success: bool
    num_faces_detected: int
    faces: List[FaceDetection]
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    device: str
    num_classes: int
    model_loaded: bool

class CelebrityListResponse(BaseModel):
    total_celebrities: int
    celebrities: List[str]

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ArcFaceMarginProduct(nn.Module):
    """ArcFace margin product for angular margin loss"""
    def __init__(self, embedding_size, n_classes, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, embeddings, labels=None):
        W = F.normalize(self.weight, dim=1)
        cos_theta = torch.mm(embeddings, W.t()).clamp(-1.0, 1.0)

        if labels is None:
            return cos_theta * self.s

        cos_y = cos_theta.gather(1, labels.view(-1, 1)).view(-1)
        sin_theta = torch.sqrt(1.0 - cos_y * cos_y).clamp(0.0, 1.0)
        phi = cos_y * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            phi = torch.where(cos_y > 0, phi, cos_y)
        else:
            phi = torch.where(cos_y > self.th, phi, cos_y - self.mm)

        logits = cos_theta.clone()
        logits.scatter_(1, labels.view(-1, 1), phi.view(-1, 1))
        logits = logits * self.s
        return logits


class FaceNetArcFace(nn.Module):
    """FaceNet (InceptionResnetV1) with ArcFace head"""
    def __init__(self, n_classes, embedding_size=512, pretrained='vggface2', s=30.0, m=0.5):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)
        self.margin_product = ArcFaceMarginProduct(
            embedding_size=embedding_size, 
            n_classes=n_classes, 
            s=s, 
            m=m
        )

    def forward(self, x, labels=None):
        embeddings = self.backbone(x)
        embeddings = F.normalize(embeddings, dim=1)
        logits = self.margin_product(embeddings, labels)
        return logits, embeddings


# ============================================================================
# MODEL & DETECTOR INITIALIZATION
# ============================================================================

model = None
mtcnn = None

def load_model():
    """Load the trained model"""
    global model
    model = FaceNetArcFace(
        n_classes=NUM_CLASSES,
        embedding_size=EMBEDDING_SIZE,
        pretrained=None,
        s=30.0,
        m=0.5
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("✅ Model loaded successfully!")
    return model


def load_face_detector():
    """Load MTCNN face detector"""
    global mtcnn
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=20,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=DEVICE,
        keep_all=True
    )
    print("✅ Face detector loaded successfully!")
    return mtcnn


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_celebrity(face_tensor):
    """Predict celebrity from face tensor"""
    with torch.no_grad():
        logits, embeddings = model(face_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        
    pred_class = CLASS_NAMES[pred_idx.item()]
    confidence_score = confidence.item()
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
    top3_predictions = [
        PredictionResult(
            celebrity_name=CLASS_NAMES[idx.item()],
            confidence=float(prob.item())
        )
        for idx, prob in zip(top3_indices[0], top3_probs[0])
    ]
    
    return pred_class, confidence_score, top3_predictions


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Celebrity Face Recognition API",
    description="API for detecting and recognizing celebrity faces using FaceNet + ArcFace",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model and detector on startup"""
    print("🚀 Starting Celebrity Face Recognition API...")
    load_model()
    load_face_detector()
    print(f"📊 Device: {DEVICE}")
    print(f"👥 Number of celebrities: {NUM_CLASSES}")
    print("✅ API ready to accept requests!")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Celebrity Face Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "celebrities": "/celebrities"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        device=str(DEVICE),
        num_classes=NUM_CLASSES,
        model_loaded=model is not None
    )


@app.get("/celebrities", response_model=CelebrityListResponse)
async def get_celebrities():
    """Get list of all supported celebrities"""
    return CelebrityListResponse(
        total_celebrities=len(CLASS_NAMES),
        celebrities=CLASS_NAMES
    )


@app.post("/predict", response_model=RecognitionResponse)
async def predict_face(file: UploadFile = File(...)):
    """
    Predict celebrity from uploaded image
    
    Parameters:
    - file: Image file (jpg, jpeg, png, webp)
    
    Returns:
    - Recognition results with bounding boxes and predictions for all detected faces
    """
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG, PNG, or WebP image."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Detect faces
        boxes, probs = mtcnn.detect(image)
        faces = mtcnn(image)
        
        # Check if faces were detected
        if faces is None or boxes is None:
            return RecognitionResponse(
                success=False,
                num_faces_detected=0,
                faces=[],
                message="No face detected in the image. Please upload a clearer image with a visible face."
            )
        
        # Process each detected face
        face_detections = []
        for idx, (face_tensor, box, prob) in enumerate(zip(faces, boxes, probs)):
            # Predict celebrity
            face_input = face_tensor.unsqueeze(0).to(DEVICE)
            pred_name, confidence, top3 = predict_celebrity(face_input)
            
            # Create face detection result
            face_detection = FaceDetection(
                face_index=idx,
                bounding_box=box.tolist(),
                detection_confidence=float(prob),
                prediction=PredictionResult(
                    celebrity_name=pred_name,
                    confidence=confidence
                ),
                top_3_predictions=top3
            )
            face_detections.append(face_detection)
        
        return RecognitionResponse(
            success=True,
            num_faces_detected=len(boxes),
            faces=face_detections,
            message=f"Successfully detected and recognized {len(boxes)} face(s)"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__== "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )