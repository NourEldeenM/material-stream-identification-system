import pickle
import time
from typing import Annotated
import cv2
from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
from feature_extraction import FeatureExtractor

feature_extractor = FeatureExtractor()
svm_model = None
knn_model = None
svm_classes = None
knn_classes = None

origins = [
  "http://localhost:5173"
]

app = FastAPI(
  title="Models Backend"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

@app.on_event("startup")
async def load_models():
  global svm_model, knn_model, svm_classes, knn_classes
  with open('models/svm_final.pkl', 'rb') as file:
    model_dict = pickle.load(file)
    svm_model = model_dict["svm"]
    svm_classes = model_dict["classes"]
    
  with open('models/knn_final.pkl', 'rb') as file:
    model_dict = pickle.load(file)
    knn_model = model_dict["knn"]
    knn_classes = model_dict["classes"]

@app.get("/")
def root():
  """Health check endpoint"""
  return {
    "status": "running",
    "service": "Models Provider Backend"
  }
  
@app.post("/api/analyze/svm")
def get_svm_model_prediction(image: Annotated[str, Form()]):
  try:
    startTime = time.time()
    image = image.split(",")[1]
    decoded_bytes = base64.b64decode(image)
    np_arr = np.frombuffer(decoded_bytes, np.uint8)
    decoded_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
  
    extracted_features = feature_extractor.extract_features(decoded_image)
    prediction = svm_model.predict(extracted_features.reshape(1, -1))
    
    request_time = time.time() - startTime
    print(f"Request time: {request_time:.3f}s")
    return {"prediction": svm_classes[prediction[0].item()]}
  except Exception as e:
    print(f"An error occurred: {e}")
    return {"prediction": f"{e}"}

@app.post("/api/analyze/knn")
def get_knn_model_prediction(image: Annotated[str, Form()]):
  try:
    startTime = time.time()
    image = image.split(",")[1]
    decoded_bytes = base64.b64decode(image)
    np_arr = np.frombuffer(decoded_bytes, np.uint8)
    decoded_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
  
    extracted_features = feature_extractor.extract_features(decoded_image)
    prediction = knn_model.predict(extracted_features.reshape(1, -1))
    
    request_time = time.time() - startTime
    print(f"Request time: {request_time:.3f}s")    
    return {"prediction": knn_classes[prediction[0].item()]}
  except Exception as e:
    print(f"An error occurred: {e}")
    return {"prediction": f"{e}"}
