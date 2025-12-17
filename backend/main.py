from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
  title="Models Backend"
)

origins = [
  "http://localhost:5173"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

@app.get("/")
def root():
  """Health check endpoint"""
  return {
    "status": "running",
    "service": "Models Provider Backend"
  }
  
@app.post("/api/analyze/svm")
def get_svm_model_prediction():
  return {"prediction": "cardboard"}

@app.post("/api/analyze/knn")
def get_knn_model_prediction():
  return {"prediction": "plastic"}