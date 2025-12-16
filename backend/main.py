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
  
@app.post("/api/analyze")
def get_model_prediction():
  return {"prediction": "cardboard"}