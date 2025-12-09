# Backend - ML Model Service

This directory will contain the machine learning backend for the Material Stream Identification System.

## Planned Structure

```
backend/
├── models/              # ML models (SVM, k-NN)
│   ├── svm_classifier.py
│   ├── knn_classifier.py
│   └── model_trainer.py
├── utils/               # Utility functions
│   ├── feature_extraction.py
│   ├── data_augmentation.py
│   └── preprocessing.py
├── api/                 # REST API endpoints
│   ├── classify.py
│   └── health.py
├── data/               # Training data (not in version control)
│   ├── raw/
│   ├── augmented/
│   └── processed/
├── requirements.txt    # Python dependencies
├── config.py          # Configuration settings
└── main.py           # API entry point
```

## Future Dependencies

- Python 3.8+
- scikit-learn (SVM, k-NN classifiers)
- OpenCV (image processing)
- NumPy (numerical operations)
- FastAPI or Flask (REST API)
- Pillow (image manipulation)

## Planned Features

1. **Feature Extraction**
   - Convert 2D/3D images to 1D feature vectors
   - Support multiple feature extraction methods
   
2. **Data Augmentation**
   - Rotation, scaling, flipping
   - Color adjustments
   - Noise addition
   - Target: 30% increase in dataset size

3. **Classifiers**
   - SVM with optimized hyperparameters
   - k-NN with configurable weighting schemes
   - Ensemble methods for improved accuracy
   - "Unknown" class for low-confidence predictions

4. **API Endpoints**
   - POST /api/classify - Classify a single image
   - POST /api/classify-batch - Classify multiple images
   - GET /api/health - Service health check
   - GET /api/metrics - Model performance metrics

## Model Requirements

- Minimum validation accuracy: 0.85 across six primary classes
- Support for "unknown" classification when confidence is low
- Real-time processing capability for live camera feeds

## To Be Implemented

This backend is currently a placeholder. Implementation will include:

1. Training pipeline for SVM and k-NN models
2. Feature extraction from waste material images
3. Data augmentation scripts
4. REST API for classification requests
5. Model evaluation and comparison tools
6. Integration with frontend application

## Setup Instructions (Future)

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python models/model_trainer.py

# Start API server
python main.py
```

The API will be available at `http://localhost:8000`
