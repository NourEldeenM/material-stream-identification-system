# material-stream-identification-system

An Automated Material Stream Identification (MSI) System that efficiently automates sorting of post-consumer waste using fundamental Machine Learning (ML) techniques therefore achieving circular economy goals.

## How To Run

1. Navigate to frontend directory and run:

   ```bash
   npm install # for the first time only
   npm run dev
   ```

- Then goto `http://localhost:5173` in your browser

2. Navigate to backend directory and run:

   ```bash
    pip install -r requirements.txt # for the first time only
    uvicorn main:app --reload # add (--workers 2) for concurrent processing
   ```

## Functional Requirements

1. Data Augmentation and Feature Extraction
2. Classifier Implementation (SVM, k-NN)
3. Interface with Live Camera (real-time) processing

## Non-Functional Requirements

1. Apply Data Augmentation techniques to increase training dataset by 30%
2. Try to achieve minimum validation accuracy of 0.85 accross the six primary classes
3. Raw 2D and 3D images should be converted into 1D numerical feature vector
4. The SVM classifier must be designed to accept the extracted feature vector as input.
5. The k-NN classifier must be designed to accept the extracted feature vector as input, and the weighting scheme (e.g., uniform, distance-based) for the classifier.
6. There should be an `unknown` class that is selected if the model couldn't classify the data properly

## Documentation Requirements

Comprehensive Technical Report (PDF): A formal document including a section comparing the chosen feature extraction methods and classifier performance.
It should also include answers for what, why, how questions when asked in Functional Requirements implementations.

## Implementation Summary

### Feature Extraction

ResNet50 CNN with 50 layers pre-trained on ImageNet dataset. Extracts 2,048-dimensional feature vectors using average pooling. Images are resized to 224x224x3 RGB format. The model achieves 95.33% testing accuracy with SVM and 93.17% with k-NN.

### Classifiers

**k-Nearest Neighbors (k-NN)**

- Non-parametric instance-based learning
- Optimal parameters: k=3, distance-weighted, Euclidean metric
- StandardScaler normalization applied
- 5-fold stratified cross-validation
- Confidence-based rejection mechanism with threshold 0.6
- Testing accuracy: 93.17%

**Support Vector Machine (SVM)**

- RBF kernel for non-linear decision boundaries
- One-vs-Rest strategy for multi-class classification
- Parameters: C=3, gamma=scale
- Probability calibration enabled
- Testing accuracy: 95.33%

### Data Augmentation

Initial dataset of 1,960 images increased to 3,000 balanced images (500 per class). Seven augmentation techniques applied: rotation, horizontal/vertical flip, brightness adjustment, and scaling. 95 corrupt images were skipped during processing.

### System Architecture

Real-time classification pipeline: Camera capture → Preprocessing → ResNet50 feature extraction → SVM/k-NN classification → Web UI display. Backend uses FastAPI, TensorFlow/Keras, scikit-learn, and OpenCV. Frontend built with React.js and Vite.
