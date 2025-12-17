# material-stream-identification-system
An Automated Material Stream Identification (MSI) System that efficiently automates sorting of post-consumer waste using fundamental Machine Learning (ML) techniques therefore achieving circular economy goals. 

## Documentation
[Main Project Documentation](https://docs.google.com/document/d/1-nBlcD_kVZX6oGrfCbp0lg60r7hUdXLqMtu3Tv3RlrM/edit?usp=sharing)

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
