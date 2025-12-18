import cv2
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


def predict(dataFilePath, bestModelPath):
    """
    Predict material classes for images in the given folder.
    
    Args:
        dataFilePath (str): Path to folder containing images
        bestModelPath (str): Path to the trained model pickle file
        
    Returns:
        list: List of predicted class IDs for each image
    """

    # Load the trained model
    with open(bestModelPath, 'rb') as f:
        model = pickle.load(f)

    svm_model = model['svm']
    scaler = model['scaler']

    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    data_path = Path(dataFilePath)
    image_files = sorted(list(data_path.glob('*.jpg')) + list(data_path.glob('*.png')) + list(data_path.glob('*.jpeg')))

    predictions = []

    for img_path in image_files:
        image = cv2.imread(str(img_path))

        if image is None:
            # If image is corrupted
            predictions.append(6)
            continue

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to ResNet50 input size
        resized = cv2.resize(rgb_image, (224, 224))

        img_array = np.expand_dims(resized, axis=0)

        preprocessed = preprocess_input(img_array)

        features = resnet_model.predict(preprocessed, verbose=0)
        features = features.flatten()

        features_scaled = scaler.transform(features.reshape(1, -1))

        prediction = svm_model.predict(features_scaled)

        predictions.append(int(prediction[0]))

    return predictions


# Example usage
if __name__ == "__main__":
    dataFilePath = "backend/dataset_augmented/glass"
    bestModelPath = "backend/models/svm_final.pkl"

    predictions = predict(dataFilePath, bestModelPath)

    classes = {
        0: "Glass",
        1: "Paper",
        2: "Cardboard",
        3: "Plastic",
        4: "Metal",
        5: "Trash",
        6: "Unknown"
    }

    print("\nClasses distribution:")
    for class_id in sorted(set(predictions)):
        count = predictions.count(class_id)
        print(f"  {classes[class_id]}: {count} images")
