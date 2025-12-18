"""
Feature Extraction for Material Stream Identification
Convert raw images into 1D fixed-size numerical feature vectors.
"""

import cv2
import numpy as np
from pathlib import Path
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


class FeatureExtractor:
    """
    Extracts features from images to convert from pixels to features.
    Uses ResNet50 CNN pre-trained on ImageNet.
    """

    def __init__(self):
        """
        Initialize the feature extractor with ResNet50.
        """
        # Load pre-trained ResNet50 without top classification layer (We want features not predictions)
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

        self.classes = {
            'glass': 0,
            'paper': 1,
            'cardboard': 2,
            'plastic': 3,
            'metal': 4,
            'trash': 5
        }

    def extract_features(self, image):
        """
        Extract ResNet50 CNN features from image.
        
        Args:
            image (np.ndarray): Input image (BGR format from cv2)
            
        Returns:
            np.ndarray: ResNet50 feature vector (2048 dimensions)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to ResNet50 input size
        resized = cv2.resize(rgb_image, (224, 224))

        # Expand dimensions for batch (Neural networks expect batches of images, not single images)
        img_array = np.expand_dims(resized, axis=0)

        # Preprocess for ResNet50
        preprocessed = preprocess_input(img_array)

        # Extract features
        features = self.model.predict(preprocessed, verbose=0)

        return features.flatten()

    def extract_from_dataset(self, dataset_path, output_path):
        """
        Extract features from dataset.
        
        Args:
            dataset_path (str): Path to dataset directory
            output_path (str): Path to save extracted features
        """
        print("FEATURE EXTRACTION")
        print("-" * 60)
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_path}")
        print("-" * 60)
        
        output_path = Path(output_path)
        
        if output_path.exists():
            print(f"Features extraction already exists in {output_path}, exiting...\n")
            return

        dataset_path = Path(dataset_path)

        Features = []
        Labels = []

        for class_name, class_id in self.classes.items():
            class_dir = dataset_path / class_name

            images = list(class_dir.glob('*.jpg'))

            print(f"\n{class_name.upper()} (ID: {class_id}): {len(images)} images")

            # Extract features from each image
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    features = self.extract_features(img)
                    Features.append(features)
                    Labels.append(class_id)

        X = np.array(Features)
        y = np.array(Labels)

        print("\n" + "-" * 60)
        print("EXTRACTION COMPLETE")
        print("-" * 60)
        print(f"Total samples: {X.shape[0]}")
        print(f"Feature vector size: {X.shape[1]}")

        # Save features
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'features': X,
            'labels': y,
            'class_mapping': self.classes
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\nFeatures saved to: {output_path}")


def main():
    """
    Run feature extraction on the augmented dataset.
    """
    dataset_path = 'dataset_augmented'
    output_path = 'features/extracted_features.pkl'

    extractor = FeatureExtractor()

    extractor.extract_from_dataset(dataset_path, output_path)


if __name__ == "__main__":
    main()
