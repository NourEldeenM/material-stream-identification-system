"""
Feature Extraction for Material Stream Identification
Convert raw images into 1D fixed-size numerical feature vectors.
"""

import cv2
import numpy as np
from pathlib import Path
import pickle


class FeatureExtractor:
    """
    Extracts features from images to convert from pixels to features.
    Uses Histogram of Oriented Gradients and Color Histograms.
    """

    def __init__(self, image_size=(256, 256)):
        """
        Initialize the feature extractor.
        
        Args:
            image_size (tuple): Target size for resizing images
        """
        self.image_size = image_size

        # HOG parameters
        self.hog = cv2.HOGDescriptor(
            _winSize=(256, 256),  # Size of detection window
            _blockSize=(32, 32),  # Size of blocks for normalization
            _blockStride=(16, 16),  # Stride of blocks (How much blocks overlap)
            _cellSize=(16, 16),  # Size of cells within blocks
            _nbins=9  # Number of orientation bins
        )

        self.classes = {
            'glass': 0,
            'paper': 1,
            'cardboard': 2,
            'plastic': 3,
            'metal': 4,
            'trash': 5
        }

    def extract_hog_features(self, image):
        """
        Extract HOG features from image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: HOG feature vector
        """
        # Resize image
        resized = cv2.resize(image, self.image_size)

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Compute HOG features
        hog_features = self.hog.compute(gray)

        return hog_features.flatten()

    def extract_color_histogram(self, image):
        """
        Extract color histogram features from image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Color histogram feature vector
        """
        resized = cv2.resize(image, self.image_size)

        # Compute histogram for each color channel
        hist_features = []
        for i in range(3):  # BGR
            hist = cv2.calcHist([resized], [i], None, [64], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)  # Normalize
            hist_features.append(hist)

        return np.concatenate(hist_features)

    def extract_features(self, image):
        """
        Extract combined feature vector from image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Combined feature vector
        """
        # Extract HOG features
        hog_features = self.extract_hog_features(image)

        # Extract color histogram features
        color_features = self.extract_color_histogram(image)

        # Combine features
        features = np.concatenate([hog_features, color_features])

        return features

    def extract_from_dataset(self, dataset_path, output_path):
        """
        Extract features from entire dataset.
        
        Args:
            dataset_path (str): Path to dataset directory
            output_path (str): Path to save extracted features
        """
        print("=" * 60)
        print("FEATURE EXTRACTION")
        print("=" * 60)
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_path}")
        print("-" * 60)

        dataset_path = Path(dataset_path)

        all_features = []
        all_labels = []

        for class_name, class_id in self.classes.items():
            class_dir = dataset_path / class_name

            image_files = list(class_dir.glob('*.jpg'))

            print(f"\n{class_name.upper()} (ID: {class_id}): {len(image_files)} images")

            # Extract features from each image
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    features = self.extract_features(img)
                    all_features.append(features)
                    all_labels.append(class_id)

        X = np.array(all_features)
        y = np.array(all_labels)

        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Total samples: {X.shape[0]}")
        print(f"Feature vector size: {X.shape[1]}")
        print(f"Class distribution: {np.bincount(y)}")

        # Save features
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'features': X,
            'labels': y,
            'class_mapping': self.classes
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\nFeatures saved to: {output_path}")
        print("=" * 60)


def main():
    """
    Run feature extraction on the augmented dataset.
    """
    dataset_path = 'dataset_augmented'
    output_path = 'features/extracted_features.pkl'

    # Initialize feature extractor
    extractor = FeatureExtractor(image_size=(256, 256))

    # Extract features
    extractor.extract_from_dataset(dataset_path, output_path)


if __name__ == "__main__":
    main()
