"""
Data Augmentation for Material Stream Identification
"""

import cv2
import numpy as np
from pathlib import Path
import random


class DataAugmentor:
    """
    Performs data augmentation on the dataset to increase training samples.
    """

    def __init__(self, dataset_path, output_path, target_number_per_class=500):
        """
        Initialize the data augmentor.
        
        Args:
            dataset_path (str): Path to original dataset
            output_path (str): Path to save new augmented dataset
            target_number_per_class (int): Target number of samples per class
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.target_number_per_class = target_number_per_class

        self.classes = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

        # Augmentation Techniques
        self.augmentation_techniques = ['rotate', 'flip_h', 'flip_v', 'brightness_up',
                                        'brightness_down', 'scale_up', 'scale_down']

    def augment_image(self, image, aug_type):
        """
        Apply specific augmentation to an image.
        
        Args:
            image (np.ndarray): Input image
            aug_type (str): Type of augmentation to apply
            
        Returns:
            np.ndarray: Augmented image
        """
        if aug_type == 'rotate':
            angle = random.uniform(-30, 30)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        elif aug_type == 'flip_h':
            image = cv2.flip(image, 1)  # Horizontal flip

        elif aug_type == 'flip_v':
            image = cv2.flip(image, 0)  # Vertical flip

        elif aug_type == 'brightness_up':
            factor = random.uniform(1.1, 1.3)
            image = cv2.convertScaleAbs(image, alpha=factor, beta=0)

        elif aug_type == 'brightness_down':
            factor = random.uniform(0.7, 0.9)
            image = cv2.convertScaleAbs(image, alpha=factor, beta=0)

        elif aug_type == 'scale_up':
            scale = random.uniform(1.1, 1.2)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            image = image_resized[start_y:start_y + h, start_x:start_x + w]

        elif aug_type == 'scale_down':
            scale = random.uniform(0.8, 0.9)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            image = cv2.copyMakeBorder(image_resized, pad_y, pad_y, pad_x, pad_x,
                                       cv2.BORDER_REFLECT)
            image = image[:h, :w]

        return image

    def augment_dataset(self):
        """
        Augment the entire dataset.
        """
        print("DATA AUGMENTATION")
        print("-" * 60)
        print(f"Source: {self.dataset_path}")
        print(f"Output: {self.output_path}")
        print("-" * 60)
        
        if self.output_path.exists():
            print(f"Dataset augmentation directory already exists in {self.output_path}, exiting...\n")
            return

        self.output_path.mkdir(parents=True, exist_ok=True)

        total_original = 0
        total_final = 0

        for class_name in self.classes:
            class_input_dir = self.dataset_path / class_name
            class_output_dir = self.output_path / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)

            all_image_files = list(class_input_dir.glob('*.jpg'))

            # Validate images - filter out corrupted ones
            print(f"\n{class_name.upper()}: Total {len(all_image_files)} images")
            valid_images = []
            for img_path in all_image_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    valid_images.append(img_path)

            image_files = valid_images
            num_original = len(image_files)
            num_corrupted = len(all_image_files) - num_original

            if num_corrupted > 0:
                print(f"Skipped {num_corrupted} corrupt images")

            aug_needed = self.target_number_per_class - num_original
            print(f"  Augmenting: {aug_needed} images")

            for img_path in image_files:
                img = cv2.imread(str(img_path))
                output_path = class_output_dir / img_path.name
                cv2.imwrite(str(output_path), img)

            if aug_needed > 0:
                aug_count = 0
                technique_idx = 0

                while aug_count < aug_needed:
                    for img_path in image_files:
                        if aug_count >= aug_needed:
                            break

                        img = cv2.imread(str(img_path))
                        # Apply augmentation
                        aug_type = self.augmentation_techniques[technique_idx % len(self.augmentation_techniques)]
                        aug_img = self.augment_image(img, aug_type)

                        # Save new image
                        aug_filename = f"aug_{aug_type}_{aug_count}_{img_path.stem}{img_path.suffix}"
                        output_path = class_output_dir / aug_filename
                        cv2.imwrite(str(output_path), aug_img)

                        aug_count += 1
                        technique_idx += 1

            final_count = num_original + aug_needed
            print(f"  Final count: {final_count}")

            total_original += num_original
            total_final += final_count

        print("\n" + "-" * 60)
        print("AUGMENTATION COMPLETE")
        print("-" * 60)
        print(f"Original valid images: {total_original}")
        print(f"Final total images: {total_final}")


def main():
    dataset_path = 'dataset'
    output_path = 'dataset_augmented'

    augmentor = DataAugmentor(
        dataset_path=dataset_path,
        output_path=output_path,
        target_number_per_class=500
    )

    augmentor.augment_dataset()


if __name__ == "__main__":
    main()
