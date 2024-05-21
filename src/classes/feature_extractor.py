import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray

class FeatureExtractor:
    def extract_features(self, images):
        features = []
        for i, image in enumerate(images):
            try:
                print(f"Processing image {i+1}/{len(images)}")
                image = rgb2gray(image)  # Convert to grayscale
                feature = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
                features.append(feature)
            except Exception as e:
                print(f"Error processing image at index {i}: {str(e)}")
        return np.array(features)
