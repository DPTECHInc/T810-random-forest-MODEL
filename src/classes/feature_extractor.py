import numpy as np
from skimage.feature import hog

class FeatureExtractor:
    def extract_features(self, images):
        features = []
        for idx, img in enumerate(images):
            try:
                # Ensure the image is grayscale and in 2D
                if img.ndim == 3 and img.shape[-1] == 1:
                    img = img[:, :, 0]
                # Compute HOG features
                hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
                features.append(hog_features)
            except Exception as e:
                print(f"Error processing image at index {idx}: {e}")
        return np.array(features)
