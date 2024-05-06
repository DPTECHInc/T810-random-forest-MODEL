import numpy as np


class FeatureExtractor:

    def extract_features(self, images):
        features = images.mean(axis=(1, 2, 3))
        return features
