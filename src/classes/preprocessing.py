import numpy as np
from sklearn.decomposition import PCA

from .feature_extractor import FeatureExtractor

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.decomposition import PCA


class Preprocessing:
    def __init__(self):
        self.pca = None

    def preprocess_data(self, images, labels, fit=True, balance_method=None):
        extractor = FeatureExtractor()
        features = extractor.extract_features(images)
        features_reshaped = np.reshape(features, (features.shape[0], -1))

        if balance_method == 'oversample':
            ros = RandomOverSampler(random_state=42)
            features_reshaped, labels = ros.fit_resample(features_reshaped, labels)
        elif balance_method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            features_reshaped, labels = rus.fit_resample(features_reshaped, labels)

        if fit:
            n_components = min(features_reshaped.shape[0], features_reshaped.shape[1], 50)
            self.pca = PCA(n_components=n_components)
            features_reduced = self.pca.fit_transform(features_reshaped)
        else:
            features_reduced = self.pca.transform(features_reshaped)

        return features_reduced, labels

    def preprocess_test_data(self, test_images):
        extractor = FeatureExtractor()
        test_features = extractor.extract_features(test_images)
        test_features_reshaped = np.reshape(test_features, (test_features.shape[0], -1))
        test_features_reduced = self.pca.transform(test_features_reshaped)
        return test_features_reduced
