import numpy as np
from sklearn.decomposition import PCA

from .feature_extractor import FeatureExtractor


class Preprocessing:
    def __init__(self):
        self.pca = None

    def preprocess_data(self, images, fit=True):
        extractor = FeatureExtractor()
        features = extractor.extract_features(images)
        features_reshaped = np.reshape(features, (features.shape[0], -1))

        if fit:
            n_components = min(features_reshaped.shape[0], features_reshaped.shape[1], 50)
            print(f"n_components set to: {n_components}")
            self.pca = PCA(n_components=n_components)
            features_reduced = self.pca.fit_transform(features_reshaped)
        else:
            features_reduced = self.pca.transform(features_reshaped)

        return features_reduced

    def preprocess_test_data(self, test_generator):
        test_features_reduced = []
        test_labels = []

        extractor = FeatureExtractor()

        for i in range(len(test_generator)):
            batch_images, batch_labels = test_generator[i]
            batch_features = extractor.extract_features(batch_images)
            batch_features_reshaped = np.reshape(batch_features, (batch_features.shape[0], -1))
            batch_features_reduced = self.pca.transform(batch_features_reshaped)

            test_features_reduced.append(batch_features_reduced)
            test_labels.append(batch_labels)

        test_features_reduced = np.vstack(test_features_reduced)
        test_labels = np.hstack(test_labels)

        return test_features_reduced, test_labels
