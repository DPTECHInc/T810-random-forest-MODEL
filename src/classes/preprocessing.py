import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.decomposition import PCA
from .feature_extractor import FeatureExtractor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessing:
    def __init__(self):
        self.pca = None

    def preprocess_data(self, images, labels, fit=True, balance_method=None):
        logging.info("Starting preprocessing data.")

        extractor = FeatureExtractor()
        features = extractor.extract_features(images)
        if features.size == 0:
            logging.error("No features were extracted. Please check the input images.")
            raise ValueError("No features were extracted. Please check the input images.")
        features_reshaped = np.reshape(features, (features.shape[0], -1))

        logging.info(f"Original features shape: {features_reshaped.shape}")
        logging.info(f"Original labels shape: {labels.shape}")

        if balance_method == 'oversample':
            logging.info("Applying Random OverSampling.")
            ros = RandomOverSampler(random_state=42)
            features_reshaped, labels = ros.fit_resample(features_reshaped, labels)
            logging.info("After oversampling.")
        elif balance_method == 'smote':
            logging.info("Applying SMOTE.")
            smote = SMOTE(random_state=42)
            features_reshaped, labels = smote.fit_resample(features_reshaped, labels)
            logging.info("After SMOTE.")

        logging.info(f"Balanced features shape: {features_reshaped.shape}")
        logging.info(f"Balanced labels shape: {labels.shape}")

        if fit:
            logging.info("Fitting PCA.")
            n_components = min(features_reshaped.shape[0], features_reshaped.shape[1], 50)
            self.pca = PCA(n_components=n_components)
            features_reduced = self.pca.fit_transform(features_reshaped)
            logging.info("PCA fitting complete.")
        else:
            features_reduced = self.pca.transform(features_reshaped)
            logging.info("PCA transformation complete.")

        logging.info(f"Reduced features shape: {features_reduced.shape}")

        logging.info("Preprocessing data complete.")
        return features_reduced, labels