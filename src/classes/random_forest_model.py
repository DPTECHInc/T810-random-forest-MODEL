import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def save_model(self, path="outputs/trained_models/rf_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path="outputs/trained_models/rf_model.pkl"):
        self.model = joblib.load(path)

    def train(self, X, y):
        # Ensure X is reshaped correctly
        X = self._reshape_data(X)
        self.model.fit(X, y)

    def evaluate(self, test_generator):
        predictions = []
        true_labels = []
        for batch_images, batch_labels in test_generator:
            batch_images_flattened = self._reshape_data(batch_images)
            batch_predictions = self.model.predict(batch_images_flattened)
            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)

        accuracy = accuracy_score(true_labels, predictions)
        print(f'Accuracy: {accuracy:.2f}')
        return accuracy

    def _reshape_data(self, X):
        # Flatten the images if they have more than 2 dimensions
        if len(X.shape) > 2:
            return np.reshape(X, (X.shape[0], -1))
        # Ensure correct shape if single feature
        elif X.shape[1] == 1:
            return X.reshape(-1, 1)
        return X
