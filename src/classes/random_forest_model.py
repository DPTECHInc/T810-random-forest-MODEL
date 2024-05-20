import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Tuple, List


class RandomForestModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

    def save_model(self, path: str = "outputs/trained_models/rf_model.pkl") -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, path: str = "outputs/trained_models/rf_model.pkl") -> None:
        try:
            self.model = joblib.load(path)
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        X = self._reshape_data(X)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._reshape_data(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._reshape_data(X)
        return self.model.predict_proba(X)

    def evaluate(self, test_generator: List[Tuple[np.ndarray, np.ndarray]]) -> float:
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

    def _reshape_data(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) > 2:
            return np.reshape(X, (X.shape[0], -1))
        elif X.shape[1] == 1:
            return X.reshape(-1, 1)
        return X
