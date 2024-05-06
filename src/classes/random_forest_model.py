import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def save_model(self, path="outputs/trained_models/rf_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path="outputs/trained_models/rf_model.pkl"):
        self.model = joblib.load(path)

    def train(self, X, y):
        X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f'Accuracy: {accuracy:.2f}')
