from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

class RandomForestModel:
    def __init__(self, n_estimators=100, max_features='auto', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @staticmethod
    def about():
        print("tf version:", tf.__version__)
        print("Number of GPU avalaible:", len(tf.config.list_physical_devices("GPU")))
        print(tf.config.list_physical_devices("GPU"))

