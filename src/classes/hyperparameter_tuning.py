from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


class HyperparameterTuning:

    def randomized_search_hyperparameters(self, features, labels, cv, param_dist, scoring):
        rf = RandomForestClassifier(random_state=42)

        total_space_size = np.prod([len(v) for v in param_dist.values()])

        n_iter = min(100, total_space_size)

        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(features, labels)
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        return best_params, best_score