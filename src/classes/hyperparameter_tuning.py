from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


class HyperparameterTuning:

    @staticmethod
    def randomized_search_hyperparameters(features, labels, cv):
        param_dist = {
            'n_estimators': [100, 200, 400, 800],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=100,
            scoring='roc_auc_ovr',
            cv=cv,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(features, labels)
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        return best_params, best_score
