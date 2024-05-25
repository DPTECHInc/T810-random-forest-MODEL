from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class HyperparameterTuning:

    @staticmethod
    def grid_search_hyperparameters(features, labels, cv):
        param_grid = {
            'n_estimators': [100, 400, 600, 1000],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 5, 20, 60, 100],
            'min_samples_split': [4, 10, 20, 40],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='roc_auc_ovr', n_jobs=-1)
        grid_search.fit(features, labels)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        return best_params, best_score
