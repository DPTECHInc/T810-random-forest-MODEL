from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold

class HyperparameterTuning:
    @staticmethod
    def grid_search_hyperparameters(train_features, train_labels):
        param_grid = {
            'n_estimators': [100, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 4]
        }

        rf = RandomForestClassifier(random_state=42)
        scoring = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scoring, cv=skf, n_jobs=-1)
        grid_search.fit(train_features, train_labels)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation ROC-AUC score: {best_score}")

        return best_params, best_score
