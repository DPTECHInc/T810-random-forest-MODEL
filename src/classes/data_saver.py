import os
import joblib
import json
import numpy as np


class DataSaver:
    def save_datas(self, x_test, y_test, path="outputs/test_data/test_data.npz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, x_test=x_test, y_test=y_test)
        print("Les données de test ont été sauvegardées avec succès.")

    def load_previous_results(self, path="outputs/reports/reports.json"):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def save_results(self, results, path="outputs/reports/reports.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        previous_results = self.load_previous_results(path)

        run_keys = [key for key in previous_results.keys() if key.startswith('run_')]
        if run_keys:
            next_index = max(int(key.split('_')[1]) for key in run_keys) + 1
        else:
            next_index = 1

        results_key = f"run_{next_index}"
        previous_results[results_key] = results

        with open(path, 'w') as f:
            json.dump(previous_results, f, indent=4)
        print(f"Les résultats de {results_key} ont été sauvegardés avec succès.")

    def save_model(self, model, path="outputs/models/random_forest_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        print(f"Le modèle a été sauvegardé avec succès à {path}.")

    def save_pca(self, pca, path="outputs/models/pca_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(pca, path)
        print(f"Le modèle PCA a été sauvegardé avec succès à {path}.")
