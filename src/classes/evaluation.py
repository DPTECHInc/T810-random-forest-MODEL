import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from .plotting import Plotting

class Evaluation:
    @staticmethod
    def evaluate_model(model, features, labels, set_name, output_path, class_names):
        y_pred = model.predict(features)
        y_pred_proba = model.predict_proba(features)

        if len(np.unique(labels)) > 1:
            roc_auc = roc_auc_score(labels, y_pred_proba, multi_class='ovr')
            print(f'{set_name} ROC-AUC: {roc_auc}')
        else:
            print(f"Cannot compute ROC AUC - only one class present in {set_name.lower()} labels")
            roc_auc = None

        conf_matrix = confusion_matrix(labels, y_pred)
        print(conf_matrix)
        class_report = classification_report(labels, y_pred, target_names=class_names, zero_division=0, output_dict=True)
        print(classification_report(labels, y_pred, target_names=class_names, zero_division=0))

        # Save the classification report plot
        print(f"Generating classification report plot for {set_name.lower()} set...")
        Plotting.plot_classification_report(class_report, f"{output_path}/{set_name.lower()}_classification_report.png")

        return roc_auc
