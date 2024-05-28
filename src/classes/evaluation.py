import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

class Evaluation:

    @staticmethod
    def evaluate_model(model, features, labels, set_name, output_path, class_names):
        y_pred_proba = model.predict_proba(features)

        labels_binarized = label_binarize(labels, classes=[0, 1, 2])

        roc_auc_dict = {}
        for i in range(len(class_names)):
            if len(np.unique(labels_binarized[:, i])) == 1:
                print(f"Skipping class {class_names[i]} due to only one class present in y_true.")
                continue
            roc_auc_dict[class_names[i]] = roc_auc_score(labels_binarized[:, i], y_pred_proba[:, i])

        if roc_auc_dict:
            macro_roc_auc = np.mean(list(roc_auc_dict.values()))
        else:
            macro_roc_auc = None

        Evaluation.save_roc_curves(labels_binarized, y_pred_proba, set_name, output_path, class_names)

        print(f"{set_name} ROC-AUC scores: {roc_auc_dict}")
        print(f"{set_name} Macro-Average ROC-AUC: {macro_roc_auc}")

        return macro_roc_auc

    @staticmethod
    def save_roc_curves(labels_binarized, y_pred_proba, set_name, output_path, class_names):
        plt.figure()
        for i, class_name in enumerate(class_names):
            if len(np.unique(labels_binarized[:, i])) == 1:
                continue
            fpr, tpr, _ = roc_curve(labels_binarized[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {class_name} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {set_name} Set')
        plt.legend(loc="lower right")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f'{set_name}_roc_curve.png'))
        plt.close()
