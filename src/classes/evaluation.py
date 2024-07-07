import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

class Evaluation:

    @staticmethod
    def evaluate_model(model, features, labels, set_name, output_path, class_names, metric="roc_auc_ovr"):
        if metric == 'accuracy':
            y_pred = model.predict(features)
            accuracy = accuracy_score(labels, y_pred)
            Evaluation.save_accuracy_curve(labels, y_pred, set_name, output_path)
            return accuracy
        elif metric == 'f1':
            y_pred = model.predict(features)
            f1 = f1_score(labels, y_pred, average='macro')
            return f1
        else:
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

            return macro_roc_auc

    @staticmethod
    def adjust_threshold(model, features, labels, output_path, step=0.01):
        y_pred_proba = model.predict_proba(features)[:, 1]
        best_threshold = 0.5
        best_accuracy = 0

        for threshold in np.arange(0.0, 1.0, step):
            y_pred = (y_pred_proba >= threshold).astype(int)
            accuracy = accuracy_score(labels, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        # Save best threshold and accuracy
        with open(os.path.join(output_path, 'best_threshold.txt'), 'w') as file:
            file.write(f"Best threshold: {best_threshold}\nBest accuracy: {best_accuracy}")

        return best_threshold, best_accuracy

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

    @staticmethod
    def save_accuracy_curve(labels, y_pred, set_name, output_path):
        plt.figure()
        correct_predictions = (labels == y_pred)
        accuracy_values = np.cumsum(correct_predictions) / np.arange(1, len(correct_predictions) + 1)
        plt.plot(accuracy_values, lw=2, label='Accuracy over samples')
        plt.xlabel('Sample index')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curve - {set_name} Set')
        plt.legend(loc="lower right")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f'{set_name}_accuracy_curve.png'))
        plt.close()
