import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score, recall_score, \
    classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns


class Evaluation:

    @staticmethod
    def evaluate_model(model, features, labels, set_name, output_path, class_names, metric="roc_auc_ovr"):
        y_pred = model.predict(features)
        y_pred_proba = model.predict_proba(features)

        # Calcul des métriques
        accuracy = accuracy_score(labels, y_pred)
        f1 = f1_score(labels, y_pred, average='macro')
        recall = recall_score(labels, y_pred, average='macro')
        roc_auc = roc_auc_score(label_binarize(labels, classes=range(len(class_names))), y_pred_proba, average='macro',
                                multi_class='ovr')

        # Sauvegarder les courbes et le rapport de classification
        Evaluation.save_roc_curves(labels, y_pred_proba, set_name, output_path, class_names)
        Evaluation.save_accuracy_curve(labels, y_pred, set_name, output_path)
        Evaluation.save_classification_report(labels, y_pred, set_name, output_path, class_names)
        Evaluation.save_confusion_matrix(labels, y_pred, set_name, output_path, class_names)

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'recall': recall,
            'roc_auc': roc_auc
        }

        return metrics

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
    def save_roc_curves(labels, y_pred_proba, set_name, output_path, class_names):
        labels_binarized = label_binarize(labels, classes=range(len(class_names)))
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

    @staticmethod
    def save_classification_report(labels, y_pred, set_name, output_path, class_names):
        report = classification_report(labels, y_pred, target_names=class_names, output_dict=True)
        report_path = os.path.join(output_path, f'{set_name}_classification_report.txt')
        with open(report_path, 'w') as file:
            for class_name, metrics in report.items():
                file.write(f"{class_name}:\n")
                for metric, value in metrics.items():
                    file.write(f"  {metric}: {value:.4f}\n")
                file.write("\n")

    @staticmethod
    def save_confusion_matrix(labels, y_pred, set_name, output_path, class_names):
        cm = confusion_matrix(labels, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {set_name} Set')
        plt.savefig(os.path.join(output_path, f'{set_name}_confusion_matrix.png'))
        plt.close()
