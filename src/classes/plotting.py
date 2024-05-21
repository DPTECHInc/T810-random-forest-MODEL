import matplotlib.pyplot as plt
import numpy as np


class Plotting:
    @staticmethod
    def plot_roc_auc_scores(scores, output_path):
        plt.figure()
        plt.plot(scores, marker='o')
        plt.title('ROC-AUC Scores from Cross-Validation')
        plt.xlabel('Fold')
        plt.ylabel('ROC-AUC Score')
        plt.ylim([0, 1])
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        print(f"ROC-AUC scores plot saved to {output_path}")

    @staticmethod
    def plot_classification_report(class_report, output_path):
        classes = list(class_report.keys())[:-3]  # Exclure 'accuracy', 'macro avg', 'weighted avg'
        metrics = ['precision', 'recall', 'f1-score']

        scores = np.zeros((len(classes), len(metrics)))

        for i, cls in enumerate(classes):
            for j, metric in enumerate(metrics):
                scores[i, j] = class_report[cls][metric]

        fig, ax = plt.subplots()
        x = np.arange(len(classes))
        width = 0.2

        for j in range(len(metrics)):
            ax.bar(x + j * width, scores[:, j], width, label=metrics[j])

        ax.set_xlabel('Classes')
        ax.set_ylabel('Scores')
        ax.set_title('Classification report by class')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_class_distribution(y_true, y_pred, class_names, output_path):
        classes = np.unique(y_true)
        true_counts = {cls: 0 for cls in classes}
        pred_counts = {cls: 0 for cls in classes}

        for cls in classes:
            true_counts[cls] = np.sum(y_true == cls)
            pred_counts[cls] = np.sum(y_pred == cls)

        true_values = [true_counts[cls] for cls in classes]
        pred_values = [pred_counts[cls] for cls in classes]

        fig, ax = plt.subplots()
        width = 0.35
        x = np.arange(len(classes))

        ax.bar(x - width / 2, true_values, width, label='True')
        ax.bar(x + width / 2, pred_values, width, label='Predicted')

        ax.set_xlabel('Classes')
        ax.set_ylabel('Counts')
        ax.set_title('Class distribution: True vs Predicted')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
