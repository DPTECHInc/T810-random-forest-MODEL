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
    def plot_classification_report(report, output_path):
        classes = ['NORMAL', 'PNEUMONIA']
        metrics = ['precision', 'recall', 'f1-score']

        report_dict = {}
        for class_name in classes:
            if class_name in report:
                report_dict[class_name] = [report[class_name][metric] for metric in metrics]
            else:
                report_dict[class_name] = [0, 0, 0]  # Defaults to 0 if class is not present

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots()
        for i, class_name in enumerate(classes):
            ax.bar(x + i * width, report_dict[class_name], width, label=class_name)

        ax.set_ylabel('Scores')
        ax.set_title('Classification report by class')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metrics)
        ax.legend()

        fig.tight_layout()
        plt.draw()  # Force the plot to be drawn
        plt.savefig(output_path)
        plt.close()
        print(f"Classification report plot saved to {output_path}")
