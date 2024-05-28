import json
import nbformat as nbf
import os
from datetime import datetime


class NotebookGenerator:
    @staticmethod
    def create_notebook(run_number, params, scores, output_path, train_y_true, train_y_score, val_y_true, val_y_score,
                        test_y_true, test_y_score):
        run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        nb = nbf.v4.new_notebook()
        text = f"""
        # Run {run_number}

        ## Parameters
        ```
        {json.dumps(params, indent=4)}
        ```

        ## Scores
        ```
        Train ROC-AUC: {scores['train_roc_auc']}
        Validation ROC-AUC (Cross-Validation): {scores['val_roc_auc_cv']}
        Validation ROC-AUC: {scores['val_roc_auc']}
        Test ROC-AUC: {scores['test_roc_auc']}
        ```

        ## Graphs
        """

        # Generate the notebook cells
        cells = [nbf.v4.new_markdown_cell(text)]

        # Insert the code to generate the graphs
        graph_code = f"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report
import numpy as np

def plot_roc_curve(y_true, y_score, class_names, title='ROC Curve'):
    plt.figure()
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve of class {{class_name}} (area = {{roc_auc:.2f}})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_class_distribution(labels, class_names, title='Class Distribution'):
    plt.figure()
    sns.countplot(labels)
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.show()

# Classes
class_names = ['NORMAL', 'BACTERIA', 'VIRUS']

# Plot ROC curves
plot_roc_curve(np.array({train_y_true.tolist()}), np.array({train_y_score.tolist()}), class_names, 'ROC Curve - Training Set')
plot_roc_curve(np.array({val_y_true.tolist()}), np.array({val_y_score.tolist()}), class_names, 'ROC Curve - Validation Set')
plot_roc_curve(np.array({test_y_true.tolist()}), np.array({test_y_score.tolist()}), class_names, 'ROC Curve - Test Set')

# Plot Class Distributions
plot_class_distribution(np.array({train_y_true.tolist()}), class_names, 'Class Distribution - Training Set')
plot_class_distribution(np.array({val_y_true.tolist()}), class_names, 'Class Distribution - Validation Set')
plot_class_distribution(np.array({test_y_true.tolist()}), class_names, 'Class Distribution - Test Set')

# Classification reports
print("Training Set Classification Report:")
print(classification_report(np.array({train_y_true.tolist()}), np.array({train_y_score.argmax(axis=1).tolist()}), target_names=class_names))

print("Validation Set Classification Report:")
print(classification_report(np.array({val_y_true.tolist()}), np.array({val_y_score.argmax(axis=1).tolist()}), target_names=class_names))

print("Test Set Classification Report:")
print(classification_report(np.array({test_y_true.tolist()}), np.array({test_y_score.argmax(axis=1).tolist()}), target_names=class_names))
"""
        cells.append(nbf.v4.new_code_cell(graph_code))

        nb['cells'] = cells
        notebook_path = os.path.join(output_path, f'run_{run_time}.ipynb')
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)
        print(f"Notebook saved to {notebook_path}")
