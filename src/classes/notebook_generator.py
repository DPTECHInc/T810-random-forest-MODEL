import json
import os

import nbformat as nbf
from datetime import datetime

class NotebookGenerator:
    @staticmethod
    def create_notebook(run_number, params, scores, output_path):
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
        ![ROC Curve - Training](training_roc_curve.png)
        ![ROC Curve - Validation](validation_roc_curve.png)
        ![ROC Curve - Test](test_roc_curve.png)
        ![Classification Report - Training](training_classification_report.png)
        ![Classification Report - Validation](validation_classification_report.png)
        ![Classification Report - Test](test_classification_report.png)
        ![Class Distribution - Training](training_class_distribution.png)
        ![Class Distribution - Validation](validation_class_distribution.png)
        ![Class Distribution - Test](test_class_distribution.png)
        """
        nb['cells'] = [nbf.v4.new_markdown_cell(text)]
        notebook_path = os.path.join(output_path, f'run_{run_time}.ipynb')
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)
        print(f"Notebook saved to {notebook_path}")
