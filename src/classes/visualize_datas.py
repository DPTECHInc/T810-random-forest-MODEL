import matplotlib.pyplot as plt
import json
from IPython.display import Image, display
import os


class VisualizeDatas:

    @staticmethod
    def display_images(run_folder, scoring):
        """
        Display the ROC curve images for Training, Validation, and Test sets.

        Args:
        run_folder (str): Path to the run folder containing the images.
        """
        images = []
        if scoring == "roc_auc":
            images = ["training_roc_curve.png", "validation_roc_curve.png", "test_roc_curve.png"]
        else:
            images = ["training_accuracy.png", "validation_accuracy.png", "test_accuracy.png"]
        for img in images:
            img_path = os.path.join(run_folder, img)
            if os.path.exists(img_path):
                display(Image(filename=img_path))
                plt.show()
            else:
                print(f"Image {img} does not exist in the folder {run_folder}.")

    @staticmethod
    def display_json_report(run_folder):
        """
        Display the technical details JSON report.

        Args:
        run_folder (str): Path to the run folder containing the JSON report.
        """
        report_path = os.path.join(run_folder, "technical_details.json")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
                print(json.dumps(report, indent=4))
        else:
            print(f"Technical details report does not exist in the folder {run_folder}.")
