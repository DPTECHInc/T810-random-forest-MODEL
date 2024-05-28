import matplotlib.pyplot as plt
import numpy as np


class ClassDistribution:


    @staticmethod
    def detailed_class_distribution(labels, set_name, run_number):
        class_labels = {0: 'NORMAL', 1: 'BACTERIA', 2: 'VIRUS'}
        unique, counts = np.unique(labels, return_counts=True)

        # Initialize distribution dictionary with all class labels
        distribution = {class_labels[label]: 0 for label in class_labels}
        for label, count in zip(unique, counts):
            distribution[class_labels[label]] = count

        plt.figure()
        plt.bar(distribution.keys(), distribution.values())
        plt.xlabel('Classes')
        plt.ylabel('Counts')
        plt.title(f'Class Distribution in {set_name} Set')
        plt.savefig(f'outputs/reports/run_{run_number}/{set_name.lower()}_class_distribution.png')
        plt.close()
        print(f"Detailed class distribution in {set_name} set: {distribution}")
