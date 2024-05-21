import numpy as np

class ClassDistribution:
    @staticmethod
    def check_class_distribution(labels, set_name):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{set_name} set class distribution: {dict(zip(unique, counts))}")

    @staticmethod
    def detailed_class_distribution(labels, set_name):
        unique, counts = np.unique(labels, return_counts=True)
        class_labels = {0: "NORMAL", 1: "PNEUMONIA"}
        distribution = {class_labels[label]: count for label, count in zip(unique, counts)}
        print(f"{set_name} set detailed class distribution: {distribution}")
