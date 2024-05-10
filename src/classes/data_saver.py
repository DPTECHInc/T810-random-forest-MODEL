import os

import numpy as np


class DataSaver:
    def save_datas(self, features, labels, features_path='outputs/test_data/X_test.npy',
                   labels_path='outputs/test_data/y_test.npy'):

        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)

        np.save(features_path, features, )
        np.save(labels_path, labels)
        print("Les données de test ont été sauvegardées avec succès.")

    def __init__(self):
        pass
