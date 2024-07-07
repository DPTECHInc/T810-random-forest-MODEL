import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.datagen = ImageDataGenerator(rescale=1./255)

    def load_images_from_directory(self, subdir, set_name):
        directory = os.path.join(self.base_path, subdir)

        generator = self.datagen.flow_from_directory(
            directory,
            target_size=(224, 224),  # Correct target size for grayscale images
            batch_size=32,
            color_mode="grayscale",  # Ensure color_mode is set to grayscale
            class_mode='categorical',
            shuffle=True
        )

        if generator.samples == 0:
            logging.error(f"No images found in {directory}. Please check the dataset path.")
            return np.array([]), np.array([])

        images, labels = [], []
        try:
            for _ in range(len(generator)):
                img, lbl = next(generator)
                img = np.squeeze(img, axis=-1)
                images.extend(img)
                labels.extend(lbl)
            images = np.array(images)
            labels = np.argmax(np.array(labels), axis=1)
        except Exception as e:
            logging.error(f"Error loading images from {directory}: {e}")
            return np.array([]), np.array([])

        return images, labels

    def encode_labels(self, labels):
        logging.info("Encoding labels.")
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(labels)
        labels_categorical = to_categorical(labels_encoded)
        logging.info("Labels encoded.")
        return labels_categorical

    def load_train_images(self):
        logging.info("Loading training images.")
        return self.load_images_from_directory('train', 'training')

    def load_val_images(self):
        logging.info("Loading validation images.")
        return self.load_images_from_directory('val', 'validation')

    def load_test_images(self):
        logging.info("Loading test images.")
        return self.load_images_from_directory('test', 'test')
