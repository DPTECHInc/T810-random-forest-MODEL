from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


class DataLoader:
    def __init__(self, base_path, image_size=(150, 150), batch_size=32):
        self.base_path = base_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def load_image(self, file_path):
        image = load_img(file_path, target_size=self.image_size)
        image = img_to_array(image)
        image /= 255.0  # Normaliser les pixels entre 0 et 1
        return image

    def load_images_from_directory(self, subdir, set_name):
        images, labels = [], []
        subdir_path = os.path.join(self.base_path, subdir)

        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(subdir_path, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.jpeg'):
                    file_path = os.path.join(class_dir, file_name)
                    image = self.load_image(file_path)
                    images.append(image)
                    if class_name == 'NORMAL':
                        labels.append('NORMAL')
                    elif 'bacteria' in file_name:
                        labels.append('BACTERIA')
                    elif 'virus' in file_name:
                        labels.append('VIRUS')

        images = np.array(images)
        labels = np.array(labels)

        # Convertir les étiquettes en catégories
        labels = self.encode_labels(labels)

        print(f"Found {len(images)} images in the {set_name} set.")
        return images, labels

    def encode_labels(self, labels):
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(labels)
        labels_categorical = to_categorical(labels_encoded)
        return labels_categorical

    def load_train_images(self):
        train_images, train_labels = self.load_images_from_directory('train', 'training')
        train_images_augmented = self.datagen.flow(train_images, train_labels, batch_size=len(train_images),
                                                   shuffle=False)
        return next(train_images_augmented)

    def load_val_images(self):
        return self.load_images_from_directory('val', 'validation')

    def load_test_images(self):
        return self.load_images_from_directory('test', 'test')
