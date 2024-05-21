from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, directory_path, image_size=(64, 64), batch_size=500):
        self.directory_path = directory_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1. / 255)

    def load_images_from_directory(self, subdir, set_name):
        images, labels = [], []
        subdir_path = os.path.join(self.directory_path, subdir)

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

        # Convert labels to categorical
        labels = self.encode_labels(labels)

        print(f"Found {len(images)} images in the {set_name} set.")
        return images, labels

    def load_image(self, file_path):
        image = load_img(file_path, target_size=self.image_size)
        image = img_to_array(image)
        image /= 255.0
        return image

    def encode_labels(self, labels):
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(labels)
        return labels_encoded

    def load_train_images(self):
        return self.load_images_from_directory('train', 'training')

    def load_val_images(self):
        return self.load_images_from_directory('val', 'validation')

    def load_test_images(self):
        generator = self.datagen.flow_from_directory(
            directory=os.path.join(self.directory_path, 'test'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        images, labels = next(generator)
        print(f"Found {generator.samples} images in the test set belonging to {generator.num_classes} classes.")
        return images, labels
