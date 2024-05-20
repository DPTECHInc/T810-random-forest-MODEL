from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class DataLoader:
    def __init__(self, directory_path, image_size=(64, 64), batch_size=500):
        self.directory_path = directory_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1./255)

    def load_images_from_directory(self, subdir, set_name):
        generator = self.datagen.flow_from_directory(
            directory=os.path.join(self.directory_path, subdir),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )
        images, labels = next(generator)
        print(f"Found {len(images)} images into the {set_name} set belonging to {generator.num_classes} classes.")
        return images, labels

    def load_train_images(self):
        return self.load_images_from_directory('train', 'training')

    def load_val_images(self):
        return self.load_images_from_directory('val', 'validation')

    def load_test_images(self):
        generator = self.datagen.flow_from_directory(
            directory=os.path.join(self.directory_path, 'test'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        print(f"Found {generator.samples} images into the test set belonging to {generator.num_classes} classes.")
        return generator
