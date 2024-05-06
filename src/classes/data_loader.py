from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    def __init__(self, directory_path, image_size=(64, 64), batch_size=500):
        self.directory_path = directory_path
        self.image_size = image_size
        self.batch_size = batch_size

    def load_images(self):
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            directory=self.directory_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        images, labels = next(generator)
        return images, labels

    def load_test_images(self,test_path):
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            directory=test_path,
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )
        return generator
