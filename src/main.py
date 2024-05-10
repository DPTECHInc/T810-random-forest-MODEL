from classes.data_loader import DataLoader
from classes.feature_extractor import FeatureExtractor
from classes.random_forest_model import RandomForestModel
from sklearn.model_selection import train_test_split
from src.classes.data_saver import DataSaver
from sklearn.decomposition import PCA
import numpy as np


def main():
    train_path = "assets/chest_xray/train"
    test_path = "assets/chest_xray/test"

    # Load training images
    data_loader = DataLoader(train_path)
    images, labels = data_loader.load_images()

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    print("Size of training set:", len(x_train))
    print("Size of test set:", len(x_test))

    # Save test and train data
    data_saver = DataSaver()
    data_saver.save_datas(x_test, y_test)

    # Extract features from training images
    extractor = FeatureExtractor()
    train_features = extractor.extract_features(x_train)

    # Check the shape of train_features
    print(f"Shape of train_features before reshaping: {train_features.shape}")

    # Ensure train_features is 2D
    train_features_reshaped = np.reshape(train_features, (train_features.shape[0], -1))
    print(f"Shape of train_features after reshaping: {train_features_reshaped.shape}")

    # Determine the appropriate n_components
    n_samples, n_features = train_features_reshaped.shape
    n_components = min(n_samples, n_features, 50)  # Adjust 50 to a suitable value
    print(f"n_components set to: {n_components}")

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    train_features_reduced = pca.fit_transform(train_features_reshaped)
    print(f"Shape of train_features after PCA: {train_features_reduced.shape}")

    # Train the model on training images' features
    rf_model = RandomForestModel()
    rf_model.train(train_features_reduced, y_train)

    # Load and preprocess test images
    test_images = data_loader.load_test_images(test_path)

    test_features_reduced = []
    test_labels = []

    batch_size = 32  # Reduce the batch size for better performance
    for batch_start in range(0, len(x_test), batch_size):
        batch_end = min(batch_start + batch_size, len(x_test))
        batch_images = x_test[batch_start:batch_end]
        batch_labels = y_test[batch_start:batch_end]

        # Flatten and reduce test features in batches
        batch_features = extractor.extract_features(batch_images)
        print(f"Shape of batch_features before reshaping: {batch_features.shape}")

        batch_features_flattened = np.reshape(batch_features, (batch_features.shape[0], -1))
        print(f"Shape of batch_features after reshaping: {batch_features_flattened.shape}")

        batch_features_reduced = pca.transform(batch_features_flattened)
        print(f"Shape of batch_features after PCA: {batch_features_reduced.shape}")

        test_features_reduced.append(batch_features_reduced)
        test_labels.append(batch_labels)

    # Concatenate all batches
    test_features_reduced = np.vstack(test_features_reduced)
    test_labels = np.hstack(test_labels)
    print(f"Shape of test_features_reduced: {test_features_reduced.shape}")
    print(f"Shape of test_labels: {test_labels.shape}")

    # Prepare test generator
    test_generator = [(test_features_reduced, test_labels)]

    # Evaluate the model on test data
    accuracy = rf_model.evaluate(test_generator)

    # Save the trained model
    rf_model.save_model()


if __name__ == "__main__":
    main()
