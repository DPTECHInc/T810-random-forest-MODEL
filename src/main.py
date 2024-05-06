from classes.data_loader import DataLoader
from classes.feature_extractor import FeatureExtractor
from classes.random_forest_model import RandomForestModel


def main():

    train_path = "assets/chest_xray/train"
    test_path = "assets/chest_xray/test"

    # load images
    data_loader = DataLoader(train_path)
    images, labels = data_loader.load_images()

    # Extract characteristic
    extractor = FeatureExtractor()
    features = extractor.extract_features(images)

    # Train and evaluate model
    rf_model = RandomForestModel()
    rf_model.train(features, labels)

    # Load test images
    test_images, test_labels = data_loader.load_test_images(test_path)
    test_features = extractor.extract_features(test_images)

    # Evaluate model and test data
    rf_model.evaluate()
    rf_model.save_model()

if __name__ == "__main__":
    main()
