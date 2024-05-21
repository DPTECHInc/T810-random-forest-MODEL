import numpy as np
from classes.data_loader import DataLoader
from classes.data_saver import DataSaver
from classes.class_distribution import ClassDistribution
from classes.preprocessing import Preprocessing
from classes.hyperparameter_tuning import HyperparameterTuning
from classes.evaluation import Evaluation
from classes.random_forest_model import RandomForestModel

def main():
    base_path = "assets/chest_xray"
    output_datas = "outputs/reports"
    output_models = "outputs/trained_models"

    # Load data
    data_loader = DataLoader(base_path)
    train_images, train_labels = data_loader.load_train_images()
    val_images, val_labels = data_loader.load_val_images()
    test_generator = data_loader.load_test_images()

    # Check class distribution
    ClassDistribution.check_class_distribution(train_labels, "Training")
    ClassDistribution.detailed_class_distribution(val_labels, "Validation")

    # Ensure that the training set has examples of both classes
    if len(np.unique(train_labels)) < 2:
        print("Training set does not contain both classes. Consider augmenting the dataset.")
        return

    # Preprocess data
    preprocessing = Preprocessing()
    train_features = preprocessing.preprocess_data(train_images)
    val_features = preprocessing.preprocess_data(val_images, fit=False)

    # Grid search for hyperparameters
    best_params, best_score = HyperparameterTuning.grid_search_hyperparameters(train_features, train_labels)

    # Unpack best_params
    best_n_estimators, best_max_features, best_max_depth, best_min_samples_split, best_min_samples_leaf = best_params

    # Train the best model on full training set
    rf_model = RandomForestModel(n_estimators=best_n_estimators, random_state=42)
    rf_model.model.max_features = best_max_features
    rf_model.model.max_depth = best_max_depth
    rf_model.model.min_samples_split = best_min_samples_split
    rf_model.model.min_samples_leaf = best_min_samples_leaf
    rf_model.fit(train_features, train_labels)

    # Evaluation on validation set
    val_roc_auc = Evaluation.evaluate_model(rf_model, val_features, val_labels, "Validation", output_datas)

    # Preprocess test data
    test_features, test_labels = preprocessing.preprocess_test_data(test_generator)

    # Evaluate model on test set
    test_roc_auc = Evaluation.evaluate_model(rf_model, test_features, test_labels, "Test", output_datas)

    # Save model and results
    data_saver = DataSaver()
    data_saver.save_model(rf_model.model, f"{output_models}/random_forest_model.pkl")
    data_saver.save_pca(preprocessing.pca, f"{output_models}/pca_model.pkl")
    data_saver.save_results({
        'train_roc_auc': best_score,
        'val_roc_auc': val_roc_auc,
        'test_roc_auc': test_roc_auc
    })

if __name__ == "__main__":
    main()
