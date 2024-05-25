import numpy as np
from classes.data_loader import DataLoader
from classes.data_saver import DataSaver
from classes.class_distribution import ClassDistribution
from classes.preprocessing import Preprocessing
from classes.hyperparameter_tuning import HyperparameterTuning
from classes.evaluation import Evaluation
from classes.random_forest_model import RandomForestModel
from sklearn.model_selection import train_test_split, cross_val_score
import json

def main():
    base_path = "assets/chest_xray"
    output_datas = "outputs/reports"
    output_models = "outputs/trained_models"
    class_names = ['NORMAL', 'BACTERIA', 'VIRUS']

    # Load data
    data_loader = DataLoader(base_path)
    train_images, train_labels = data_loader.load_train_images()
    val_images, val_labels = data_loader.load_val_images()
    test_images, test_labels = data_loader.load_test_images()

    # Check class distribution
    ClassDistribution.check_class_distribution(np.argmax(train_labels, axis=1), "Training")
    ClassDistribution.detailed_class_distribution(np.argmax(val_labels, axis=1), "Validation")

    # Ensure that the training set has examples of all classes
    unique_classes = np.unique(np.argmax(train_labels, axis=1))
    if len(unique_classes) < len(class_names):
        print("Training set does not contain all classes. Consider augmenting the dataset.")
        return

    # Preprocess data
    preprocessing = Preprocessing()
    train_features = preprocessing.preprocess_data(train_images)
    val_features = preprocessing.preprocess_data(val_images, fit=False)
    test_features = preprocessing.preprocess_data(test_images, fit=False)

    # Convert labels from one-hot to single integer format
    train_labels_single = np.argmax(train_labels, axis=1)
    val_labels_single = np.argmax(val_labels, axis=1)
    test_labels_single = np.argmax(test_labels, axis=1)

    # Split training data for cross-validation
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels_single, test_size=0.2, random_state=42)

    # Grid search for hyperparameters with cross-validation
    best_params, best_score = HyperparameterTuning.grid_search_hyperparameters(X_train, y_train)

    # Train the best model on full training set
    rf_model = RandomForestModel(
        n_estimators=best_params['n_estimators'],
        max_features=best_params['max_features'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )
    rf_model.fit(train_features, train_labels_single)

    # Evaluation on validation set using cross-validation
    val_roc_auc_cv = cross_val_score(rf_model.model, X_val, y_val, cv=5, scoring='roc_auc_ovr').mean()

    # Evaluation on validation set (simple split)
    val_roc_auc = Evaluation.evaluate_model(rf_model, val_features, val_labels_single, "Validation", output_datas, class_names)

    # Evaluate model on test set
    test_roc_auc = Evaluation.evaluate_model(rf_model, test_features, test_labels_single, "Test", output_datas, class_names)

    # Save model and results
    data_saver = DataSaver()
    data_saver.save_model(rf_model.model, f"{output_models}/random_forest_model.pkl")
    data_saver.save_pca(preprocessing.pca, f"{output_models}/pca_model.pkl")
    data_saver.save_results({
        'train_roc_auc': best_score,
        'val_roc_auc_cv': val_roc_auc_cv,
        'val_roc_auc': val_roc_auc,
        'test_roc_auc': test_roc_auc
    })

    # Save additional technical details for documentation
    technical_details = {
        'best_hyperparameters': best_params,
        'cross_validation_scores': val_roc_auc_cv,
        'validation_scores': val_roc_auc,
        'test_scores': test_roc_auc
    }
    with open(f"{output_datas}/technical_details.json", 'w') as f:
        json.dump(technical_details, f, indent=4)

if __name__ == "__main__":
    main()
