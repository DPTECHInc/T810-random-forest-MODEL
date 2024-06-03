import os
import numpy as np
from classes.data_loader import DataLoader
from classes.data_saver import DataSaver
from classes.class_distribution import ClassDistribution
from classes.preprocessing import Preprocessing
from classes.hyperparameter_tuning import HyperparameterTuning
from classes.evaluation import Evaluation
from classes.random_forest_model import RandomForestModel
from classes.notebook_generator import NotebookGenerator
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import json


def main():
    base_path = "assets/chest_xray"
    output_datas = "outputs/reports"
    output_models = "outputs/trained_models"
    model_path = f"{output_models}/random_forest_model.pkl"
    pca_path = f"{output_models}/pca_model.pkl"
    class_names = ['NORMAL', 'BACTERIA', 'VIRUS']

    # Generate unique run identifier
    run_number = len([name for name in os.listdir(output_datas) if os.path.isdir(os.path.join(output_datas, name))]) + 1
    run_folder = os.path.join(output_datas, f"run_{run_number}")
    os.makedirs(run_folder, exist_ok=True)

    # Load data
    data_loader = DataLoader(base_path)
    train_images, train_labels = data_loader.load_train_images()
    val_images, val_labels = data_loader.load_val_images()
    test_images, test_labels = data_loader.load_test_images()

    # Check class distribution
    ClassDistribution.detailed_class_distribution(np.argmax(train_labels, axis=1), "Training", run_number)
    ClassDistribution.detailed_class_distribution(np.argmax(val_labels, axis=1), "Validation", run_number)
    ClassDistribution.detailed_class_distribution(np.argmax(test_labels, axis=1), "Test", run_number)

    # Ensure that the training set has examples of all classes
    unique_classes = np.unique(np.argmax(train_labels, axis=1))
    if len(unique_classes) < len(class_names):
        print("Training set does not contain all classes. Consider augmenting the dataset.")
        return

    # Preprocess data with oversampling
    preprocessing = Preprocessing()
    train_features, train_labels = preprocessing.preprocess_data(train_images, train_labels, fit=True,
                                                                 balance_method='oversample')
    val_features, val_labels = preprocessing.preprocess_data(val_images, val_labels, fit=False)
    test_features = preprocessing.preprocess_test_data(test_images)

    # Convert labels from one-hot to single integer format
    train_labels_single = np.argmax(train_labels, axis=1)
    val_labels_single = np.argmax(val_labels, axis=1)
    test_labels_single = np.argmax(test_labels, axis=1)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5)

    best_params = None
    best_score = None

    # Load or train the model
    rf_model = RandomForestModel()
    if os.path.exists(model_path) and os.path.exists(pca_path):
        # Load the model and PCA
        rf_model.load(model_path)
        preprocessing.pca = joblib.load(pca_path)
        best_params = rf_model.get_params()
        print("Loaded existing model and PCA.")
    else:
        # Hyperparameter tuning with RandomizedSearchCV
        best_params, best_score = HyperparameterTuning.randomized_search_hyperparameters(train_features,
                                                                                         train_labels_single, cv=skf)

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

        # Save the trained model and PCA
        rf_model.save(model_path)
        joblib.dump(preprocessing.pca, pca_path)
        print("Trained and saved new model and PCA.")

    # Evaluation on validation set using stratified cross-validation
    val_roc_auc_cv = cross_val_score(rf_model.model, val_features, val_labels_single, cv=skf, scoring='roc_auc_ovr').mean()

    # Collect evaluation results
    train_y_true, train_y_score = train_labels_single, rf_model.predict_proba(train_features)
    val_y_true, val_y_score = val_labels_single, rf_model.predict_proba(val_features)
    test_y_true, test_y_score = test_labels_single, rf_model.predict_proba(test_features)

    # Evaluation on validation set (simple split)
    val_roc_auc = Evaluation.evaluate_model(rf_model.model, val_features, val_labels_single, "Validation", run_folder,
                                            class_names)

    # Evaluate model on test set
    test_roc_auc = Evaluation.evaluate_model(rf_model.model, test_features, test_labels_single, "Test", run_folder,
                                             class_names)

    # Evaluate model on training set
    train_roc_auc = Evaluation.evaluate_model(rf_model.model, train_features, train_labels_single, "Training", run_folder,
                                              class_names)

    # Save model and results
    data_saver = DataSaver()
    data_saver.save_results({
        'train_roc_auc': train_roc_auc,
        'val_roc_auc_cv': val_roc_auc_cv,
        'val_roc_auc': val_roc_auc,
        'test_roc_auc': test_roc_auc
    })

    # Save additional technical details for documentation
    technical_details = {
        'best_hyperparameters': best_params if best_params else "Loaded model, no hyperparameter tuning performed",
        'cross_validation_scores': val_roc_auc_cv,
        'validation_scores': val_roc_auc,
        'test_scores': test_roc_auc
    }
    with open(f"{run_folder}/technical_details.json", 'w') as f:
        json.dump(technical_details, f, indent=4)

    # Update reports.json
    reports_path = os.path.join(output_datas, "reports.json")
    if os.path.exists(reports_path):
        with open(reports_path, 'r') as f:
            reports = json.load(f)
    else:
        reports = {}

    reports[f"run_{run_number}"] = {
        'train_roc_auc': train_roc_auc,
        'val_roc_auc': val_roc_auc,
        'test_roc_auc': test_roc_auc
    }

    with open(reports_path, 'w') as f:
        json.dump(reports, f, indent=4)

    # Create and save the notebook
    NotebookGenerator.create_notebook(run_number, best_params if best_params else {}, {
        'train_roc_auc': train_roc_auc,
        'val_roc_auc_cv': val_roc_auc_cv,
        'val_roc_auc': val_roc_auc,
        'test_roc_auc': test_roc_auc
    }, run_folder)


if __name__ == "__main__":
    main()
