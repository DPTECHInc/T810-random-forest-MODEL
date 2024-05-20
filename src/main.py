import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from classes.data_loader import DataLoader
from classes.feature_extractor import FeatureExtractor
from classes.random_forest_model import RandomForestModel
from classes.data_saver import DataSaver

def check_class_distribution(labels, set_name):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"{set_name} set class distribution: {dict(zip(unique, counts))}")

def plot_roc_auc_scores(scores, output_path):
    plt.figure()
    plt.plot(scores, marker='o')
    plt.title('ROC-AUC Scores from Cross-Validation')
    plt.xlabel('Fold')
    plt.ylabel('ROC-AUC Score')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"ROC-AUC scores plot saved to {output_path}")

def detailed_class_distribution(labels, set_name):
    unique, counts = np.unique(labels, return_counts=True)
    class_labels = {0: "NORMAL", 1: "PNEUMONIA"}
    distribution = {class_labels[label]: count for label, count in zip(unique, counts)}
    print(f"{set_name} set detailed class distribution: {distribution}")

def plot_classification_report(report, output_path):
    classes = ['NORMAL', 'PNEUMONIA']
    metrics = ['precision', 'recall', 'f1-score']

    report_dict = {}
    for class_name in classes:
        if class_name in report:
            print(f"{class_name} report: {report[class_name]}")  # Debug: Print report details
            report_dict[class_name] = [report[class_name][metric] for metric in metrics]
        else:
            report_dict[class_name] = [0, 0, 0]  # Defaults to 0 if class is not present

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    for i, class_name in enumerate(classes):
        ax.bar(x + i * width, report_dict[class_name], width, label=class_name)

    ax.set_ylabel('Scores')
    ax.set_title('Classification report by class')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.legend()

    fig.tight_layout()
    plt.draw()  # Force the plot to be drawn
    plt.savefig(output_path)
    plt.close()
    print(f"Classification report plot saved to {output_path}")

def grid_search_hyperparameters(train_features, train_labels):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],  # Ajuster le nombre d'arbres
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None],  # Tester avec des arbres de profondeur infinie
        'min_samples_split': [2, 10],  # Régularisation
        'min_samples_leaf': [1, 4]  # Régularisation
    }

    best_score = 0
    best_params = None

    for n_estimators in param_grid['n_estimators']:
        for max_features in param_grid['max_features']:
            for max_depth in param_grid['max_depth']:
                for min_samples_split in param_grid['min_samples_split']:
                    for min_samples_leaf in param_grid['min_samples_leaf']:
                        print(f"Training with n_estimators={n_estimators}, max_features={max_features}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")

                        rf_model = RandomForestModel(n_estimators=n_estimators, random_state=42)
                        rf_model.model.max_features = max_features
                        rf_model.model.max_depth = max_depth
                        rf_model.model.min_samples_split = min_samples_split
                        rf_model.model.min_samples_leaf = min_samples_leaf

                        skf = StratifiedKFold(n_splits=5)  # Use 5 folds for cross-validation
                        scores = cross_val_score(rf_model, train_features, train_labels, cv=skf, scoring='roc_auc')

                        mean_score = np.mean(scores)
                        print(f'Cross-validation ROC-AUC scores: {scores}')
                        print(f'Mean ROC-AUC score: {mean_score}')

                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = (n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf)

    print(f"Best parameters: n_estimators={best_params[0]}, max_features={best_params[1]}, max_depth={best_params[2]}, min_samples_split={best_params[3]}, min_samples_leaf={best_params[4]}")
    print(f"Best cross-validation ROC-AUC score: {best_score}")

    return best_params, best_score

def main():
    base_path = "assets/chest_xray"

    # Load training, validation, and test images
    data_loader = DataLoader(base_path)
    train_images, train_labels = data_loader.load_train_images()
    val_images, val_labels = data_loader.load_val_images()
    test_generator = data_loader.load_test_images()

    # Check class distribution
    check_class_distribution(train_labels, "Training")
    detailed_class_distribution(val_labels, "Validation")

    # Ensure that the training set has examples of both classes
    if len(np.unique(train_labels)) < 2:
        print("Training set does not contain both classes. Consider augmenting the dataset.")
        return

    # Extract features from training images
    extractor = FeatureExtractor()
    train_features = extractor.extract_features(train_images)
    val_features = extractor.extract_features(val_images)

    # Ensure features are 2D
    train_features_reshaped = np.reshape(train_features, (train_features.shape[0], -1))
    val_features_reshaped = np.reshape(val_features, (val_features.shape[0], -1))

    # Determine the appropriate n_components for PCA
    n_components = min(train_features_reshaped.shape[0], train_features_reshaped.shape[1], 50)
    print(f"n_components set to: {n_components}")

    pca = PCA(n_components=n_components)
    train_features_reduced = pca.fit_transform(train_features_reshaped)
    val_features_reduced = pca.transform(val_features_reshaped)

    # Grid search for hyperparameters
    best_params, best_score = grid_search_hyperparameters(train_features_reduced, train_labels)

    # Unpack best_params
    best_n_estimators, best_max_features, best_max_depth, best_min_samples_split, best_min_samples_leaf = best_params

    # Train the best model on full training set
    rf_model = RandomForestModel(n_estimators=best_n_estimators, random_state=42)
    rf_model.model.max_features = best_max_features
    rf_model.model.max_depth = best_max_depth
    rf_model.model.min_samples_split = best_min_samples_split
    rf_model.model.min_samples_leaf = best_min_samples_leaf
    rf_model.fit(train_features_reduced, train_labels)

    # Evaluation on validation set
    y_pred_val = rf_model.predict(val_features_reduced)
    if len(np.unique(val_labels)) > 1:
        val_roc_auc = roc_auc_score(val_labels, y_pred_val)
        print(f'Validation ROC-AUC: {val_roc_auc}')
    else:
        print("Cannot compute ROC AUC - only one class present in val_labels")
        val_roc_auc = None

    val_conf_matrix = confusion_matrix(val_labels, y_pred_val, labels=[0, 1])
    print(val_conf_matrix)
    val_class_report = classification_report(val_labels, y_pred_val, labels=[0, 1], zero_division=0, output_dict=True)
    print(classification_report(val_labels, y_pred_val, labels=[0, 1], zero_division=0))

    # Save the classification report plot for the validation set
    print("Generating classification report plot for validation set...")
    plot_classification_report(val_class_report, "outputs/reports/validation_classification_report.png")

    # Load and preprocess test images
    test_features_reduced = []
    test_labels = []

    # Iterate over the test generator
    for i in range(len(test_generator)):
        batch_images, batch_labels = test_generator[i]
        batch_features = extractor.extract_features(batch_images)
        batch_features_reshaped = np.reshape(batch_features, (batch_features.shape[0], -1))
        batch_features_reduced = pca.transform(batch_features_reshaped)

        test_features_reduced.append(batch_features_reduced)
        test_labels.append(batch_labels)

    # Concatenate all batches
    test_features_reduced = np.vstack(test_features_reduced)
    test_labels = np.hstack(test_labels)
    print(f"Shape of test_features_reduced: {test_features_reduced.shape}")
    print(f"Shape of test_labels: {test_labels.shape}")

    detailed_class_distribution(test_labels, "Test")

    # Save the test data
    data_saver = DataSaver()
    data_saver.save_datas(test_features_reduced, test_labels)

    # Evaluate the model on test data
    y_pred_test = rf_model.predict(test_features_reduced)
    accuracy = rf_model.evaluate([(test_features_reduced, test_labels)])

    if len(np.unique(test_labels)) > 1:
        test_roc_auc = roc_auc_score(test_labels, y_pred_test)
        print(f'Test ROC-AUC: {test_roc_auc}')
    else:
        print("Cannot compute ROC AUC - only one class present in test_labels")
        test_roc_auc = None

    test_conf_matrix = confusion_matrix(test_labels, y_pred_test, labels=[0, 1])
    print(test_conf_matrix)
    test_class_report = classification_report(test_labels, y_pred_test, labels=[0, 1], zero_division=0, output_dict=True)
    print(classification_report(test_labels, y_pred_test, labels=[0, 1], zero_division=0))

    # Save the classification report plot for the test set
    print("Generating classification report plot for test set...")
    plot_classification_report(test_class_report, "outputs/reports/test_classification_report.png")

    # Save the trained model and results
    rf_model.save_model()
    data_saver.save_results({
        'train_roc_auc': best_score,
        'val_roc_auc': val_roc_auc,
        'test_roc_auc': test_roc_auc
    })

if __name__ == "__main__":
    main()
