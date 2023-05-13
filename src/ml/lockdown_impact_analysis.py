import logging
import sys

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight


def configure_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('lockdown_impact_analysis.log', 'w'),
                                  logging.StreamHandler(sys.stdout)])


def load_data():
    """
    Load and preprocess the dataset.

    Returns:
    X (DataFrame): The input features.
    y (Series): The target variable.
    """

    # Load the dataset from an Excel file
    df = pd.read_excel('../feature_engineering/results/final_normal_merged.xlsx')

    # Convert the target 'NO' into a binary variable, based on whether its value is above or below the median
    df['NO'] = [1 if i > df['NO'].median() else 0 for i in df['NO']]

    # Select the input features and the target variable
    X = df[['NO2', 'NOX', 'SO2', 'rh', 'wd', 'ws', 'area', 'day_of_week', 'lockdown']]
    y = df['NO']

    return X, y


def evaluate_period(gbm, X_test_period, y_test_period, period_name):
    """
    Evaluate the model performance during a specific period (e.g., lockdown or non-lockdown).

    Parameters:
    gbm (GradientBoostingClassifier): The trained model.
    X_test_period (DataFrame): The input features for the test set during a specific period.
    y_test_period (Series): The target variable for the test set during a specific period.
    period_name (str): The name of the period (e.g., 'lockdown', 'non-lockdown').

    """

    # Predict on the test set for the specific period
    y_pred_period = gbm.predict(X_test_period)

    # Compute accuracy and F1-score for the specific period
    accuracy_period = accuracy_score(y_test_period, y_pred_period)
    f1_period = f1_score(y_test_period, y_pred_period)

    # Log the accuracy and F1-score for the specific period
    logging.info(f"Accuracy during {period_name} period: {accuracy_period}")
    logging.info(f"F1 score during {period_name} period: {f1_period}")


def plot_results(gbm, X, y, period_name):
    """
    Plot the ROC AUC curve and the impact of lockdowns on air quality.

    Args:
    gbm (GradientBoostingClassifier): The trained model.
    X (DataFrame): The input features.
    y (Series): The target variable.
    period_name (str): The name of the period ('lockdown' or 'non-lockdown').
    """

    # Calculate the predicted probabilities
    y_pred_proba = gbm.predict_proba(X)[:, 1]

    # Calculate the false positive rate, true positive rate, and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    # Calculate the AUC
    auc_score = auc(fpr, tpr)

    # Create a new figure
    plt.figure(figsize=(10, 7))

    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f"ROC AUC ({period_name}) = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')  # Add a diagonal line for reference
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc='lower right')

    # Show the plot
    plt.show()


def cross_validate(X, y):
    """
    Perform cross-validation on the provided data.

    Parameters:
    X (DataFrame): The input features.
    y (Series): The target variable.

    Returns:
    scores (List): A list of accuracy scores for each fold.
    gbm (GradientBoostingClassifier): The trained model.
    """

    # Compute class weights to handle class imbalance
    class_weights = class_weight.compute_sample_weight('balanced', y)
    # Define cross-validation strategy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Instantiate the Gradient Boosting Classifier
    # Decrease max_depth, increase learning_rate, and set subsample < 1 for regularization
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=42, subsample=0.8)

    # Lists to store the accuracy, precision, recall and F1-score of each fold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    auc_scores = []

    # Perform cross-validation
    for train_index, test_index in skf.split(X, y):
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model on the training data
        gbm.fit(X_train, y_train, sample_weight=class_weights[train_index])

        # Predict on the test data
        y_pred = gbm.predict(X_test)
        y_pred_proba = gbm.predict_proba(X_test)[:, 1]  # For AUC

        # Compute accuracy, precision, recall, F1-score and AUC
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)

        # Log the scores
        logging.info(f"Accuracy in this fold: {accuracy}")
        logging.info(f"Precision in this fold: {precision}")
        logging.info(f"Recall in this fold: {recall}")
        logging.info(f"F1 score in this fold: {f1}")
        logging.info(f"AUC in this fold: {auc_score}")

        # Store the scores of this fold
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc_score)

    # Return the lists of scores and the trained model
    return accuracies, precisions, recalls, f1_scores, auc_scores, gbm


def save_to_excel(accuracies, precisions, recalls, f1_scores, auc_scores, period_name):
    """
    Save model performance metrics to an Excel file.

    Args:
    accuracies (List): List of accuracy scores.
    precisions (List): List of precision scores.
    recalls (List): List of recall scores.
    f1_scores (List): List of F1 scores.
    auc_scores (List): List of AUC scores.
    period_name (str): The name of the period ('lockdown' or 'non-lockdown').
    """

    # Create a DataFrame with the performance metrics
    df = pd.DataFrame({
        'Fold': range(1, len(accuracies) + 1),
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores,
        'AUC': auc_scores
    })

    # Set 'Fold' as the index of the DataFrame
    df.set_index('Fold', inplace=True)

    # Save the DataFrame to an Excel file
    df.to_excel(f'{period_name}_model_performance.xlsx')


def main():
    configure_logging()
    X, y = load_data()

    # Split the data into lockdown and non-lockdown periods
    X_lockdown = X[X['lockdown'] == 1]
    y_lockdown = y[X['lockdown'] == 1]
    X_non_lockdown = X[X['lockdown'] == 0]
    y_non_lockdown = y[X['lockdown'] == 0]

    # Train a model on the lockdown period
    accuracies, precisions, recalls, f1_scores, auc_scores, gbm_lockdown = cross_validate(X_lockdown, y_lockdown)
    save_to_excel(accuracies, precisions, recalls, f1_scores, auc_scores, 'lockdown')
    average_accuracy = sum(accuracies) / len(accuracies)
    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)
    average_f1_score = sum(f1_scores) / len(f1_scores)
    average_auc_score = sum(auc_scores) / len(auc_scores)

    logging.info(f"The average accuracy across all folds during lockdown is {average_accuracy}")
    logging.info(f"The average precision across all folds during lockdown is {average_precision}")
    logging.info(f"The average recall across all folds during lockdown is {average_recall}")
    logging.info(f"The average F1 score across all folds during lockdown is {average_f1_score}")
    logging.info(f"The average AUC across all folds during lockdown is {average_auc_score}")

    # Plot the results for the lockdown period
    # plot_results(gbm_lockdown, X_lockdown, y_lockdown, 'lockdown')

    # Train a model on the non-lockdown period
    accuracies, precisions, recalls, f1_scores, auc_scores, gbm_non_lockdown = cross_validate(X_non_lockdown,
                                                                                              y_non_lockdown)
    save_to_excel(accuracies, precisions, recalls, f1_scores, auc_scores, 'non-lockdown')
    average_accuracy = sum(accuracies) / len(accuracies)
    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)
    average_f1_score = sum(f1_scores) / len(f1_scores)
    average_auc_score = sum(auc_scores) / len(auc_scores)

    logging.info(f"The average accuracy across all folds during non-lockdown is {average_accuracy}")
    logging.info(f"The average precision across all folds during non-lockdown is {average_precision}")
    logging.info(f"The average recall across all folds during non-lockdown is {average_recall}")
    logging.info(f"The average F1 score across all folds during non-lockdown is {average_f1_score}")
    logging.info(f"The average AUC across all folds during non-lockdown is {average_auc_score}")

    # Plot the results for the non-lockdown period
    # plot_results(gbm_non_lockdown, X_non_lockdown, y_non_lockdown, 'non-lockdown')


if __name__ == "__main__":
    main()
