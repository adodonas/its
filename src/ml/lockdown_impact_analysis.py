import sys
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.utils import class_weight
import logging


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
    gbm = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42, subsample=0.8)

    # A list to store the accuracy of each fold
    scores = []

    # Perform cross-validation
    for train_index, test_index in skf.split(X, y):
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model on the training data
        gbm.fit(X_train, y_train, sample_weight=class_weights[train_index])

        # Predict on the test data
        y_pred = gbm.predict(X_test)

        # Compute accuracy and F1-score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log the accuracy and F1-score
        logging.info(f"Accuracy in this fold: {accuracy}")
        logging.info(f"F1 score in this fold: {f1}")

        # Separate the test set into lockdown and non-lockdown periods
        X_test_lockdown = X_test[X_test['lockdown'] == 1]
        y_test_lockdown = y_test[X_test['lockdown'] == 1]
        X_test_non_lockdown = X_test[X_test['lockdown'] == 0]
        y_test_non_lockdown = y_test[X_test['lockdown'] == 0]

        # Evaluate the model performance during lockdown and non-lockdown periods
        evaluate_period(gbm, X_test_lockdown, y_test_lockdown, 'lockdown')
        evaluate_period(gbm, X_test_non_lockdown, y_test_non_lockdown, 'non-lockdown')

        # Store the accuracy of this fold
        scores.append(accuracy)

    # Return the list of scores and the trained model
    return scores, gbm


def main():
    configure_logging()
    X, y = load_data()
    scores, gbm = cross_validate(X, y)
    average_score = sum(scores) / len(scores)
    logging.info(f"The average accuracy across all folds is {average_score}")
    # Plot the results for the lockdown period
    X_lockdown = X[X['lockdown'] == 1]
    y_lockdown = y[X['lockdown'] == 1]
    plot_results(gbm, X_lockdown, y_lockdown, 'lockdown')

    # Plot the results for the non-lockdown period
    X_non_lockdown = X[X['lockdown'] == 0]
    y_non_lockdown = y[X['lockdown'] == 0]
    plot_results(gbm, X_non_lockdown, y_non_lockdown, 'non-lockdown')


if __name__ == "__main__":
    main()
