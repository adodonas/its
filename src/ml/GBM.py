import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_excel('../feature_engineering/results/final_normal_merged.xlsx')

# The target column 'NO' is turned into a binary variable based on whether
# its value is above or below the median. This transformation turns the original
# regression task into a binary classification task, which is a requirement
# for using the GradientBoostingClassifier.
df['NO'] = [1 if i > df['NO'].median() else 0 for i in df['NO']]

# Split the data into features (X) and target (y)
X = df[['NO2', 'NOX', 'SO2', 'rh', 'wd', 'ws', 'area', 'day_of_week']]
y = df['NO']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gradient Boosting Classifier
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42)

# Fit the model
gbm.fit(X_train, y_train)

# Predict on the test set
y_pred = gbm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"The accuracy of the model is {accuracy}")
print(f"The F1 score of the model is {f1}")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(conf_mat, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curve
y_pred_proba = gbm.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = auc(fpr, tpr)

plt.figure(figsize=(10,7))
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
