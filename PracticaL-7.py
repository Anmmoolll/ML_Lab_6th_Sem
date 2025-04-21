import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset from a local file, using the first row as headers
data = pd.read_csv('diabetes.csv', header=0)  # or header='infer' if header is present

# Convert 'Pregnancies' column to numeric, handling errors (if needed)
data['Pregnancies'] = pd.to_numeric(data['Pregnancies'], errors='coerce')

# Split the data into features (X) and target (y)
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (normalize them)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Initialize and train the RandomForestClassifier (Bagging method)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
rf_preds = rf_model.predict(X_test)

# Evaluate performance
rf_accuracy = accuracy_score(y_test, rf_preds)



# Initialize and train the AdaBoostClassifier (Boosting method)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)

# Make predictions on the test set
ada_preds = ada_model.predict(X_test)

# Evaluate performance
ada_accuracy = accuracy_score(y_test, ada_preds)



# Initialize and train the GradientBoostingClassifier (Boosting method)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions on the test set
gb_preds = gb_model.predict(X_test)

# Evaluate performance
gb_accuracy = accuracy_score(y_test, gb_preds)



# Initialize and train the XGBoostClassifier (Boosting method)
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
xgb_preds = xgb_model.predict(X_test)

# Evaluate performance
xgb_accuracy = accuracy_score(y_test, xgb_preds)



# Print Accuracy for each model
print("Random Forest Accuracy:", rf_accuracy)
print("AdaBoost Accuracy:", ada_accuracy)
print("Gradient Boosting Accuracy:", gb_accuracy)
print("XGBoost Accuracy:", xgb_accuracy)



# List of models and their corresponding accuracies
models = ['Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost']
accuracies = [rf_accuracy, ada_accuracy, gb_accuracy, xgb_accuracy]

# Plot the accuracies
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.title("Model Comparison: Accuracy of Different Classifiers", fontsize=16)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.ylim([0, 1])  # To ensure the accuracy scale is from 0 to 1
plt.show()
