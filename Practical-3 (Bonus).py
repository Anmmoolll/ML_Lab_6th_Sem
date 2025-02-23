import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, precision_recall_curve, roc_curve, mean_absolute_error, 
    mean_squared_error
)



# Load dataset
df = pd.read_csv("customer_churn.csv")

# Encode categorical variables
label_encoders = {}
categorical_columns = ["Gender", "Contract", "PaymentMethod"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for future use

# Feature Scaling
scaler = StandardScaler()
df[['Tenure', 'MonthlyCharges', 'TotalCharges', 'SupportTickets']] = scaler.fit_transform(df[['Tenure', 'MonthlyCharges', 'TotalCharges', 'SupportTickets']])

# Define features (X) and target variable (y)
X = df.drop(columns=["CustomerID", "Churn"])  # Remove non-relevant columns
y = df["Churn"]

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred_rf = clf.predict(X_test)
y_prob_rf = clf.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

y_pred_lr = log_reg.predict(X_test)
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]



# Function to print evaluation metrics
def print_metrics(model_name, y_test, y_pred, y_prob):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC-AUC Score: {roc_auc:.2f}")

# Print metrics for Random Forest and Logistic Regression
print_metrics("Random Forest", y_test, y_pred_rf, y_prob_rf)
print_metrics("Logistic Regression", y_test, y_pred_lr, y_prob_lr)



# Precision-Recall Curve
def plot_precision_recall(y_test, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, marker='.', label=model_name)
    
plt.figure(figsize=(8, 6))
plot_precision_recall(y_test, y_prob_rf, "Random Forest")
plot_precision_recall(y_test, y_prob_lr, "Logistic Regression")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# ROC Curve
def plot_roc_curve(y_test, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, marker='.', label=model_name)
    
plt.figure(figsize=(8, 6))
plot_roc_curve(y_test, y_prob_rf, "Random Forest")
plot_roc_curve(y_test, y_prob_lr, "Logistic Regression")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()



# Train Regression Model (Using Logistic Regression Probabilities)
y_train_prob = log_reg.predict_proba(X_train)[:, 1]
y_test_prob = log_reg.predict_proba(X_test)[:, 1]

# Compute Regression Metrics
mae = mean_absolute_error(y_test, y_test_prob)
mse = mean_squared_error(y_test, y_test_prob)
rmse = np.sqrt(mse)

print(f"\nRegression Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")



# Residual Plot
plt.figure(figsize=(8, 6))
residuals = y_test - y_test_prob
sns.histplot(residuals, bins=20, kde=True, color='blue')
plt.xlabel("Residuals (Actual - Predicted)")
plt.title("Residual Distribution for Churn Probability")
plt.show()



from sklearn.feature_selection import SelectKBest, f_classif

# Apply feature selection
selector = SelectKBest(score_func=f_classif, k=5)  # Select top 5 features
X_selected = selector.fit_transform(X, y)

# Split the dataset again
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train the model again with selected features
clf_fs = RandomForestClassifier(n_estimators=100, random_state=42)
clf_fs.fit(X_train_fs, y_train_fs)

# Predict and evaluate
y_pred_fs = clf_fs.predict(X_test_fs)
accuracy_fs = accuracy_score(y_test_fs, y_pred_fs)

print(f"\nModel Accuracy with Feature Selection: {accuracy_fs:.2f}")
