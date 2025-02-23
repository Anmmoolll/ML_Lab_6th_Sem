import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (Replace 'titanic.csv' with 'iris.csv' if using the Iris dataset)
df = pd.read_csv("test_data.csv")

# 1. Data Cleaning: Handling Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing numerical values with mean
df.dropna(inplace=True)  # Drop remaining missing values (categorical)

# 2. Exploratory Data Analysis (EDA)
print("\nSummary Statistics:\n", df.describe())

# Correlation Matrix
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Distribution of numerical columns
df.hist(figsize=(10,8), bins=20)
plt.suptitle("Distribution of Numerical Features")
plt.show()

# 3. Export Cleaned Data to CSV
df.to_csv("cleaned_test_data.csv", index=False)  # Change file name if using Iris dataset

print("Cleaned data has been exported successfully!")
