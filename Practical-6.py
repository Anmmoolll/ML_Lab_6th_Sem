import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generating features
num_visits = np.random.randint(1, 50, size=n_samples)  # Number of visits (1-50)
time_spent = np.random.uniform(1, 60, size=n_samples)  # Time spent on website in minutes (1-60)

# Age group: Randomly select between 5 different groups
age_group = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], size=n_samples)

# Product categories viewed: Randomly select from 5 categories
product_categories_viewed = np.random.choice(['Electronics', 'Fashion', 'Home & Kitchen', 'Beauty', 'Sports'], size=n_samples)

# Encode age group and product categories as dummy variables
age_group_encoded = pd.get_dummies(age_group, prefix='Age', drop_first=True)  # Drop the first category to avoid multicollinearity
product_categories_encoded = pd.get_dummies(product_categories_viewed, prefix='Product', drop_first=True)

# Now generate a basic probability of purchase based on some logic:
purchase_probability = (num_visits * 0.1 + time_spent * 0.05)

# Add some influence from categorical variables
purchase_probability += age_group_encoded['Age_25-34'] * 1.0  # Customers in the 25-34 age group have a higher chance
purchase_probability += product_categories_encoded['Product_Electronics'] * 1.5  # Electronics viewers have higher chance of purchase

# Normalize the purchase probability between 0 and 1
purchase_probability = np.clip(purchase_probability / purchase_probability.max(), 0, 1)

# Generate the target variable (Purchased: 1 = Yes, 0 = No) using a binomial distribution
purchased = np.random.binomial(1, purchase_probability)

# Create DataFrame
df = pd.DataFrame({
    'Num_Visits': num_visits,
    'Time_Spent': time_spent,
    'Purchased': purchased
})

# Append the encoded age group and product category variables
df = pd.concat([df, age_group_encoded, product_categories_encoded], axis=1)

# Display the first few rows
df.head()



# Save the dataset to a CSV file
df.to_csv('customer_browsing_data.csv', index=False)




df = pd.read_csv('customer_browsing_data.csv')





# Import required libraries
import pandas as pd

# Load the dataset
df = pd.read_csv('customer_browsing_data.csv')

# Display the first few rows
df.head()





from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df.drop('Purchased', axis=1)  # Drop the target column
y = df['Purchased']  # Target variable (whether purchase was made)

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the datasets
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")





from sklearn.tree import DecisionTreeClassifier

# Initialize the model
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
dt_classifier.fit(X_train, y_train)





from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate classification report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()





# Initialize the model with smaller tree settings
dt_classifier = DecisionTreeClassifier(
    random_state=42,
    max_depth=3,  # Limit depth to make the tree smaller
    min_samples_leaf=10,  # Each leaf must have at least 10 samples
    min_samples_split=20  # Require at least 20 samples to split a node
)

# Train the model
dt_classifier.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(10, 8))  # Make sure the figure size is reasonable for a smaller tree
plot_tree(dt_classifier, 
          feature_names=X.columns, 
          class_names=['No Purchase', 'Purchase'], 
          filled=True, 
          fontsize=10)  # Keep font size reasonable for readability

plt.title("Smaller Decision Tree Visualization")
plt.show()