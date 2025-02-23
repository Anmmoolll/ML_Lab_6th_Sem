import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error



# Load the dataset
df = pd.read_csv("house_prices.csv")

# Select features and target variable
X = df[['SquareFootage', 'NumRooms', 'HouseAge']]
y = df['Price']

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables for KNN (important for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# List of k values to experiment with
k_values = [3, 5, 7, 10, 15]
mae_list, mse_list, rmse_list = [], [], []

for k in k_values:
    # Initialize and train KNN model
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_knn = knn.predict(X_test_scaled)
    
    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred_knn)
    mse = mean_squared_error(y_test, y_pred_knn)
    rmse = np.sqrt(mse)
    
    # Store results
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    
    print(f"\nKNN Regression (k={k}):")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")



# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr.predict(X_test)

# Compute evaluation metrics
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

print("\nLinear Regression Performance:")
print(f"MAE: {mae_lr:.2f}")
print(f"MSE: {mse_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")



plt.figure(figsize=(8, 5))
plt.plot(k_values, mae_list, marker='o', label='MAE', linestyle='dashed')
plt.plot(k_values, mse_list, marker='s', label='MSE', linestyle='dashed')
plt.plot(k_values, rmse_list, marker='^', label='RMSE', linestyle='dashed')

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Error")
plt.title("KNN Regression Performance for Different k Values")
plt.legend()
plt.grid()
plt.show()
