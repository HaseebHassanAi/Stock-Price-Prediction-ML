# -----------------------------
# Stock Price Prediction
# -----------------------------
# This script predicts the next day's closing price of a stock
# using multiple regression models: Linear Regression, Decision Tree, Random Forest, and KNN.
# It includes data cleaning, outlier removal, feature scaling, model training, and next-day prediction.
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv('/content/drive/MyDrive/Stock_Price_dataset.csv')

# Basic inspection of the data
print(df.head())
print(df.info())
print(df.describe())

# -----------------------------
# Create target variable for next-day Close price
# -----------------------------
df['target'] = df['Close'].shift(-1)  # Shift Close price by -1 to predict next day
df.dropna(inplace=True)  # Remove the last row with NaN target

# -----------------------------
# Drop unnecessary columns
# -----------------------------
df = df.drop(['Date', 'Adj Close'], axis=1)
df = df.drop_duplicates()  # Remove duplicate rows
print(df.head())

# -----------------------------
# Visualize features before outlier removal
# -----------------------------
features = ['Open', 'High', 'Low', 'Close', 'Volume']

for col in features:
    plt.figure(figsize=(6, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()

for col in features:
    plt.figure(figsize=(6, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# -----------------------------
# Remove outliers using IQR method
# -----------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]  # Keep rows within bounds

# -----------------------------
# Visualize features after outlier removal
# -----------------------------
for col in features:
    plt.figure(figsize=(6, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'After Outlier Removal Histogram of {col}')
    plt.show()

for col in features:
    plt.figure(figsize=(6, 6))
    sns.boxplot(x=df[col])
    plt.title(f'After Outlier Removal Boxplot of {col}')
    plt.show()

# -----------------------------
# Prepare features and target
# -----------------------------
X = df[features]  # Feature set
y = df['target']  # Target variable

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Initialize models
# -----------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'KNN': KNeighborsRegressor()
}

# -----------------------------
# Train models and evaluate performance
# -----------------------------
for name, mdl in models.items():
    # Train model
    mdl.fit(X_train, y_train)
    
    # Training performance
    y_train_pred = mdl.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    print(f"{name} Training R2: {r2_train:.4f}")
    
    # Testing performance
    y_test_pred = mdl.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"{name} Testing R2: {r2_test:.4f}")

# -----------------------------
# Predict next day Close using last available day features
# -----------------------------
last_day_features = df[features].iloc[-1].values.reshape(1, -1)
last_day_scaled = scaler.transform(last_day_features)

print("\nLast Day Features:")
print(last_day_features)

print("\nNext Day Close Predictions:")
for name, mdl in models.items():
    next_close = mdl.predict(last_day_scaled)

    print(f"{name}: {next_close[0]:.2f}")
