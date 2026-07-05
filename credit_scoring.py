# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

print("All libraries imported successfully!")

# Load Dataset
df = pd.read_csv("credit_risk_dataset.csv")

# Show first 5 rows
print("\nFirst 5 Rows of Dataset:")
print(df.head())
# Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)
# Remove Missing Values
df = df.dropna()

# Convert Text Columns into Numbers
encoder = LabelEncoder()

for column in df.select_dtypes(include="object").columns:
    df[column] = encoder.fit_transform(df[column])

print("\nData after preprocessing:")
print(df.head())
# Select Features (X) and Target (y)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())
# Split the dataset into Training and Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Create Random Forest Model
model = RandomForestClassifier(random_state=42)

# Train the Model
model.fit(X_train, y_train)

print("\nModel trained successfully!")
# Make Predictions
y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)
# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)