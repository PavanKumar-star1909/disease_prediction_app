# scripts/preprocess_train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ------------------ Create models folder inside project ------------------
os.makedirs('models', exist_ok=True)

# ------------------ Preprocessing Functions ------------------

def preprocess_liver(df):
    df = df.copy()
    df.dropna(inplace=True)
    # Encode Gender column
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    X = df.drop('Dataset', axis=1)
    y = df['Dataset']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Save scaler for later use in app
    joblib.dump(scaler, 'models/liver_scaler.pkl')
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_kidney(df):
    df = df.copy()
    df.replace('?', np.nan, inplace=True)

    # Drop 'id' column if it exists (not predictive)
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    # Fill numeric missing values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill categorical missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Encode all categorical columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Determine target column
    if 'classification' in df.columns:
        target_col = 'classification'
    elif 'class' in df.columns:
        target_col = 'class'
    else:
        raise ValueError("Kidney dataset target column not found!")

    df[target_col] = LabelEncoder().fit_transform(df[target_col])
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Save scaler for later use in app
    joblib.dump(scaler, 'models/kidney_scaler.pkl')
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_parkinsons(df):
    df = df.copy()
    
    # Drop non-numeric columns (like name/ID)
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    df.drop(columns=non_numeric_cols, inplace=True)
    
    # Target column
    if 'status' not in df.columns:
        raise ValueError("Parkinson's dataset target column 'status' not found!")
    X = df.drop('status', axis=1)
    y = df['status']
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Save scaler for later use in app
    joblib.dump(scaler, 'models/parkinsons_scaler.pkl')
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Train and Save Models ------------------

def train_and_save_model(X_train, X_test, y_train, y_test, filename):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {filename} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    # Save model inside project folder
    joblib.dump(model, f'models/{filename}')

# ------------------ Main Script ------------------

if __name__ == "__main__":
    # Load datasets
    liver_df = pd.read_csv(r'C:\Users\vishv\Documents\Multiple_Disease_Prediction\data\indian_liver_patient.csv')
    kidney_df = pd.read_csv(r'C:\Users\vishv\Documents\Multiple_Disease_Prediction\data\kidney_disease.csv')
    parkinsons_df = pd.read_csv(r'C:\Users\vishv\Documents\Multiple_Disease_Prediction\data\parkinsons.csv')

    # Train Liver Disease Model
    print("Training Liver Disease Model...")
    X_train, X_test, y_train, y_test = preprocess_liver(liver_df)
    train_and_save_model(X_train, X_test, y_train, y_test, 'liver_model.pkl')

    # Train Kidney Disease Model
    print("Training Kidney Disease Model...")
    X_train, X_test, y_train, y_test = preprocess_kidney(kidney_df)
    train_and_save_model(X_train, X_test, y_train, y_test, 'kidney_model.pkl')

    # Train Parkinson's Disease Model
    print("Training Parkinson's Disease Model...")
    X_train, X_test, y_train, y_test = preprocess_parkinsons(parkinsons_df)
    train_and_save_model(X_train, X_test, y_train, y_test, 'parkinsons_model.pkl')

    print("All models trained and saved successfully in 'models/' folder inside the project.")