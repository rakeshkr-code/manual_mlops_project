import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from datetime import datetime
import subprocess
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / 'config.yaml'
# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

def load_train_data(train_file: Path) -> pd.DataFrame:
    """
    Load cleaned training data from a CSV file which is versioned as development data.
    Args:
        train_file (Path): Path to the cleaned training data CSV file.
    Returns:
        pd.DataFrame: Loaded training data as a DataFrame.
    """
    if not train_file.exists():
        raise FileNotFoundError(f"Training data file not found at {train_file}")
    df = pd.read_csv(train_file)
    print(f"✓ Loaded training data successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  From file: {train_file}")
    return df

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.
    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
    Returns:
        X (pd.DataFrame): DataFrame containing feature columns.
        y (pd.Series): Series containing the target variable.
    """
    # Target variable
    y = df['Machine failure']
    # Drop all failure mode columns (they're labels, not features)
    failure_columns = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(columns=[col for col in failure_columns if col in df.columns])
    print(f"✓ Separated features and target successfully...")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Feature columns: {X.columns.tolist()}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")
    return X, y

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.
    Args:
        X_train (pd.DataFrame): Training feature DataFrame.
        X_test (pd.DataFrame): Testing feature DataFrame.
    Returns:
        X_train_scaled (pd.DataFrame): Scaled training features.
        X_test_scaled (pd.DataFrame): Scaled testing features.
        scaler (StandardScaler): The fitted scaler object for future use.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Save scaler
    scaler_path = ROOT_DIR / 'models' / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler to: {scaler_path}")
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier on the training data.
    Args:
        X_train (pd.DataFrame): Training feature DataFrame.
        y_train (pd.Series): Training target Series.
    Returns:
        model (RandomForestClassifier): The trained Random Forest model.
    """
    params = config['model_params']
    random_state = config['random_state']
    print(f"\n{'='*80}")
    print("TRAINING MODEL")
    print(f"{'-'*80}")
    print(f"Algorithm: {params['algorithm']}")
    print(f"Parameters: {params}")
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        class_weight=params['class_weight'],
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    print("✓ Model training complete")
    return model

def evaluate_model(model: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model on both training and testing data, and compute various performance metrics.
    Args:
        model (RandomForestClassifier): The trained Random Forest model.
        X_train (pd.DataFrame): Training feature DataFrame.
        y_train (pd.Series): Training target Series.
        X_test (pd.DataFrame): Testing feature DataFrame.
        y_test (pd.Series): Testing target Series.
    Returns:
        metrics (dict): A dictionary containing all computed performance metrics.
    """
    print(f"\n{'='*80}")
    print("EVALUATING MODEL")
    print(f"{'-'*80}")
    # Training predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    # Calculate metrics
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1_score': f1_score(y_train, y_train_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
    }
    print("✓ Model evaluation complete")
    
    # Print results
    print("\nTRAINING SET METRICS:")
    for metric, value in metrics['train'].items():
        print(f"  {metric:12s}: {value:.4f}")
    print("\nTEST SET METRICS:")
    for metric, value in metrics['test'].items():
        print(f"  {metric:12s}: {value:.4f}")
    print("\nCONFUSION MATRIX (Test):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print("\nCLASSIFICATION REPORT (Test):")
    print(classification_report(y_test, y_test_pred, target_names=['No Failure', 'Failure']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTOP 5 FEATURE IMPORTANCES:")
    print(feature_importance.head(5).to_string(index=False))
    
    return metrics

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"CONFIG: {CONFIG_PATH}")

    # Load training data
    train_file = ROOT_DIR / config['data_path']['processed_dir'] / 'v4_dev_cleaned.csv'
    df = load_train_data(train_file)
    
    # Prepare features and target
    X, y = prepare_features(df)
    
    # Split into training and testing sets
    test_size = config['train']['test_size']
    random_state = config['random_state']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)
    print(f"✓ Split data into training and testing sets:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set: {X_test.shape[0]} samples")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    