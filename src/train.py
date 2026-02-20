import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from datetime import datetime
import subprocess
import json
from typing import Any, Dict, List
from sklearn.base import BaseEstimator
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

def get_git_hash() -> str:
    # Get user permission to commit code and retrieve Git hash
    """
    Get the Git hash of the latest commit after optionally committing current code changes. 
     This is important for traceability and reproducibility of the training run. 
     The user is prompted to allow an automated commit of any current code changes, and if they agree, 
     the script will stage all changes, commit with a user-provided message (or a default one), 
     and then retrieve the Git hash from the latest commit. If the user declines or if any errors occur during Git operations, 
     a warning is printed and 'N/A' is returned for the Git hash, with instructions to manually record it 
     in the manifest file for traceability.
    Returns:
        git_hash (str): The Git hash of the latest commit if successful, or 'N/A' if not.
    """
    try:
        permission = input("Do you want to commit the current code changes and retrieve the Git hash? (y/n): ").strip().lower()
        if permission == 'y' or permission == 'yes':
            # Stage all changes
            subprocess.run(['git', 'add', '.'], check=True)
            # Commit changes with message from user input else a generic one
            commit_message = input("Enter a commit message (or press Enter for default): ").strip()
            if not commit_message:
                commit_message = "Automated commit from training script after saving model and metrics"
                print(f"No commit message entered. Using default commit message.")
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            # Get git hash automatically from the latest commit by running subprocess command from python
            output = subprocess.check_output(["git", "log", "-1", "--format=%H %cI"], 
                                                cwd=ROOT_DIR, text=True)
            git_hash, _ = output.strip().split()    # other info like commit timestamp is ignored for now
            print(f"✓ Git hash obtained from latest automated commit: {git_hash}")
            return git_hash
        else:
            print("Warning: No Git commit was made. Please manually record the Git hash in the manifest file for traceability.")
            return 'N/A'
    except subprocess.CalledProcessError as e:
        print(f"Error during Git operations: {e}")
        print("Warning: Git operations failed. Please put the Git hash manually in the manifest file for traceability.")
        return 'N/A'
    except Exception as e:
        print(f"An unexpected error occurred during Git operations: {e}")
        print("Warning: An unexpected error occurred during Git operations. Please put the Git hash manually in the manifest file for traceability.")
        return 'N/A'

def save_model_and_metadata(model: BaseEstimator, scaler: BaseEstimator, metrics: Dict[str, Any], 
                            feature_names: List[str], dataset_version: str) -> tuple[int, Dict[str, Any]]:
    """
    Save the trained model, scaler, and metadata (including performance metrics and feature importance).
    Args:
        model (BaseEstimator): The trained machine learning model to be saved.
        scaler (BaseEstimator): The fitted scaler object used for feature scaling, to be saved for future use in inference.
        metrics (Dict[str, Any]): A dictionary containing performance metrics of the model on training and testing data.
        feature_names (List[str]): A list of feature names corresponding to the features used in training the model.
        dataset_version (str): The version of the dataset used for training, to be included in the metadata for traceability.
    Returns:
        version (int): The version number assigned to the saved model.
        metadata (Dict[str, Any]): A dictionary ...
    """
    model_dir = ROOT_DIR / config['deployment']['model_dir']
    model_dir.mkdir(parents=True, exist_ok=True)
    # Determine version number
    existing_models = list(model_dir.glob('model_v*.pkl'))
    print(f"✓ Found {len(existing_models)} existing model(s) in {model_dir}")
    version = len(existing_models) + 1
    print(f"  Assigning version number: v{version} to the new model")
    # Save model
    model_filename = f'model_v{version}.pkl'
    model_path = model_dir / model_filename
    joblib.dump(model, model_path)
    print(f"\n✓ Saved model to: {model_path}")
    
    # Prepare metadata
    git_hash = get_git_hash()  # Get Git hash with user permission and error handling
    metadata = {
        'version': version,
        'model_filename': model_filename,
        'training_date': datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        'git_commit_hash': git_hash,
        'dataset_version': dataset_version,
        'config': config,
        'metrics': metrics,
        'feature_names': feature_names,
        'model_type': config['model_params']['algorithm'],
        'scaler_path': 'models/scaler.pkl',
        'label_encoder_path': 'models/label_encoder.pkl'
    }
    
    # Save metadata as JSON
    metadata_filename = f'model_v{version}_metadata.json'
    metadata_path = model_dir / metadata_filename
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")
    
    # Append to model_metadata.log
    log_path = model_dir / 'model_metadata.log'
    log_entry = f"""
{'='*80}
MODEL VERSION: v{version}
{'='*80}
Model File: {model_filename}
Metadata Json File: {metadata_filename}
Training Date: {metadata['training_date']}
Git Hash: {metadata['git_commit_hash']}
Dataset Version: {metadata['dataset_version']}

HYPERPARAMETERS:
{json.dumps(config['model_params'], indent=2)}

TRAINING METRICS:
  Accuracy:  {metrics['train']['accuracy']:.4f}
  Precision: {metrics['train']['precision']:.4f}
  Recall:    {metrics['train']['recall']:.4f}
  F1-Score:  {metrics['train']['f1_score']:.4f}
  ROC-AUC:   {metrics['train']['roc_auc']:.4f}

TEST METRICS:
  Accuracy:  {metrics['test']['accuracy']:.4f}
  Precision: {metrics['test']['precision']:.4f}
  Recall:    {metrics['test']['recall']:.4f}
  F1-Score:  {metrics['test']['f1_score']:.4f}
  ROC-AUC:   {metrics['test']['roc_auc']:.4f}

{'='*80}

"""
    
    with open(log_path, 'a') as f:
        f.write(log_entry)
    print(f"✓ Updated model log: {log_path}")
    
    # Update config with current model
    config['deployment']['current_model'] = model_filename
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Updated config.yaml with current model: {model_filename}")
    
    return version, metadata

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"CONFIG: {CONFIG_PATH}")

    # Load training data
    dataset_version = 'v4_dev_cleaned.csv'
    # train_file = ROOT_DIR / config['data_path']['processed_dir'] / (dataset_version := 'v4_dev_cleaned.csv')
    train_file = ROOT_DIR / config['data_path']['processed_dir'] / dataset_version
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
    
    # Save model and metadata
    version, metadata = save_model_and_metadata(model, scaler, metrics, X.columns.tolist(), dataset_version)
    print(f"\n✓ Training pipeline completed successfully. Model version: v{version}")
    