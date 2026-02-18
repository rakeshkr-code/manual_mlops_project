import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
from typing import Literal
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import subprocess

ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / 'config.yaml'

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

def load_df(file_path: Path) -> pd.DataFrame:
    """
    Load data from a CSV file as a pandas df.
    Args:
        file_path (Path): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataframe.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ [Error] Failed to load data: {str(e)}")
        raise e
    return df

def split_dev_production(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically: first 7000 for development (training + testing), last 3000 for production.
    Args:
        df (pd.DataFrame): Input dataframe.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_dev, df_production)
    """
    dev_size = config['processing']['dev_split']
    
    df_dev = df.iloc[:dev_size].copy()
    df_production = df.iloc[dev_size:].copy()
    
    print(f"\n✓ Split data chronologically:")
    print(f"  - Development set: {len(df_dev)} samples (rows 0-{dev_size-1})")
    print(f"  - Production set: {len(df_production)} samples (rows {dev_size}-{len(df)-1})")
    print(f"  - Development failure rate: {df_dev['Machine failure'].mean()*100:.2f}%")
    print(f"  - Production failure rate: {df_production['Machine failure'].mean()*100:.2f}%")
    
    return df_dev, df_production

def save_versioned_data(
        df: pd.DataFrame, version: int, data_type: str, 
        description: str, production: bool = False
    ) -> Path:
    """
    Save versioned dataset and update manifest.
    Args:
        df (pd.DataFrame): Dataframe to save.
        version (int): Version number (e.g., 1, 2, 3).
        data_type (str): Type of data (e.g., "cleaned", "train", "production").
        description (str): Description of the dataset for manifest.
        production (bool): Whether this is a production dataset.
    Returns:
        Path: The path to the saved file.
    """
    if not production:
        save_dir = ROOT_DIR / config['data_path']['processed_dir']
    else:
        save_dir = ROOT_DIR / config['data_path']['production_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"v{version}_{data_type}.csv"
    filepath = save_dir / filename
    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")
    
    # Update manifest
    userdecision = input(f"Do you want to update the manifest for this dataset? (y/n): ").strip().lower()
    if userdecision == 'y' or userdecision == 'yes':
        update_manifest(version, filename, data_type, description)
    return filepath

def update_manifest(version: int, filename: str, data_type: str, description: str, git_hash: str = None):
    """
    Update manifest.txt with data lineage information.
    Args:
        version (int): Version number of the dataset.
        filename (str): Name of the data file.
        data_type (str): Type of data (e.g., "cleaned", "train", "production").
        description (str): Description of the dataset for manifest.
        git_hash (str, optional): Git hash to include in the manifest. If None, will attempt to get it automatically or ask the user.
    Returns:
        None
    """
    manifest_path = ROOT_DIR / 'datastore' / 'manifest.txt'
    
    print(f"\n[Manifest Update] Version: {version}, Filename: {filename}...\n")

    # Get git hash
    if git_hash is None:
        userinput_git = input("Do you want to include the latest git hash automatically? (y/n): ").strip().lower()
        if userinput_git == 'y' or userinput_git == 'yes':
            try:
                git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], 
                                                  cwd=ROOT_DIR).decode('ascii').strip()
                print(f"✓ Git hash obtained from latest commit automatically: {git_hash}")
            except:
                git_hash = "Not a git repository"
                print("▲ Warning: Could not obtain git hash automatically. This may not be a git repository.")
        else:
            git_hash = input("Please enter the git hash to include in the manifest (or leave blank): ").strip()
            if git_hash == "":
                # If user leaves it blank, we can set it to "Not provided" and give warning
                git_hash = "Not provided"
                print("▲ Warning: Git hash not provided. Consider including it for better traceability.")
    else:
        # If git_hash is provided as an argument, we use it directly without asking the user.
        print(f"✓ Git hash provided as argument: {git_hash}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    manifest_entry = f"""
# {version:02d}
Data File: {filename}
Data Type: {data_type}
File Location: datastore/processed/
Script Path: src/data_prep.py
Git Hash: {git_hash}
Timestamp: {timestamp}
Description: {description}
Comments: {description}

=================================================================================
"""
    
    with open(manifest_path, 'a') as f:
        f.write(manifest_entry)
    
    print(f"✓ Updated manifest: {manifest_path}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PREDICTIVE MAINTENANCE - DATA PREPARATION PIPELINE")
    print("="*80)
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"CONFIG: {CONFIG_PATH}")
    
    # Target: Separating Production Data from Development Data before doing any cleaning or preprocessing or EDA 
    # to prevent data leakage and ensure realistic model evaluation and deployment.
    # - Load raw data, split into development and production sets, save versioned datasets, and update manifest.
    
    # 1. Load raw data
    print("\n[STEP 1] Loading raw data...")
    raw_data_path = ROOT_DIR / config['data_path']['raw_file']
    df_raw = load_df(raw_data_path)
    # 2. Split into development and production sets
    print("\n[STEP 2] Splitting data into development and production sets...")
    df_dev, df_production = split_dev_production(df_raw)
    # 3. Save versioned datasets
    print("\n[STEP 3] Saving versioned datasets...")
    save_versioned_data(df_production, 2, "production", "Production dataset for model deployment", production=True)
    save_versioned_data(df_dev, 3, "development", "Development dataset for training and testing")
