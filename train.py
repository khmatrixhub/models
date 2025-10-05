# Walk-forward training and evaluation for forex data
import click
from datetime import datetime, timedelta
import pandas as pd
import shutil
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE
import os
import logging
import json
from pathlib import Path
from collections import defaultdict

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)


def setup_logging(log_dir, base_filename):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    counter = 1
    while True:
        log_filename = log_dir / f"{base_filename}_{counter:04d}.log"
        if not log_filename.exists():
            break
        counter += 1
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return str(log_filename)


def extract_family_key(feature_name):
    """Extract feature family from feature name (e.g., 'ma_5' -> 'ma')"""
    if '_' in feature_name:
        return feature_name.split('_')[0]
    return feature_name


def top_features_per_family(features, importances, max_per_family=2):
    """Select top features per family to avoid over-representation"""
    family_dict = defaultdict(list)

    # Group features by family
    for feat, imp in zip(features, importances):
        key = extract_family_key(feat)
        family_dict[key].append((feat, imp))

    # Select top N from each family
    selected = []
    for group in family_dict.values():
        top_feats = sorted(group, key=lambda x: -x[1])[:max_per_family]
        selected.extend(top_feats)

    # Sort final output by importance
    selected_sorted = sorted(selected, key=lambda x: -x[1])
    return [f for f, _ in selected_sorted], [i for _, i in selected_sorted]


def train_model(X_train, y_train, config, n_jobs=-1):
    """Train a Random Forest model with feature selection"""
    logging.info(f"DEBUG: Initial data shapes - X_train: {X_train.shape}, y_train: {len(y_train)}")
    
    # Check for duplicate indices first
    if X_train.index.duplicated().any():
        logging.warning(f"DEBUG: Found {X_train.index.duplicated().sum()} duplicate indices in X_train")
        X_train = X_train[~X_train.index.duplicated(keep='first')]
        logging.info(f"DEBUG: After X_train deduplication: {X_train.shape}")
    
    if y_train.index.duplicated().any():
        logging.warning(f"DEBUG: Found {y_train.index.duplicated().sum()} duplicate indices in y_train")
        y_train = y_train[~y_train.index.duplicated(keep='first')]
        logging.info(f"DEBUG: After y_train deduplication: {len(y_train)}")
    
    # Clean features first
    initial_X_len = len(X_train)
    X_train = X_train.dropna()
    logging.info(f"DEBUG: After dropna() - X_train: {X_train.shape} (removed {initial_X_len - len(X_train)} rows)")
    
    # Align labels with cleaned features
    initial_y_len = len(y_train)
    y_train = y_train.loc[X_train.index]
    logging.info(f"DEBUG: After index alignment - y_train: {len(y_train)} (removed {initial_y_len - len(y_train)} rows)")
    
    # Check if shapes match now
    if len(X_train) != len(y_train):
        logging.error(f"DEBUG: Shape mismatch after alignment! X_train: {len(X_train)}, y_train: {len(y_train)}")
        logging.error(f"DEBUG: X_train index range: {X_train.index.min()} to {X_train.index.max()}")
        logging.error(f"DEBUG: y_train index range: {y_train.index.min()} to {y_train.index.max()}")
        logging.error(f"DEBUG: Common indices: {len(X_train.index.intersection(y_train.index))}")
        raise ValueError(f"Shape mismatch: X_train has {len(X_train)} samples, y_train has {len(y_train)} samples")
    
    use_rfe = config.get("use_rfe", False)
    use_bagging = config.get("bagging", {}).get("enabled", False)
    bagging_estimators = config.get("bagging", {}).get("n_estimators", 100)
    n_estimators = config.get("n_estimators", 100)
    
    logging.info(f"Training RF model with {n_estimators} estimators, bagging={use_bagging}, use_rfe={use_rfe}")

    # Additional validation for invalid values
    if X_train.isna().any().any():
        logging.warning("Found NaN values in training data after initial cleaning")
        nan_columns = X_train.columns[X_train.isna().any()].tolist()
        logging.warning(f"Columns with NaN: {nan_columns}")
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        logging.info(f"DEBUG: After second dropna() - X_train: {X_train.shape}, y_train: {len(y_train)}")
    
    if np.isinf(X_train).any().any():
        logging.warning("Found infinite values in training data")
        inf_columns = X_train.columns[np.isinf(X_train).any()].tolist()
        logging.warning(f"Columns with inf: {inf_columns}")
        X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
        y_train = y_train.loc[X_train.index]
        logging.info(f"DEBUG: After inf cleaning - X_train: {X_train.shape}, y_train: {len(y_train)}")

    # Initialize classifier
    if use_bagging:
        base_classifier = RandomForestClassifier(
            n_estimators=1, 
            class_weight='balanced_subsample',
            bootstrap=False,
            criterion='entropy', 
            random_state=42
        )
        rf_classifier = BaggingClassifier(
            estimator=base_classifier,
            n_estimators=bagging_estimators,
            max_features=1.0,
            bootstrap=False,
            n_jobs=n_jobs,
            random_state=42
        )
    else:
        rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced_subsample',
            criterion='entropy', 
            random_state=42,
            n_jobs=n_jobs
        )

    # Feature selection
    if use_rfe:
        rfe_estimators = config.get("rfe", {}).get("n_estimators", 50)
        rfe_features = config.get("rfe", {}).get("n_features", 20)
        logging.info(f"RFE settings: estimators={rfe_estimators}, features={rfe_features}")
        
        rfe_classifier = RandomForestClassifier(
            n_estimators=rfe_estimators,
            class_weight='balanced_subsample',
            criterion='entropy',
            random_state=42,
            n_jobs=n_jobs
        )
        rfe = RFE(estimator=rfe_classifier, n_features_to_select=rfe_features)
        rfe.fit(X_train, y_train.values.ravel())
        selected_features = X_train.columns[rfe.support_]
    else:
        # Use importance-based feature selection with family grouping
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=n_jobs)
        rf_temp.fit(X_train, y_train)
        
        importances = rf_temp.feature_importances_
        top_features, top_imps = top_features_per_family(
            X_train.columns,
            importances,
            max_per_family=config.get("max_features_per_family", 3)
        )
        
        # Take top K overall features - ensure deterministic selection
        K = config.get("max_features", 50)
        selected_features = top_features[:K]  # Already sorted by importance in top_features_per_family
        
        logging.info(f"Selected {len(selected_features)} features from {len(X_train.columns)} total")
        logging.info(f"Top 5 selected features: {selected_features[:5]}")

    # Train final model on selected features
    X_train_selected = X_train[selected_features]
    
    logging.info(f"Training data shape: {X_train_selected.shape}")
    logging.info(f"Target distribution: {y_train.value_counts().to_dict()}")
    
    rf_classifier.fit(X_train_selected, y_train.values.ravel())
    
    return rf_classifier, selected_features


def predict_model(model, X_test, selected_features):
    """Make predictions using trained model"""
    logging.info(f"DEBUG PREDICT: Input X_test shape: {X_test.shape}")
    logging.info(f"DEBUG PREDICT: Selected features count: {len(selected_features)}")
    
    if len(X_test) == 0:
        logging.warning("DEBUG PREDICT: Empty X_test, returning empty series")
        return pd.Series([], dtype=float)
    
    # Check for duplicates in test data
    if X_test.index.duplicated().any():
        logging.warning(f"DEBUG PREDICT: Found {X_test.index.duplicated().sum()} duplicate indices in X_test")
        X_test = X_test[~X_test.index.duplicated(keep='first')]
        logging.info(f"DEBUG PREDICT: After deduplication: {X_test.shape}")
    
    X_test = X_test.dropna()
    logging.info(f"DEBUG PREDICT: After dropna: {X_test.shape}")
    
    if len(X_test) == 0:
        logging.warning("DEBUG PREDICT: Empty X_test after cleaning, returning empty series")
        return pd.Series([], dtype=float)
    
    # Check if all selected features exist in X_test
    missing_features = [f for f in selected_features if f not in X_test.columns]
    if missing_features:
        logging.error(f"DEBUG PREDICT: Missing features in X_test: {missing_features}")
        logging.error(f"DEBUG PREDICT: Available features: {list(X_test.columns)}")
        raise ValueError(f"Missing features: {missing_features}")
    
    X_test_selected = X_test[selected_features]
    logging.info(f"DEBUG PREDICT: X_test_selected shape: {X_test_selected.shape}")
    logging.info(f"DEBUG PREDICT: Expected features: {len(selected_features)}")
    
    # Check for any remaining NaN or inf values
    if X_test_selected.isna().any().any():
        logging.warning("DEBUG PREDICT: Found NaN in selected features")
        nan_cols = X_test_selected.columns[X_test_selected.isna().any()].tolist()
        logging.warning(f"DEBUG PREDICT: NaN columns: {nan_cols}")
        X_test_selected = X_test_selected.dropna()
        logging.info(f"DEBUG PREDICT: After NaN removal: {X_test_selected.shape}")
    
    if np.isinf(X_test_selected).any().any():
        logging.warning("DEBUG PREDICT: Found inf in selected features")
        inf_cols = X_test_selected.columns[np.isinf(X_test_selected).any()].tolist()
        logging.warning(f"DEBUG PREDICT: Inf columns: {inf_cols}")
        X_test_selected = X_test_selected.replace([np.inf, -np.inf], np.nan).dropna()
        logging.info(f"DEBUG PREDICT: After inf removal: {X_test_selected.shape}")
    
    if len(X_test_selected) == 0:
        logging.warning("DEBUG PREDICT: Empty X_test_selected after cleaning, returning empty series")
        return pd.Series([], dtype=float)
    
    try:
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        logging.info(f"DEBUG PREDICT: Successfully predicted {len(y_pred_proba)} samples")
        return pd.Series(y_pred_proba, index=X_test_selected.index)
    except Exception as e:
        logging.error(f"DEBUG PREDICT: Prediction failed: {str(e)}")
        raise


def predict_model(model, X_test, selected_features):
    """Make predictions using trained model"""
    logging.info(f"DEBUG PREDICT: Input X_test shape: {X_test.shape}")
    logging.info(f"DEBUG PREDICT: Selected features count: {len(selected_features)}")
    
    if len(X_test) == 0:
        logging.warning("DEBUG PREDICT: Empty X_test, returning empty series")
        return pd.Series([], dtype=float)
    
    # Check for duplicates in test data
    if X_test.index.duplicated().any():
        logging.warning(f"DEBUG PREDICT: Found {X_test.index.duplicated().sum()} duplicate indices in X_test")
        X_test = X_test[~X_test.index.duplicated(keep='first')]
        logging.info(f"DEBUG PREDICT: After deduplication: {X_test.shape}")
    
    X_test = X_test.dropna()
    logging.info(f"DEBUG PREDICT: After dropna: {X_test.shape}")
    
    if len(X_test) == 0:
        logging.warning("DEBUG PREDICT: Empty X_test after cleaning, returning empty series")
        return pd.Series([], dtype=float)
    
    # Check if all selected features exist in X_test
    missing_features = [f for f in selected_features if f not in X_test.columns]
    if missing_features:
        logging.error(f"DEBUG PREDICT: Missing features in X_test: {missing_features}")
        logging.error(f"DEBUG PREDICT: Available features: {list(X_test.columns)}")
        raise ValueError(f"Missing features: {missing_features}")
    
    X_test_selected = X_test[selected_features]
    logging.info(f"DEBUG PREDICT: X_test_selected shape: {X_test_selected.shape}")
    logging.info(f"DEBUG PREDICT: Expected features: {len(selected_features)}")
    
    # Check for any remaining NaN or inf values
    if X_test_selected.isna().any().any():
        logging.warning("DEBUG PREDICT: Found NaN in selected features")
        nan_cols = X_test_selected.columns[X_test_selected.isna().any()].tolist()
        logging.warning(f"DEBUG PREDICT: NaN columns: {nan_cols}")
        X_test_selected = X_test_selected.dropna()
        logging.info(f"DEBUG PREDICT: After NaN removal: {X_test_selected.shape}")
    
    if np.isinf(X_test_selected).any().any():
        logging.warning("DEBUG PREDICT: Found inf in selected features")
        inf_cols = X_test_selected.columns[np.isinf(X_test_selected).any()].tolist()
        logging.warning(f"DEBUG PREDICT: Inf columns: {inf_cols}")
        X_test_selected = X_test_selected.replace([np.inf, -np.inf], np.nan).dropna()
        logging.info(f"DEBUG PREDICT: After inf removal: {X_test_selected.shape}")
    
    if len(X_test_selected) == 0:
        logging.warning("DEBUG PREDICT: Empty X_test_selected after cleaning, returning empty series")
        return pd.Series([], dtype=float)
    
    try:
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        logging.info(f"DEBUG PREDICT: Successfully predicted {len(y_pred_proba)} samples")
        return pd.Series(y_pred_proba, index=X_test_selected.index)
    except Exception as e:
        logging.error(f"DEBUG PREDICT: Prediction failed: {str(e)}")
        raise


@click.command()
@click.option('--config', type=click.Path(exists=True), help='Path to configuration YAML file')
@click.option('--start-train-date', type=click.DateTime(formats=["%m%d%Y"]), help='Start date for training in MMDDYYYY format')
@click.option('--end-date', type=click.DateTime(formats=["%m%d%Y"]), help='End date for evaluation in MMDDYYYY format')
@click.option('--retrain-frequency', default=5, help='Retrain model every N days')
@click.option('--min-train-days', default=30, help='Minimum days of data before first training')
@click.option('--n-jobs', default=-1, help='Number of parallel jobs')
@click.option('--experiment', default='default', help='Experiment name for organizing results')
@click.option('--label-name', default=None, type=str, help='Name of label configuration to use (e.g., "conservative", "aggressive", "swing")')
def main(config, start_train_date, end_date, retrain_frequency, min_train_days, n_jobs, experiment, label_name):
    """
    Walk-forward training and evaluation for forex trading models
    """
    # Load configuration
    import yaml
    with open(config, 'r') as f:
        settings = yaml.safe_load(f)
    
    # Setup directories using new structure: results/<config_name>/
    config_path = Path(config)
    config_name = config_path.stem  # Get config filename without extension
    
    # Base directory for this config under results/
    base_results_dir = Path("results") / config_name
    
    # Data directories (where mainoffset.py saves outputs)
    signals_dir = base_results_dir / "signals"
    data_dir = base_results_dir / "data"  # Features and labels are here now
    
    # Training experiment directories (where train_forex.py saves outputs)
    models_dir = base_results_dir / "models" / experiment
    predictions_dir = base_results_dir / "predictions" / experiment
    experiment_results_dir = base_results_dir / "models" / experiment / "results"
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    experiment_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Aliases for backward compatibility
    results_dir = experiment_results_dir
    
    # Setup logging
    log_file = setup_logging(base_results_dir / "logs", f"train_forex_{experiment}")
    logging.info(f"Starting walk-forward training experiment: {experiment}")
    logging.info(f"Config: {config_name}")
    logging.info(f"Data directory: {data_dir}")
    logging.info(f"Models directory: {models_dir}")
    logging.info(f"Results directory: {results_dir}")
    
    # Copy config for reference
    shutil.copy(config, results_dir / "config.yaml")
    
    # Save experiment metadata
    experiment_metadata = {
        'experiment_name': experiment,
        'start_date': start_train_date.strftime("%Y%m%d") if start_train_date else settings['start_date'],
        'end_date': end_date.strftime("%Y%m%d") if end_date else settings['end_date'],
        'retrain_frequency': retrain_frequency,
        'min_train_days': min_train_days,
        'n_jobs': n_jobs
    }
    
    # Parse dates and create date range
    spot = settings['spot']
    
    # Get label configuration
    if 'labels' not in settings or not isinstance(settings['labels'], dict):
        raise ValueError("Config must have 'labels' as a dictionary of named configurations")
    
    # If no label name specified, use the first one
    if label_name is None:
        label_name = list(settings['labels'].keys())[0]
        logging.info(f"No label name specified, using first available: '{label_name}'")
    
    if label_name not in settings['labels']:
        available_labels = ', '.join(settings['labels'].keys())
        raise ValueError(f"Label '{label_name}' not found in config. Available labels: {available_labels}")
    
    selected_label = settings['labels'][label_name]
    pt = selected_label['pt']
    sl = selected_label['sl']
    max_hold = selected_label['max_hold']
    
    # Add label config to metadata
    experiment_metadata['label_config'] = {
        'name': label_name,
        'profit_target': pt,
        'stop_loss': sl,
        'max_hold': max_hold,
        'all_configs': settings['labels']
    }
    
    with open(results_dir / "experiment_metadata.json", 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    
    logging.info(f"Using label configuration '{label_name}': PT={pt}, SL={sl}, MaxHold={max_hold}")
    logging.info(f"Available label configurations in config: {list(settings['labels'].keys())}")
    
    # Use dates from config if not provided via CLI
    if start_train_date is None:
        start_train_date = datetime.strptime(settings['start_date'], "%m%d%Y")
    if end_date is None:
        end_date = datetime.strptime(settings['end_date'], "%m%d%Y")
    
    # Generate weekdays for processing
    weekdays = []
    current_date = start_train_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday to Friday
            weekdays.append(current_date)
        current_date += timedelta(days=1)
    
    logging.info(f"Processing {len(weekdays)} trading days from {start_train_date} to {end_date}")
    
    # Initialize tracking variables
    model = None
    selected_features = None
    all_features = pd.DataFrame()
    all_labels = pd.Series(dtype=int)
    all_predictions = pd.Series(dtype=float)
    all_test_labels = pd.Series(dtype=int)
    all_profits = pd.Series(dtype=float)
    
    # Track processing statistics
    dates_processed = []
    dates_skipped = []
    dates_with_predictions = []
    dates_with_training = []
    
    training_config = {
        'n_estimators': 100,
        'use_rfe': False,
        'max_features': 30,
        'max_features_per_family': 3,
        'bagging': {'enabled': False}
    }
    
    # Walk-forward validation loop
    for i, date in enumerate(weekdays):
        date_str = date.strftime("%Y%m%d")
        logging.info(f"Processing date {date_str} ({i+1}/{len(weekdays)})")
        
        try:
            # Load data for current date with currency pair in filename
            features_file = data_dir / f"{date_str}_{spot}_features.parquet"
            
            # Construct label filename using selected label configuration
            labels_file = data_dir / f"{date_str}_{spot}_{pt}_{sl}_{max_hold}_y.parquet"
            
            logging.info(f"Looking for label file: {labels_file.name}")
            
            if not features_file.exists():
                logging.warning(f"Missing features file for {date_str}: {features_file.name}")
                dates_skipped.append(date_str)
                continue
            
            if not labels_file.exists():
                logging.warning(f"Missing label file for {date_str}: {labels_file.name}")
                dates_skipped.append(date_str)
                continue
            
            # Check for signals file
            signals_file = signals_dir / f"{date_str}_{spot}_signals.parquet"
            if not signals_file.exists():
                logging.warning(f"Missing signals file for {date_str}: {signals_file.name}")
                dates_skipped.append(date_str)
                continue
            
            # Load signals, features, and labels
            signals = pd.read_parquet(signals_file)
            features = pd.read_parquet(features_file)
            labels_df = pd.read_parquet(labels_file)
            
            logging.info(f"DEBUG: Raw data loaded - signals: {signals.shape}, features: {features.shape}, labels_df: {labels_df.shape}")
            
            # Filter to only signal bars (where signal != 0)
            # This ensures we only train on MA crossover bars
            if 'signal' in signals.columns:
                signal_series = signals['signal']
            else:
                signal_series = signals.iloc[:, 0]  # First column if name is different
            
            signal_mask = signal_series != 0
            signal_count = signal_mask.sum()
            
            logging.info(f"DEBUG: Filtering to {signal_count} signal bars (out of {len(signal_mask)} total)")
            
            # Filter all data to only signal bars
            signals = signals[signal_mask]
            features = features[signal_mask]
            labels_df = labels_df[signal_mask]
            
            logging.info(f"DEBUG: After signal filtering - signals: {signals.shape}, features: {features.shape}, labels_df: {labels_df.shape}")
            logging.info(f"DEBUG: Features index type: {type(features.index)}, Labels index type: {type(labels_df.index)}")
            
            # DEBUG: Check for duplicates in raw data
            features_dups = features.index.duplicated().sum()
            labels_dups = labels_df.index.duplicated().sum()
            logging.info(f"DEBUG: Raw duplicates - features: {features_dups}, labels: {labels_dups}")
            
            if features_dups > 0:
                logging.warning(f"DEBUG: Found {features_dups} duplicate indices in raw features")
                # Show some examples of duplicate timestamps
                dup_indices = features.index[features.index.duplicated(keep=False)]
                unique_dups = dup_indices.unique()[:5]  # Show first 5 duplicate timestamps
                logging.warning(f"DEBUG: Example duplicate timestamps: {unique_dups.tolist()}")
                
                # Show how many times each duplicate appears
                for dup_time in unique_dups:
                    count = (features.index == dup_time).sum()
                    logging.warning(f"DEBUG: Timestamp {dup_time} appears {count} times")
            
            # Clean and align data
            initial_features_len = len(features)
            features = features.dropna()
            logging.info(f"DEBUG: Features after dropna: {features.shape} (removed {initial_features_len - len(features)})")
            
            # Align labels with features BEFORE creating binary labels
            labels_df = labels_df.loc[features.index]
            logging.info(f"DEBUG: Labels after alignment: {labels_df.shape}")
            
            if len(features) == 0:
                logging.warning(f"No valid features for {date_str}")
                dates_skipped.append(date_str)
                continue
            
            # Extract binary labels and profits from label DataFrame
            # Labels file now contains: 'label' (1/-1), 'profit' (actual %), 'exit_bars'
            if 'profit' not in labels_df.columns:
                logging.error(f"Label file missing 'profit' column for {date_str}")
                logging.error(f"Available columns: {list(labels_df.columns)}")
                dates_skipped.append(date_str)
                continue
            
            if 'label' not in labels_df.columns:
                logging.error(f"Label file missing 'label' column for {date_str}")
                logging.error(f"Available columns: {list(labels_df.columns)}")
                dates_skipped.append(date_str)
                continue
            
            # Convert labels to binary (1 = profitable, 0 = loss)
            # Original label is 1 (profit) or -1 (loss)
            binary_labels = (labels_df['label'] > 0).astype(int)
            profits = labels_df['profit']
            
            logging.info(f"DEBUG: Label statistics - Profitable: {(binary_labels == 1).sum()}/{len(binary_labels)} ({(binary_labels == 1).mean()*100:.1f}%)")
            logging.info(f"DEBUG: Profit statistics - Mean: {profits.mean():.6f}, Median: {profits.median():.6f}, Std: {profits.std():.6f}")
            
            logging.info(f"DEBUG: Final aligned data - features: {features.shape}, binary_labels: {len(binary_labels)}, profits: {len(profits)}")
            
            # Verify indices match
            if not features.index.equals(binary_labels.index):
                logging.error(f"DEBUG: Index mismatch after processing!")
                logging.error(f"Features index: {features.index[:5]}...")
                logging.error(f"Labels index: {binary_labels.index[:5]}...")
                dates_skipped.append(date_str)
                continue
            
            logging.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
            dates_processed.append(date_str)
            
            # Make predictions if model exists
            if model is not None and selected_features is not None:
                logging.info(f"DEBUG: Making predictions for {date_str}")
                logging.info(f"DEBUG: Model type: {type(model)}")
                logging.info(f"DEBUG: Selected features: {len(selected_features)}")
                logging.info(f"DEBUG: Features for prediction: {features.shape}")
                
                predictions = predict_model(model, features, selected_features)
                
                if len(predictions) > 0:
                    logging.info(f"DEBUG: Received {len(predictions)} predictions")
                    
                    # IMPORTANT: Align test labels and profits with the actual prediction indices
                    # The prediction function may have removed duplicates or NaN values
                    test_labels_aligned = binary_labels.loc[predictions.index]
                    profits_aligned = profits.loc[predictions.index]
                    
                    logging.info(f"DEBUG: Aligned data - predictions: {len(predictions)}, test_labels: {len(test_labels_aligned)}, profits: {len(profits_aligned)}")
                    
                    # Verify alignment
                    if len(predictions) != len(test_labels_aligned) or len(predictions) != len(profits_aligned):
                        logging.error(f"DEBUG: Alignment failed! Predictions: {len(predictions)}, Labels: {len(test_labels_aligned)}, Profits: {len(profits_aligned)}")
                        continue
                    
                    all_predictions = pd.concat([all_predictions, predictions])
                    all_test_labels = pd.concat([all_test_labels, test_labels_aligned])
                    all_profits = pd.concat([all_profits, profits_aligned])
                    
                    logging.info(f"DEBUG: Total accumulated - predictions: {len(all_predictions)}, test_labels: {len(all_test_labels)}, profits: {len(all_profits)}")
                    
                    # Verify accumulated data alignment
                    if len(all_predictions) != len(all_test_labels) or len(all_predictions) != len(all_profits):
                        logging.error(f"DEBUG: Accumulated data misalignment!")
                        logging.error(f"Predictions: {len(all_predictions)}, Test labels: {len(all_test_labels)}, Profits: {len(all_profits)}")
                        continue
                    
                    # Evaluate current performance - simplified version for logging
                    threshold_50 = (all_predictions >= 0.5).astype(int)
                    current_accuracy = accuracy_score(all_test_labels, threshold_50)
                    current_profit = all_profits[all_predictions >= 0.5].sum()
                    current_trades = (all_predictions >= 0.5).sum()
                    current_mean_profit = all_profits[all_predictions >= 0.5].mean() if current_trades > 0 else 0
                    
                    logging.info(f"Current performance (threshold 0.5):")
                    logging.info(f"  Accuracy: {current_accuracy:.3f}, Total Profit: {current_profit:.6f}")
                    logging.info(f"  Trades: {current_trades}, Mean Profit: {current_mean_profit:.6f}")
                else:
                    logging.warning(f"DEBUG: No predictions returned for {date_str}")
            
            # Accumulate training data
            logging.info(f"DEBUG: Before accumulation - all_features: {all_features.shape}, all_labels: {len(all_labels)}")
            logging.info(f"DEBUG: Adding current data - features: {features.shape}, labels: {len(binary_labels)}")
            
            # Check for duplicate indices before accumulating
            if len(all_features) > 0:
                duplicate_indices = all_features.index.intersection(features.index)
                if len(duplicate_indices) > 0:
                    logging.warning(f"DEBUG: Found {len(duplicate_indices)} duplicate indices, removing from new data")
                    features = features.loc[~features.index.isin(duplicate_indices)]
                    binary_labels = binary_labels.loc[~binary_labels.index.isin(duplicate_indices)]
                    logging.info(f"DEBUG: After duplicate removal - features: {features.shape}, labels: {len(binary_labels)}")
            
            all_features = pd.concat([all_features, features])
            all_labels = pd.concat([all_labels, binary_labels])
            
            logging.info(f"DEBUG: After accumulation - all_features: {all_features.shape}, all_labels: {len(all_labels)}")
            
            # Check for duplicates in accumulated data
            if all_features.index.duplicated().any():
                logging.warning(f"DEBUG: Found duplicated indices in accumulated features: {all_features.index.duplicated().sum()}")
                all_features = all_features[~all_features.index.duplicated(keep='first')]
                all_labels = all_labels[~all_labels.index.duplicated(keep='first')]
                logging.info(f"DEBUG: After deduplication - all_features: {all_features.shape}, all_labels: {len(all_labels)}")
            
            # Retrain model if conditions are met
            should_retrain = (
                (i % retrain_frequency == 0 and i >= min_train_days) or
                (model is None and len(all_features) >= min_train_days)
            )
            
            if should_retrain:
                logging.info(f"Retraining model with {len(all_features)} samples")
                
                try:
                    model, selected_features = train_model(
                        all_features, 
                        all_labels, 
                        training_config, 
                        n_jobs
                    )
                    
                    # Verify model was trained correctly
                    logging.info(f"DEBUG TRAINING: Model trained successfully")
                    logging.info(f"DEBUG TRAINING: Selected features: {len(selected_features)}")
                    logging.info(f"DEBUG TRAINING: Model classes: {getattr(model, 'classes_', 'N/A')}")
                    logging.info(f"DEBUG TRAINING: Training data shape: {all_features[selected_features].shape}")
                    
                    # Test the model with a small sample to verify it works
                    test_sample = all_features[selected_features].iloc[:5]
                    try:
                        test_pred = model.predict_proba(test_sample)
                        logging.info(f"DEBUG TRAINING: Model test prediction successful, shape: {test_pred.shape}")
                    except Exception as test_e:
                        logging.error(f"DEBUG TRAINING: Model test prediction failed: {str(test_e)}")
                        raise test_e
                    
                    # Save model info
                    model_info = {
                        'date': date_str,
                        'features': len(selected_features),
                        'samples': len(all_features),
                        'selected_features': list(selected_features),
                        'label_config': {
                            'name': label_name,
                            'profit_target': pt,
                            'stop_loss': sl,
                            'max_hold': max_hold
                        }
                    }
                    
                    with open(models_dir / f"model_info_{date_str}.json", 'w') as f:
                        json.dump(model_info, f, indent=2)
                    
                    logging.info(f"Model retrained successfully with {len(selected_features)} features")
                    
                except Exception as e:
                    logging.error(f"Failed to train model: {str(e)}")
                    import traceback
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    continue
        
        except Exception as e:
            logging.error(f"Error processing {date_str}: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    # Final evaluation and results export
    if len(all_predictions) > 0:
        logging.info("Generating final evaluation results...")
        
        # Create detailed predictions dataframe like in train_events.py
        predictions_df = pd.DataFrame({
            'true_label': all_test_labels,
            'prediction_proba': all_predictions,
            'profit': all_profits
        })
        
        # Add additional columns
        predictions_df['trade_date'] = predictions_df.index.date
        predictions_df['exit_price'] = predictions_df['profit'] + 1.0  # Assuming profit is in decimal form
        predictions_df['return'] = predictions_df['profit']
        
        # Evaluate across multiple thresholds like train_events.py
        thresholds = np.arange(0.0, 1.05, 0.05)
        
        precisions = []
        recalls = []
        f1s = []
        accuracies = []
        profits = []
        mean_profits = []
        median_profits = []
        counts = []
        
        for threshold in thresholds:
            logging.info(f"Evaluating threshold {threshold:.2f}")
            
            # Convert probabilities to binary predictions
            y_pred_binary = (all_predictions >= threshold).astype(int)
            
            # Calculate classification metrics
            accuracy = accuracy_score(all_test_labels, y_pred_binary)
            precision = precision_score(all_test_labels, y_pred_binary, zero_division=0)
            recall = recall_score(all_test_labels, y_pred_binary, zero_division=0)
            f1 = f1_score(all_test_labels, y_pred_binary, zero_division=0)
            
            # Calculate trading metrics
            selected_trades = all_predictions >= threshold
            total_profit = all_profits[selected_trades].sum() if selected_trades.any() else 0
            mean_profit = all_profits[selected_trades].mean() if selected_trades.any() else 0
            median_profit = all_profits[selected_trades].median() if selected_trades.any() else 0
            trade_count = selected_trades.sum()
            
            # Store results
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            profits.append(total_profit)
            mean_profits.append(mean_profit)
            median_profits.append(median_profit)
            counts.append(trade_count)
            
            # Create filtered predictions for this threshold
            threshold_predictions = predictions_df[predictions_df['prediction_proba'] >= threshold].copy()
            
            if len(threshold_predictions) > 0:
                # Daily aggregation like train_events.py
                daily_results = threshold_predictions.groupby('trade_date')['profit'].agg(['count', 'sum', 'mean'])
                daily_results['cumsum'] = threshold_predictions.groupby('trade_date')['profit'].sum().cumsum()
                daily_results['total_signals'] = daily_results['count'].cumsum()
                daily_results['total_mean'] = daily_results['cumsum'] / daily_results['total_signals']
                daily_results['drawdown'] = daily_results['cumsum'] - daily_results['cumsum'].cummax()
                
                # Save daily results
                threshold_str = f"{threshold:.2f}".replace('.', '_')
                daily_file = results_dir / f"daily_{threshold_str}.csv.bz2"
                daily_results.to_csv(daily_file)
                
                # Save detailed predictions for this threshold
                predictions_file = results_dir / f"predicted_results_{threshold_str}.csv.bz2"
                threshold_predictions.to_csv(predictions_file)
        
        # Create comprehensive results summary
        final_results = pd.DataFrame({
            'threshold': thresholds,
            'count': counts,
            'accuracy': accuracies,
            'precision': precisions,
            'recall': recalls,
            'f1': f1s,
            'total_profit': profits,
            'mean_profit': mean_profits,
            'median_profit': median_profits,
        })
        
        # Save results
        final_results.to_csv(results_dir / "evaluation_results.csv", index=False)
        final_results.to_parquet(results_dir / "evaluation_results.parquet")
        final_results.to_parquet(results_dir / "results.parquet")  # train_events.py naming
        
        # Save all predictions
        predictions_df.to_parquet(results_dir / "all_predictions.parquet")
        predictions_df.to_parquet(results_dir / "predicted_results.parquet")  # train_events.py naming
        
        # Daily profit summary (like train_events.py)
        daily_profit = predictions_df.groupby('trade_date')['profit'].sum()
        daily_profit.to_csv(results_dir / "daily.csv.bz2")
        
        # Print detailed summary like train_events.py
        best_f1_row = final_results.loc[final_results['f1'].idxmax()]
        best_profit_row = final_results.loc[final_results['total_profit'].idxmax()]
        best_sharpe_threshold = 0.5  # You can calculate Sharpe ratio if needed
        
        logging.info("="*60)
        logging.info("FINAL RESULTS SUMMARY")
        logging.info("="*60)
        logging.info(f"Total predictions made: {len(all_predictions)}")
        logging.info(f"Total profit (all trades): {all_profits.sum():.6f}")
        logging.info(f"Mean profit per trade: {all_profits.mean():.6f}")
        logging.info(f"Median profit per trade: {all_profits.median():.6f}")
        logging.info(f"Win rate (profitable trades): {(all_profits > 0).mean():.3f}")
        logging.info("-"*60)
        logging.info(f"Best F1 Score: {best_f1_row['f1']:.3f} at threshold {best_f1_row['threshold']:.2f}")
        logging.info(f"  - Trades: {best_f1_row['count']}, Total Profit: {best_f1_row['total_profit']:.6f}")
        logging.info(f"  - Mean Profit: {best_f1_row['mean_profit']:.6f}")
        logging.info("-"*60)
        logging.info(f"Best Total Profit: {best_profit_row['total_profit']:.6f} at threshold {best_profit_row['threshold']:.2f}")
        logging.info(f"  - Trades: {best_profit_row['count']}, F1 Score: {best_profit_row['f1']:.3f}")
        logging.info(f"  - Mean Profit: {best_profit_row['mean_profit']:.6f}")
        logging.info("-"*60)
        
        # Show top performing thresholds
        top_profit_thresholds = final_results.nlargest(5, 'total_profit')[['threshold', 'total_profit', 'count', 'mean_profit', 'f1']]
        logging.info("Top 5 thresholds by total profit:")
        for _, row in top_profit_thresholds.iterrows():
            logging.info(f"  {row['threshold']:.2f}: Profit={row['total_profit']:.6f}, "
                        f"Trades={row['count']}, Mean={row['mean_profit']:.6f}, F1={row['f1']:.3f}")
        
        logging.info("-"*60)
        logging.info(f"Results saved to: {results_dir}")
        
        # Print processing statistics
        logging.info("\n" + "="*60)
        logging.info("TRAINING SUMMARY")
        logging.info("="*60)
        logging.info(f"Total dates processed: {len(dates_processed)}")
        logging.info(f"Total dates skipped: {len(dates_skipped)}")
        logging.info(f"Dates with predictions: {len(dates_with_predictions)}")
        logging.info(f"Dates with training: {len(dates_with_training)}")
        
        if dates_skipped:
            logging.info(f"\nSkipped dates: {', '.join(dates_skipped[:20])}{'...' if len(dates_skipped) > 20 else ''}")
        
        # Print complete results table for all thresholds
        logging.info("\nComplete Results for All Thresholds:")
        logging.info("="*120)
        logging.info(final_results.to_string(index=False))
        logging.info("="*120)
        
        # Output feature importance from the final model
        if model is not None and selected_features is not None:
            logging.info("\n" + "="*60)
            logging.info("FEATURE IMPORTANCE ANALYSIS")
            logging.info("="*60)
            
            try:
                # Get feature importances from the trained model
                feature_importances = model.feature_importances_
                
                # Create a dataframe with features and their importance
                importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': feature_importances
                }).sort_values('importance', ascending=False)
                
                # Add family column
                importance_df['family'] = importance_df['feature'].apply(extract_family_key)
                
                # Save to file
                importance_df.to_csv(results_dir / "feature_importance.csv", index=False)
                importance_df.to_parquet(results_dir / "feature_importance.parquet")
                
                # Log top features
                logging.info(f"\nTop 20 Most Important Features:")
                logging.info("-"*80)
                logging.info(f"{'Rank':<6} {'Feature':<25} {'Family':<15} {'Importance':<12}")
                logging.info("-"*80)
                
                for idx, row in importance_df.head(20).iterrows():
                    rank = idx + 1
                    logging.info(f"{rank:<6} {row['feature']:<25} {row['family']:<15} {row['importance']:<12.6f}")
                
                # Group by family and show aggregate importance
                family_importance = importance_df.groupby('family')['importance'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
                
                logging.info("\n" + "-"*80)
                logging.info("Feature Importance by Family:")
                logging.info("-"*80)
                logging.info(f"{'Family':<20} {'Total Imp':<15} {'Mean Imp':<15} {'Count':<10}")
                logging.info("-"*80)
                
                for family, row in family_importance.iterrows():
                    logging.info(f"{family:<20} {row['sum']:<15.6f} {row['mean']:<15.6f} {int(row['count']):<10}")
                
                logging.info("-"*80)
                logging.info(f"Feature importance saved to: {results_dir / 'feature_importance.csv'}")
                
            except Exception as e:
                logging.error(f"Error calculating feature importance: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
        else:
            logging.warning("No model available for feature importance analysis")
        
    else:
        logging.warning("No predictions were made. Check your data and configuration.")


if __name__ == '__main__':
    main()