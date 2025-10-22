# Walk-forward regression training for forex profit prediction
import click
from datetime import datetime, timedelta
import pandas as pd
import shutil
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    """Train a Random Forest Regressor model with feature selection"""
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
    model_type = config.get("model_type", "random_forest")  # random_forest or gradient_boosting
    
    logging.info(f"Training regression model with {n_estimators} estimators, model_type={model_type}, bagging={use_bagging}, use_rfe={use_rfe}")

    # Additional validation for invalid values in features
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
    
    # CRITICAL: Clean target variable (y_train) for NaN and inf values
    if y_train.isna().any():
        logging.warning(f"Found {y_train.isna().sum()} NaN values in target variable (y_train)")
        valid_mask = ~y_train.isna()
        y_train = y_train[valid_mask]
        X_train = X_train.loc[y_train.index]
        logging.info(f"DEBUG: After y_train NaN removal - X_train: {X_train.shape}, y_train: {len(y_train)}")
    
    if np.isinf(y_train).any():
        logging.warning(f"Found {np.isinf(y_train).sum()} infinite values in target variable (y_train)")
        valid_mask = ~np.isinf(y_train)
        y_train = y_train[valid_mask]
        X_train = X_train.loc[y_train.index]
        logging.info(f"DEBUG: After y_train inf removal - X_train: {X_train.shape}, y_train: {len(y_train)}")
    
    # Final validation
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError(f"No valid training data after cleaning! X_train: {len(X_train)}, y_train: {len(y_train)}")
    
    if len(X_train) != len(y_train):
        raise ValueError(f"Shape mismatch after cleaning: X_train: {len(X_train)}, y_train: {len(y_train)}")

    # Initialize regressor based on model type
    if model_type == "gradient_boosting":
        base_regressor = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        rf_regressor = base_regressor
    elif use_bagging:
        base_regressor = RandomForestRegressor(
            n_estimators=1, 
            criterion='squared_error',
            random_state=42
        )
        rf_regressor = BaggingRegressor(
            estimator=base_regressor,
            n_estimators=bagging_estimators,
            max_features=1.0,
            bootstrap=False,
            n_jobs=n_jobs,
            random_state=42
        )
    else:
        rf_regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion='squared_error',
            random_state=42,
            n_jobs=n_jobs
        )

    # FINAL validation before feature selection - ensure absolutely no NaN/inf
    logging.info(f"DEBUG: Pre-feature-selection validation - X_train: {X_train.shape}, y_train: {len(y_train)}")
    
    # Check X_train one more time
    if X_train.isna().any().any():
        nan_count = X_train.isna().sum().sum()
        logging.error(f"CRITICAL: Found {nan_count} NaN in X_train before feature selection")
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        logging.info(f"DEBUG: After final X_train cleaning: {X_train.shape}, y_train: {len(y_train)}")
    
    if np.isinf(X_train.values).any():
        inf_count = np.isinf(X_train.values).sum()
        logging.error(f"CRITICAL: Found {inf_count} inf in X_train before feature selection")
        X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
        y_train = y_train.loc[X_train.index]
        logging.info(f"DEBUG: After final X_train inf cleaning: {X_train.shape}, y_train: {len(y_train)}")
    
    # Check y_train one more time
    if y_train.isna().any():
        nan_count = y_train.isna().sum()
        logging.error(f"CRITICAL: Found {nan_count} NaN in y_train before feature selection")
        valid_mask = ~y_train.isna()
        y_train = y_train[valid_mask]
        X_train = X_train.loc[y_train.index]
        logging.info(f"DEBUG: After final y_train NaN cleaning: {X_train.shape}, y_train: {len(y_train)}")
    
    if np.isinf(y_train.values).any():
        inf_count = np.isinf(y_train.values).sum()
        logging.error(f"CRITICAL: Found {inf_count} inf in y_train before feature selection")
        valid_mask = ~np.isinf(y_train.values)
        y_train = y_train[valid_mask]
        X_train = X_train.loc[y_train.index]
        logging.info(f"DEBUG: After final y_train inf cleaning: {X_train.shape}, y_train: {len(y_train)}")
    
    # Verify we still have data
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError(f"No data left after final cleaning! X_train: {len(X_train)}, y_train: {len(y_train)}")
    
    logging.info(f"DEBUG: Data validated for feature selection - X_train: {X_train.shape}, y_train: {len(y_train)}")
    logging.info(f"DEBUG: X_train has NaN: {X_train.isna().any().any()}, has inf: {np.isinf(X_train.values).any()}")
    logging.info(f"DEBUG: y_train has NaN: {y_train.isna().any()}, has inf: {np.isinf(y_train.values).any()}")
    
    # Feature selection
    if use_rfe:
        rfe_estimators = config.get("rfe", {}).get("n_estimators", 50)
        rfe_features = config.get("rfe", {}).get("n_features", 20)
        logging.info(f"RFE settings: estimators={rfe_estimators}, features={rfe_features}")
        
        rfe_regressor = RandomForestRegressor(
            n_estimators=rfe_estimators,
            criterion='squared_error',
            random_state=42,
            n_jobs=n_jobs
        )
        rfe = RFE(estimator=rfe_regressor, n_features_to_select=rfe_features)
        
        # Ensure y_train is 1D array without NaN
        y_train_array = y_train.values.ravel()
        logging.info(f"DEBUG: y_train_array for RFE - shape: {y_train_array.shape}, has NaN: {np.isnan(y_train_array).any()}")
        
        rfe.fit(X_train, y_train_array)
        selected_features = X_train.columns[rfe.support_]
    else:
        # Use importance-based feature selection with family grouping
        rf_temp = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=n_jobs)
        
        # Ensure y_train is clean for fitting
        y_train_array = y_train.values.ravel()
        logging.info(f"DEBUG: y_train_array for feature selection - shape: {y_train_array.shape}, has NaN: {np.isnan(y_train_array).any()}")
        
        rf_temp.fit(X_train, y_train_array)
        
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
    logging.info(f"Target statistics: Mean={y_train.mean():.6f}, Median={y_train.median():.6f}, Std={y_train.std():.6f}")
    
    # Final check before training
    y_train_final = y_train.values.ravel()
    logging.info(f"DEBUG: Final training - y_train shape: {y_train_final.shape}, has NaN: {np.isnan(y_train_final).any()}")
    
    rf_regressor.fit(X_train_selected, y_train_final)
    
    return rf_regressor, selected_features


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
        y_pred = model.predict(X_test_selected)
        logging.info(f"DEBUG PREDICT: Successfully predicted {len(y_pred)} samples")
        return pd.Series(y_pred, index=X_test_selected.index)
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
    Walk-forward regression training for forex profit prediction
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
    
    # Data directories (where generate.py saves outputs)
    signals_dir = base_results_dir / "signals"
    data_dir = base_results_dir / "data"  # Features and labels are here now
    
    # Training experiment directories (where regression.py saves outputs)
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
    log_file = setup_logging(base_results_dir / "logs", f"regression_{experiment}")
    logging.info(f"Starting walk-forward regression training experiment: {experiment}")
    logging.info(f"Config: {config_name}")
    logging.info(f"Data directory: {data_dir}")
    logging.info(f"Models directory: {models_dir}")
    logging.info(f"Results directory: {results_dir}")
    
    # Copy config for reference
    shutil.copy(config, results_dir / "config.yaml")
    
    # Save experiment metadata
    experiment_metadata = {
        'experiment_name': experiment,
        'model_type': 'regression',
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
    
    # Add training config to metadata
    experiment_metadata['training_config'] = settings.get('training', {})
    
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
    all_profits = pd.Series(dtype=float)  # Training target (actual profits)
    all_predictions = pd.Series(dtype=float)  # Predicted profits
    all_test_profits = pd.Series(dtype=float)  # True profits for test data
    
    # Track processing statistics
    dates_processed = []
    dates_skipped = []
    
    # Load training configuration from YAML, with defaults as fallback
    training_config = settings.get('training', {})
    
    # Apply defaults for any missing values
    training_config.setdefault('n_estimators', 100)
    training_config.setdefault('use_rfe', False)
    training_config.setdefault('max_features', 30)
    training_config.setdefault('max_features_per_family', 3)
    training_config.setdefault('bagging', {'enabled': False, 'n_estimators': 100})
    training_config.setdefault('reverse_signals', False)
    training_config.setdefault('model_type', 'random_forest')  # random_forest or gradient_boosting
    
    # Extract reverse_signals flag for easy access
    reverse_signals = training_config.get('reverse_signals', False)
    
    if reverse_signals:
        logging.info("=" * 60)
        logging.info("SIGNAL REVERSAL ENABLED")
        logging.info("All signals will be reversed: Long->Short, Short->Long")
        logging.info("Profits will be flipped accordingly")
        logging.info("=" * 60)
    
    logging.info(f"Training configuration: {training_config}")
    
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
            
            # Reverse signals if configured (for testing if signal has inverse predictive power)
            if reverse_signals:
                if 'signal' in signals.columns:
                    original_signals = signals['signal'].copy()
                    signals['signal'] = -signals['signal']
                    logging.info(f"DEBUG: Reversed signals (original range: [{original_signals.min()}, {original_signals.max()}] -> reversed: [{signals['signal'].min()}, {signals['signal'].max()}])")
                else:
                    # If signal column has different name, reverse first column
                    original_signals = signals.iloc[:, 0].copy()
                    signals.iloc[:, 0] = -signals.iloc[:, 0]
                    logging.info(f"DEBUG: Reversed signals in first column")
                
                # IMPORTANT: Also reverse the profits since we're trading the opposite direction
                if 'profit' in labels_df.columns:
                    original_profit_mean = labels_df['profit'].mean()
                    labels_df['profit'] = -labels_df['profit']
                    logging.info(f"DEBUG: Reversed profits (original mean: {original_profit_mean:.6f} -> reversed mean: {labels_df['profit'].mean():.6f})")
                
                # Also reverse the label (1 becomes -1, -1 becomes 1)
                if 'label' in labels_df.columns:
                    labels_df['label'] = -labels_df['label']
                    logging.info(f"DEBUG: Reversed labels (1<->-1)")
            
            # Filter to only signal bars (where signal != 0)
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
            
            # If signals are reversed, also reverse signal-adjusted features
            if reverse_signals:
                adj_features = ['returns_adj', 'log_returns_adj', 'ma5_distance_adj', 'ma20_distance_adj']
                for adj_feat in adj_features:
                    if adj_feat in features.columns:
                        features[adj_feat] = -features[adj_feat]
                        logging.info(f"DEBUG: Reversed feature {adj_feat}")
            
            logging.info(f"DEBUG: After signal filtering - signals: {signals.shape}, features: {features.shape}, labels_df: {labels_df.shape}")
            
            # Clean and align data
            initial_features_len = len(features)
            features = features.dropna()
            logging.info(f"DEBUG: Features after dropna: {features.shape} (removed {initial_features_len - len(features)})")
            
            # Align labels with features
            labels_df = labels_df.loc[features.index]
            logging.info(f"DEBUG: Labels after alignment: {labels_df.shape}")
            
            if len(features) == 0:
                logging.warning(f"No valid features for {date_str}")
                dates_skipped.append(date_str)
                continue
            
            # Extract profit values (regression target)
            if 'profit' not in labels_df.columns:
                logging.error(f"Label file missing 'profit' column for {date_str}")
                logging.error(f"Available columns: {list(labels_df.columns)}")
                dates_skipped.append(date_str)
                continue
            
            profits = labels_df['profit']
            
            logging.info(f"DEBUG: Profit statistics - Mean: {profits.mean():.6f}, Median: {profits.median():.6f}, Std: {profits.std():.6f}")
            logging.info(f"DEBUG: Final aligned data - features: {features.shape}, profits: {len(profits)}")
            
            # Verify indices match
            if not features.index.equals(profits.index):
                logging.error(f"DEBUG: Index mismatch after processing!")
                logging.error(f"Features index: {features.index[:5]}...")
                logging.error(f"Profits index: {profits.index[:5]}...")
                dates_skipped.append(date_str)
                continue
            
            logging.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
            dates_processed.append(date_str)
            
            # Make predictions if model exists
            if model is not None and selected_features is not None:
                logging.info(f"DEBUG: Making predictions for {date_str}")
                
                predictions = predict_model(model, features, selected_features)
                
                if len(predictions) > 0:
                    logging.info(f"DEBUG: Received {len(predictions)} predictions")
                    
                    # Align test profits with the actual prediction indices
                    profits_aligned = profits.loc[predictions.index]
                    
                    logging.info(f"DEBUG: Aligned data - predictions: {len(predictions)}, test_profits: {len(profits_aligned)}")
                    
                    all_predictions = pd.concat([all_predictions, predictions])
                    all_test_profits = pd.concat([all_test_profits, profits_aligned])
                    
                    logging.info(f"DEBUG: Total accumulated - predictions: {len(all_predictions)}, test_profits: {len(all_test_profits)}")
                    
                    # Validate data before calculating metrics
                    if all_predictions.isna().any():
                        logging.warning(f"Found {all_predictions.isna().sum()} NaN in accumulated predictions")
                        all_predictions = all_predictions.dropna()
                        all_test_profits = all_test_profits.loc[all_predictions.index]
                        logging.info(f"DEBUG: After cleaning predictions - predictions: {len(all_predictions)}, test_profits: {len(all_test_profits)}")
                    
                    if all_test_profits.isna().any():
                        logging.warning(f"Found {all_test_profits.isna().sum()} NaN in accumulated test_profits")
                        all_test_profits = all_test_profits.dropna()
                        all_predictions = all_predictions.loc[all_test_profits.index]
                        logging.info(f"DEBUG: After cleaning test_profits - predictions: {len(all_predictions)}, test_profits: {len(all_test_profits)}")
                    
                    # Only calculate metrics if we have valid data
                    if len(all_predictions) > 0 and len(all_test_profits) > 0:
                        # Evaluate current performance
                        try:
                            mse = mean_squared_error(all_test_profits, all_predictions)
                            mae = mean_absolute_error(all_test_profits, all_predictions)
                            r2 = r2_score(all_test_profits, all_predictions)
                        except Exception as e:
                            logging.error(f"Error calculating metrics: {e}")
                            logging.error(f"all_predictions has NaN: {all_predictions.isna().any()}, all_test_profits has NaN: {all_test_profits.isna().any()}")
                            mse = mae = r2 = 0.0
                        
                        # Trading metrics (take trades with predicted profit > 0)
                        positive_pred_mask = all_predictions > 0
                        if positive_pred_mask.sum() > 0:
                            actual_profit_on_positive = all_test_profits[positive_pred_mask].sum()
                            mean_actual_profit = all_test_profits[positive_pred_mask].mean()
                            trade_count = positive_pred_mask.sum()
                            win_rate = (all_test_profits[positive_pred_mask] > 0).mean()
                        else:
                            actual_profit_on_positive = 0
                            mean_actual_profit = 0
                            trade_count = 0
                            win_rate = 0
                        
                        logging.info(f"Current performance:")
                        logging.info(f"  MSE: {mse:.8f}, MAE: {mae:.6f}, R2: {r2:.4f}")
                        logging.info(f"  Trades (pred>0): {trade_count}, Total Profit: {actual_profit_on_positive:.6f}")
                        logging.info(f"  Mean Profit: {mean_actual_profit:.6f}, Win Rate: {win_rate:.3f}")
                    else:
                        logging.warning(f"No valid data for metrics calculation")
            
            # Accumulate training data
            logging.info(f"DEBUG: Before accumulation - all_features: {all_features.shape}, all_profits: {len(all_profits)}")
            
            # Check for duplicate indices before accumulating
            if len(all_features) > 0:
                duplicate_indices = all_features.index.intersection(features.index)
                if len(duplicate_indices) > 0:
                    logging.warning(f"DEBUG: Found {len(duplicate_indices)} duplicate indices, removing from new data")
                    features = features.loc[~features.index.isin(duplicate_indices)]
                    profits = profits.loc[~profits.index.isin(duplicate_indices)]
                    logging.info(f"DEBUG: After duplicate removal - features: {features.shape}, profits: {len(profits)}")
            
            all_features = pd.concat([all_features, features])
            all_profits = pd.concat([all_profits, profits])
            
            logging.info(f"DEBUG: After accumulation - all_features: {all_features.shape}, all_profits: {len(all_profits)}")
            
            # Retrain model if conditions are met
            # Use i (day index) for min_train_days check, not len(all_features) (sample count)
            should_retrain = (
                (i % retrain_frequency == 0 and i >= min_train_days) or
                (model is None and i >= min_train_days)
            )
            
            if should_retrain:
                logging.info(f"Retraining model with {len(all_features)} samples from {i+1} days")
                
                try:
                    model, selected_features = train_model(
                        all_features, 
                        all_profits, 
                        training_config, 
                        n_jobs
                    )
                    
                    logging.info(f"DEBUG TRAINING: Model trained successfully")
                    logging.info(f"DEBUG TRAINING: Selected features: {len(selected_features)}")
                    
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
        
        # Clean accumulated predictions and test profits before final evaluation
        logging.info(f"DEBUG: Pre-cleaning - predictions: {len(all_predictions)}, test_profits: {len(all_test_profits)}")
        logging.info(f"DEBUG: Predictions has NaN: {all_predictions.isna().any()}, Test profits has NaN: {all_test_profits.isna().any()}")
        
        if all_predictions.isna().any():
            logging.warning(f"Found {all_predictions.isna().sum()} NaN in final predictions, removing...")
            all_predictions = all_predictions.dropna()
            all_test_profits = all_test_profits.loc[all_predictions.index]
        
        if all_test_profits.isna().any():
            logging.warning(f"Found {all_test_profits.isna().sum()} NaN in final test profits, removing...")
            all_test_profits = all_test_profits.dropna()
            all_predictions = all_predictions.loc[all_test_profits.index]
        
        logging.info(f"DEBUG: Post-cleaning - predictions: {len(all_predictions)}, test_profits: {len(all_test_profits)}")
        
        if len(all_predictions) == 0 or len(all_test_profits) == 0:
            logging.error("No valid predictions after cleaning! Cannot generate results.")
        else:
            # Create detailed predictions dataframe
            predictions_df = pd.DataFrame({
                'predicted_profit': all_predictions,
                'actual_profit': all_test_profits
            })
            
            # Add additional columns
            predictions_df['trade_date'] = predictions_df.index.date
            predictions_df['prediction_error'] = predictions_df['actual_profit'] - predictions_df['predicted_profit']
            predictions_df['abs_error'] = predictions_df['prediction_error'].abs()
            
            # Calculate overall regression metrics
            try:
                mse = mean_squared_error(all_test_profits, all_predictions)
                mae = mean_absolute_error(all_test_profits, all_predictions)
                r2 = r2_score(all_test_profits, all_predictions)
                rmse = np.sqrt(mse)
            except Exception as e:
                logging.error(f"Error calculating final metrics: {e}")
                mse = mae = r2 = rmse = 0.0
        
        logging.info("="*60)
        logging.info("REGRESSION MODEL PERFORMANCE")
        logging.info("="*60)
        logging.info(f"Mean Squared Error (MSE): {mse:.8f}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        logging.info(f"Mean Absolute Error (MAE): {mae:.6f}")
        logging.info(f"R-squared (R2): {r2:.4f}")
        logging.info("-"*60)
        
        # Evaluate across multiple profit thresholds for trading decisions
        # Include -inf to show ALL predictions (baseline: take all signals regardless of predicted profit)
        # This is equivalent to the classification approach where all signals are traded
        thresholds = [-np.inf, 0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
        
        threshold_results = []
        
        for threshold in thresholds:
            # Take trades where predicted profit > threshold
            trade_mask = all_predictions >= threshold
            
            if trade_mask.sum() > 0:
                trade_count = trade_mask.sum()
                total_profit = all_test_profits[trade_mask].sum()
                mean_profit = all_test_profits[trade_mask].mean()
                median_profit = all_test_profits[trade_mask].median()
                win_rate = (all_test_profits[trade_mask] > 0).mean()
                
                # Prediction accuracy for selected trades
                try:
                    selected_mse = mean_squared_error(all_test_profits[trade_mask], all_predictions[trade_mask])
                    selected_mae = mean_absolute_error(all_test_profits[trade_mask], all_predictions[trade_mask])
                except Exception as e:
                    logging.warning(f"Error calculating metrics for threshold {threshold}: {e}")
                    selected_mse = 0.0
                    selected_mae = 0.0
            else:
                trade_count = 0
                total_profit = 0
                mean_profit = 0
                median_profit = 0
                win_rate = 0
                selected_mse = 0
                selected_mae = 0
            
            threshold_results.append({
                'threshold': threshold,
                'count': trade_count,
                'total_profit': total_profit,
                'mean_profit': mean_profit,
                'median_profit': median_profit,
                'win_rate': win_rate,
                'mse': selected_mse,
                'mae': selected_mae
            })
        
        # Create results DataFrame
        final_results = pd.DataFrame(threshold_results)
        
        # Save results
        final_results.to_csv(results_dir / "evaluation_results.csv", index=False)
        final_results.to_parquet(results_dir / "evaluation_results.parquet")
        final_results.to_parquet(results_dir / "results.parquet")
        
        # Save all predictions
        predictions_df.to_parquet(results_dir / "all_predictions.parquet")
        predictions_df.to_parquet(results_dir / "predicted_results.parquet")
        
        # Daily profit summary
        daily_profit = predictions_df.groupby('trade_date')['actual_profit'].sum()
        daily_profit.to_csv(results_dir / "daily.csv.bz2")
        
        # Print detailed summary
        best_profit_row = final_results.loc[final_results['total_profit'].idxmax()]
        
        logging.info("="*60)
        logging.info("TRADING PERFORMANCE SUMMARY")
        logging.info("="*60)
        logging.info(f"Total predictions made: {len(all_predictions)}")
        logging.info(f"Total profit (all pred>0): {all_test_profits[all_predictions > 0].sum():.6f}")
        logging.info(f"Mean predicted profit: {all_predictions.mean():.6f}")
        logging.info(f"Mean actual profit (all): {all_test_profits.mean():.6f}")
        logging.info("-"*60)
        logging.info(f"Best Total Profit: {best_profit_row['total_profit']:.6f} at threshold {best_profit_row['threshold']:.4f}")
        logging.info(f"  - Trades: {best_profit_row['count']}, Win Rate: {best_profit_row['win_rate']:.3f}")
        logging.info(f"  - Mean Profit: {best_profit_row['mean_profit']:.6f}")
        logging.info("-"*60)
        
        # Show top performing thresholds
        top_profit_thresholds = final_results.nlargest(5, 'total_profit')
        logging.info("Top 5 thresholds by total profit:")
        for _, row in top_profit_thresholds.iterrows():
            logging.info(f"  {row['threshold']:.4f}: Profit={row['total_profit']:.6f}, "
                        f"Trades={row['count']}, Mean={row['mean_profit']:.6f}, WinRate={row['win_rate']:.3f}")
        
        logging.info("-"*60)
        logging.info(f"Results saved to: {results_dir}")
        
        # Print complete results table
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
                # Get feature importances
                if isinstance(model, BaggingRegressor):
                    logging.info("Extracting feature importances from BaggingRegressor...")
                    importances_list = []
                    for estimator in model.estimators_:
                        if hasattr(estimator, 'feature_importances_'):
                            importances_list.append(estimator.feature_importances_)
                    
                    if importances_list:
                        feature_importances = np.mean(importances_list, axis=0)
                        logging.info(f"Aggregated importances from {len(importances_list)} estimators")
                    else:
                        logging.warning("No feature importances found in base estimators")
                        feature_importances = None
                elif isinstance(model, GradientBoostingRegressor):
                    feature_importances = model.feature_importances_
                else:
                    feature_importances = model.feature_importances_
                
                if feature_importances is not None:
                    importance_df = pd.DataFrame({
                        'feature': selected_features,
                        'importance': feature_importances
                    }).sort_values('importance', ascending=False)
                    
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
                    
                    # Group by family
                    family_importance = importance_df.groupby('family')['importance'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
                    
                    logging.info("\n" + "-"*80)
                    logging.info("Feature Importance by Family:")
                    logging.info("-"*80)
                    logging.info(f"{'Family':<20} {'Total Imp':<15} {'Mean Imp':<15} {'Count':<10}")
                    logging.info("-"*80)
                    
                    for family, row in family_importance.iterrows():
                        logging.info(f"{family:<20} {row['sum']:<15.6f} {row['mean']:<15.6f} {int(row['count']):<10}")
                    
                    logging.info("-"*80)
                
            except Exception as e:
                logging.error(f"Error calculating feature importance: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
    else:
        logging.warning("No predictions were made. Check your data and configuration.")


if __name__ == '__main__':
    main()
