"""
Tokyo Multi-Horizon Earthquake Prediction Experiment

This script implements an enhanced earthquake prediction pipeline with:
- Multi-horizon labeling (7d, 14d, 30d)
- Class-weighted models to handle imbalance
- Empirical features (distance to recent large events, cumulative seismic moment)
- Comprehensive evaluation metrics and visualizations
"""

import pandas as pd
import numpy as np
import json
import yaml
import os
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, brier_score_loss, classification_report
)
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Will use Logistic Regression only.")


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r


def calculate_seismic_moment(magnitude):
    """
    Calculate seismic moment from magnitude using Hanks & Kanamori (1979).
    M0 = 10^(1.5 * Mw + 9.1) dyne·cm
    Returns log10(M0)
    """
    return 1.5 * magnitude + 9.1


def run_tokyo_experiment():
    """
    Main experiment runner for Tokyo earthquake prediction.
    """
    print("=" * 80)
    print("TOKYO MULTI-HORIZON EARTHQUAKE PREDICTION EXPERIMENT")
    print("=" * 80)
    
    # Set up paths
    ARTIFACTS_PATH = Path('./artifacts')
    RAW_PATH = Path('./data/raw/usgs_events')
    META_PATH = Path('./data/metadata')
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/8] Loading earthquake data...")
    
    # Find the most recent data file
    json_files = sorted(RAW_PATH.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {RAW_PATH}")
    
    window_hash = json_files[-1].stem
    print(f"  Using data file: {window_hash}.json")
    
    with open(RAW_PATH / f'{window_hash}.json', 'r') as f:
        data = json.load(f)
    
    # Load metadata if available
    meta_file = META_PATH / f'{window_hash}.yaml'
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
        print(f"  Metadata loaded: {meta.get('feature_count', 'N/A')} events")
    
    # Parse earthquake features
    print("\n[2/8] Parsing earthquake features...")
    rows = []
    for feat in data['features']:
        prop = feat['properties']
        coords = feat['geometry']['coordinates']
        rows.append({
            'id': feat['id'],
            'time': datetime.utcfromtimestamp(prop['time']/1000.0),
            'mag': prop['mag'],
            'depth_km': coords[2],
            'lon': coords[0],
            'lat': coords[1],
        })
    
    df = pd.DataFrame(rows).dropna()
    df['date'] = df['time'].dt.date
    df = df.sort_values('time')
    
    print(f"  Total events: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Magnitude range: {df['mag'].min():.1f} to {df['mag'].max():.1f}")
    
    # Daily aggregation
    print("\n[3/8] Creating daily aggregations...")
    start = df['date'].min()
    end = df['date'].max()
    dates = pd.date_range(start, end, freq='D')
    
    daily = df.groupby('date').agg(
        count=('id', 'count'),
        mean_mag=('mag', 'mean'),
        max_mag=('mag', 'max'),
        mean_depth=('depth_km', 'mean'),
    ).reindex(dates.date, fill_value=0)
    daily.index = pd.to_datetime(daily.index)
    
    # Enhanced feature engineering
    print("\n[4/8] Engineering empirical features...")
    
    # Temporal features
    for window in [7, 30, 90]:
        daily[f'count_{window}d'] = daily['count'].rolling(window, min_periods=1).sum()
        daily[f'mean_mag_{window}d'] = daily['mean_mag'].rolling(window, min_periods=1).mean()
        daily[f'max_mag_{window}d'] = daily['max_mag'].rolling(window, min_periods=1).max()
    
    # Cumulative seismic moment (log scale)
    df['seismic_moment_log'] = df['mag'].apply(calculate_seismic_moment)
    moment_daily = df.groupby('date')['seismic_moment_log'].sum()
    daily['cumulative_moment_30d'] = moment_daily.reindex(
        daily.index.date, fill_value=0
    ).rolling(30, min_periods=1).sum().values
    
    # Distance to nearest large event (M >= 5.0)
    large_events = df[df['mag'] >= 5.0].copy()
    tokyo_center_lat = 35.6762
    tokyo_center_lon = 139.6503
    
    daily['dist_to_large_event_90d'] = np.nan
    
    for i, date in enumerate(daily.index):
        # Get large events in past 90 days
        past_90_days = date - timedelta(days=90)
        recent_large = large_events[
            (large_events['time'] >= past_90_days) & 
            (large_events['time'] < date)
        ]
        
        if len(recent_large) > 0:
            # Calculate distance to nearest large event from Tokyo center
            distances = recent_large.apply(
                lambda row: haversine_distance(
                    tokyo_center_lon, tokyo_center_lat, 
                    row['lon'], row['lat']
                ),
                axis=1
            )
            daily.loc[date, 'dist_to_large_event_90d'] = distances.min()
        else:
            daily.loc[date, 'dist_to_large_event_90d'] = 999.0  # Large value if no events
    
    # Multi-horizon labeling
    print("\n[5/8] Creating multi-horizon labels...")
    
    horizons = [7, 14, 30]
    for horizon in horizons:
        daily[f'label_{horizon}d'] = 0
        for i in range(len(daily) - horizon):
            future_window = daily.iloc[i+1:i+1+horizon]
            if (future_window['max_mag'] >= 5.0).any():
                daily.loc[daily.index[i], f'label_{horizon}d'] = 1
    
    # Report label distribution
    for horizon in horizons:
        pos_count = daily[f'label_{horizon}d'].sum()
        total = len(daily[f'label_{horizon}d'].dropna())
        print(f"  {horizon}d horizon: {pos_count} positive / {total} total ({pos_count/total*100:.1f}%)")
    
    # Train/test split (80/20)
    print("\n[6/8] Splitting train/test sets...")
    
    split_idx = int(len(daily) * 0.8)
    train = daily.iloc[:split_idx].copy()
    test = daily.iloc[split_idx:].copy()
    
    print(f"  Train: {len(train)} days ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"  Test: {len(test)} days ({test.index[0].date()} to {test.index[-1].date()})")
    
    # Feature columns
    feature_cols = [
        'count_7d', 'count_30d', 'count_90d',
        'mean_mag_7d', 'mean_mag_30d', 'mean_mag_90d',
        'max_mag_7d', 'max_mag_30d', 'max_mag_90d',
        'cumulative_moment_30d',
        'dist_to_large_event_90d'
    ]
    
    # Model training and evaluation
    print("\n[7/8] Training models...")
    
    results = {}
    
    for horizon in horizons:
        print(f"\n  === {horizon}-day horizon ===")
        
        # Prepare data
        X_train = train[feature_cols].fillna(0)
        y_train = train[f'label_{horizon}d']
        X_test = test[feature_cols].fillna(0)
        y_test = test[f'label_{horizon}d']
        
        # Remove rows with NaN labels
        train_mask = ~y_train.isna()
        test_mask = ~y_test.isna()
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        # Logistic Regression with class weighting
        print("    Training Logistic Regression...")
        lr = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        lr.fit(X_train, y_train)
        
        y_pred_lr = lr.predict(X_test)
        y_prob_lr = lr.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics_lr = {
            'precision': precision_score(y_test, y_pred_lr, zero_division=0),
            'recall': recall_score(y_test, y_pred_lr, zero_division=0),
            'f1': f1_score(y_test, y_pred_lr, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob_lr) if len(np.unique(y_test)) > 1 else 0,
            'brier': brier_score_loss(y_test, y_prob_lr)
        }
        
        print(f"    Precision: {metrics_lr['precision']:.3f}")
        print(f"    Recall: {metrics_lr['recall']:.3f}")
        print(f"    F1: {metrics_lr['f1']:.3f}")
        print(f"    ROC-AUC: {metrics_lr['roc_auc']:.3f}")
        print(f"    Brier: {metrics_lr['brier']:.3f}")
        
        results[f'{horizon}d'] = {
            'logistic_regression': metrics_lr,
            'y_test': y_test,
            'y_pred_lr': y_pred_lr,
            'y_prob_lr': y_prob_lr
        }
        
        # LightGBM if available
        if LIGHTGBM_AVAILABLE:
            print("    Training LightGBM...")
            
            # Calculate scale_pos_weight
            scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
            
            lgb_model = lgb.LGBMClassifier(
                scale_pos_weight=scale_pos_weight,
                n_estimators=100,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            
            y_pred_lgb = lgb_model.predict(X_test)
            y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
            
            metrics_lgb = {
                'precision': precision_score(y_test, y_pred_lgb, zero_division=0),
                'recall': recall_score(y_test, y_pred_lgb, zero_division=0),
                'f1': f1_score(y_test, y_pred_lgb, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob_lgb) if len(np.unique(y_test)) > 1 else 0,
                'brier': brier_score_loss(y_test, y_prob_lgb)
            }
            
            print(f"    [LightGBM] Precision: {metrics_lgb['precision']:.3f}")
            print(f"    [LightGBM] Recall: {metrics_lgb['recall']:.3f}")
            print(f"    [LightGBM] F1: {metrics_lgb['f1']:.3f}")
            print(f"    [LightGBM] ROC-AUC: {metrics_lgb['roc_auc']:.3f}")
            print(f"    [LightGBM] Brier: {metrics_lgb['brier']:.3f}")
            
            results[f'{horizon}d']['lightgbm'] = metrics_lgb
            results[f'{horizon}d']['y_pred_lgb'] = y_pred_lgb
            results[f'{horizon}d']['y_prob_lgb'] = y_prob_lgb
    
    # Save results
    print("\n[8/8] Saving results...")
    
    # Save metrics to JSON
    metrics_file = ARTIFACTS_PATH / 'tokyo_experiment_results.json'
    metrics_summary = {}
    for horizon_key, horizon_data in results.items():
        metrics_summary[horizon_key] = {
            'logistic_regression': horizon_data['logistic_regression']
        }
        if 'lightgbm' in horizon_data:
            metrics_summary[horizon_key]['lightgbm'] = horizon_data['lightgbm']
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"  Results saved to: {metrics_file}")
    
    # Save processed data
    daily_file = ARTIFACTS_PATH / 'tokyo_daily_features.csv'
    daily.to_csv(daily_file)
    print(f"  Daily features saved to: {daily_file}")
    
    # Create visualization
    print("\n  Creating visualization...")
    
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 4*len(horizons)))
    if len(horizons) == 1:
        axes = [axes]
    
    for idx, horizon in enumerate(horizons):
        ax = axes[idx]
        horizon_key = f'{horizon}d'
        
        y_prob = results[horizon_key]['y_prob_lr']
        y_test = results[horizon_key]['y_test']
        
        # Plot predictions over time
        test_dates = test.index[~test[f'label_{horizon}d'].isna()]
        
        ax.plot(test_dates, y_prob, label='Predicted Probability', alpha=0.7)
        ax.scatter(test_dates[y_test == 1], [0.5] * sum(y_test == 1), 
                  color='red', label='Actual M≥5.0 Events', s=100, marker='x')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{horizon}-day Horizon Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = ARTIFACTS_PATH / 'tokyo_experiment_predictions.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved to: {plot_file}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    run_tokyo_experiment()
