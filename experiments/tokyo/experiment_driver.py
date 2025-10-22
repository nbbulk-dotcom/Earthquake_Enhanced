
#!/usr/bin/env python3
"""
GEO_EARTH Tokyo Phase 1 Experiment Driver
Multi-horizon earthquake prediction with empirical features, class weights, and LightGBM.

Features:
- Multi-horizon labels (7d, 14d, 30d)
- Rolling aggregates (count, mean_mag, max_mag)
- Empirical features: cumulative seismic moment (cum_logM0_30d), distance to recent large events
- Class-weighted Logistic Regression
- LightGBM with scale_pos_weight
- TimeSeriesSplit CV for confidence intervals
- Comprehensive metrics: precision, recall, f1, ROC AUC, Brier score, precision@10%, recall@10%
- Calibration plots
- Provenance tracking (SHA256, git commit, timestamps)
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km using haversine formula."""
    R = 6371.0  # Earth radius in km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def Mw_to_M0(Mw):
    """Convert moment magnitude to seismic moment (N·m)."""
    return 10 ** (1.5 * Mw + 9.1)


def load_usgs_data(raw_file):
    """Load and normalize USGS event data."""
    with open(raw_file, "r") as f:
        data = json.load(f)
    
    events = []
    for feature in data.get("features", []):
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        events.append({
            "time": pd.to_datetime(props["time"], unit="ms", utc=True),
            "latitude": coords[1],
            "longitude": coords[0],
            "depth": coords[2],
            "mag": props["mag"],
            "magType": props.get("magType", "unknown"),
            "place": props.get("place", ""),
            "id": props["id"]
        })
    
    df = pd.DataFrame(events)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def create_daily_aggregates(df, start_date, end_date):
    """Create daily aggregates with missing-day fill."""
    df["date"] = df["time"].dt.date
    daily = df.groupby("date").agg({
        "mag": ["count", "mean", "max"],
        "latitude": "mean",
        "longitude": "mean"
    }).reset_index()
    
    daily.columns = ["date", "count", "mean_mag", "max_mag", "mean_lat", "mean_lon"]
    
    # Fill missing days
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    full_daily = pd.DataFrame({"date": date_range.date})
    daily = full_daily.merge(daily, on="date", how="left")
    daily["count"] = daily["count"].fillna(0)
    daily["mean_mag"] = daily["mean_mag"].fillna(daily["mean_mag"].mean())
    daily["max_mag"] = daily["max_mag"].fillna(0)
    daily["mean_lat"] = daily["mean_lat"].fillna(daily["mean_lat"].mean())
    daily["mean_lon"] = daily["mean_lon"].fillna(daily["mean_lon"].mean())
    
    return daily


def create_multi_horizon_labels(df, events_df, horizons=[7, 14, 30], threshold_mag=5.0):
    """Create binary labels for multiple prediction horizons."""
    for horizon in horizons:
        label_col = f"label_{horizon}d"
        df[label_col] = 0
        
        for idx, row in df.iterrows():
            current_date = pd.to_datetime(row["date"])
            future_window = events_df[
                (events_df["time"] > current_date) &
                (events_df["time"] <= current_date + timedelta(days=horizon)) &
                (events_df["mag"] >= threshold_mag)
            ]
            if len(future_window) > 0:
                df.at[idx, label_col] = 1
    
    return df


def add_rolling_features(df, windows=[7, 14, 30]):
    """Add rolling aggregate features."""
    for window in windows:
        df[f"count_{window}d"] = df["count"].rolling(window=window, min_periods=1).sum()
        df[f"mean_mag_{window}d"] = df["mean_mag"].rolling(window=window, min_periods=1).mean()
        df[f"max_mag_{window}d"] = df["max_mag"].rolling(window=window, min_periods=1).max()
    return df


def add_empirical_features(df, events_df):
    """Add empirical features: cumulative seismic moment and distance to recent large events."""
    # Cumulative seismic moment over 30 days
    df["cum_logM0_30d"] = 0.0
    
    # Distance to recent large events (M >= 5.0) in last 90 days
    df["dist_to_recent_large_90d"] = np.inf
    
    for idx, row in df.iterrows():
        current_date = pd.to_datetime(row["date"])
        
        # Cumulative seismic moment (30-day window)
        past_30d = events_df[
            (events_df["time"] > current_date - timedelta(days=30)) &
            (events_df["time"] <= current_date)
        ]
        if len(past_30d) > 0:
            M0_sum = past_30d["mag"].apply(Mw_to_M0).sum()
            df.at[idx, "cum_logM0_30d"] = np.log10(M0_sum) if M0_sum > 0 else 0
        
        # Distance to recent large events (90-day window)
        large_events_90d = events_df[
            (events_df["time"] > current_date - timedelta(days=90)) &
            (events_df["time"] <= current_date) &
            (events_df["mag"] >= 5.0)
        ]
        if len(large_events_90d) > 0:
            distances = [
                haversine_distance(row["mean_lat"], row["mean_lon"], 
                                 ev["latitude"], ev["longitude"])
                for _, ev in large_events_90d.iterrows()
            ]
            df.at[idx, "dist_to_recent_large_90d"] = min(distances)
    
    # Replace inf with a large value (e.g., 1000 km)
    df["dist_to_recent_large_90d"] = df["dist_to_recent_large_90d"].replace(np.inf, 1000.0)
    
    return df


def train_evaluate_model(X_train, y_train, X_test, y_test, model_type="logreg", pos_weight=None):
    """Train and evaluate a model."""
    if model_type == "logreg":
        # Class-weighted Logistic Regression
        class_weights = {0: 1, 1: pos_weight} if pos_weight else None
        model = LogisticRegression(max_iter=1000, class_weight=class_weights, random_state=42)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    elif model_type == "lgbm":
        # LightGBM with scale_pos_weight
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "scale_pos_weight": pos_weight if pos_weight else 1.0,
            "random_state": 42
        }
        model = lgb.train(params, train_data, num_boost_round=100)
        y_pred_proba = model.predict(X_test)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Binary predictions (threshold = 0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
    brier = brier_score_loss(y_test, y_pred_proba)
    
    # Precision and recall at top 10%
    top_10_pct_idx = int(0.1 * len(y_pred_proba))
    top_indices = np.argsort(y_pred_proba)[-top_10_pct_idx:]
    precision_at_10 = y_test.iloc[top_indices].sum() / len(top_indices) if len(top_indices) > 0 else 0.0
    recall_at_10 = y_test.iloc[top_indices].sum() / y_test.sum() if y_test.sum() > 0 else 0.0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "precision_at_10pct": precision_at_10,
        "recall_at_10pct": recall_at_10
    }
    
    return model, y_pred_proba, metrics


def cross_validate_model(X, y, model_type="logreg", n_splits=5, pos_weight=None):
    """Perform time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_metrics = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        _, _, metrics = train_evaluate_model(X_train, y_train, X_test, y_test, 
                                             model_type=model_type, pos_weight=pos_weight)
        cv_metrics.append(metrics)
    
    # Aggregate metrics
    agg_metrics = {}
    for key in cv_metrics[0].keys():
        values = [m[key] for m in cv_metrics]
        agg_metrics[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
    
    return agg_metrics


def plot_calibration_curve(y_true, y_pred_proba, model_name, horizon, output_dir):
    """Plot calibration curve."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10, strategy="uniform"
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve - {model_name} ({horizon}d horizon)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / f"calibration_{model_name.lower().replace(' ', '_')}_{horizon}d.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return str(output_path)


def compute_provenance(raw_file):
    """Compute provenance metadata."""
    # File SHA256
    with open(raw_file, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Git commit
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
    except:
        git_commit = "unknown"
    
    # Timestamps
    timestamp_utc = datetime.utcnow().isoformat() + "Z"
    
    return {
        "raw_file_sha256": file_hash,
        "git_commit": git_commit,
        "timestamp_utc": timestamp_utc,
        "script_version": "1.0.0"
    }


def main():
    parser = argparse.ArgumentParser(description="Tokyo Phase 1 Experiment Driver")
    parser.add_argument("--raw_file", required=True, help="Path to raw USGS events JSON")
    args = parser.parse_args()
    
    # Setup
    raw_file = Path(args.raw_file)
    if not raw_file.exists():
        print(f"Error: Raw file not found: {raw_file}")
        sys.exit(1)
    
    output_dir = Path("experiments/tokyo/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GEO_EARTH Tokyo Phase 1 Experiment")
    print("=" * 80)
    
    # Load data
    print("\n[1/7] Loading USGS data...")
    events_df = load_usgs_data(raw_file)
    print(f"  Loaded {len(events_df)} events")
    print(f"  Date range: {events_df['time'].min()} to {events_df['time'].max()}")
    print(f"  Magnitude range: {events_df['mag'].min():.2f} to {events_df['mag'].max():.2f}")
    
    # Create daily aggregates
    print("\n[2/7] Creating daily aggregates...")
    start_date = events_df["time"].min().date()
    end_date = events_df["time"].max().date()
    daily_df = create_daily_aggregates(events_df, start_date, end_date)
    print(f"  Created {len(daily_df)} daily records")
    
    # Create multi-horizon labels
    print("\n[3/7] Creating multi-horizon labels...")
    horizons = [7, 14, 30]
    daily_df = create_multi_horizon_labels(daily_df, events_df, horizons=horizons, threshold_mag=5.0)
    for horizon in horizons:
        label_col = f"label_{horizon}d"
        pos_count = daily_df[label_col].sum()
        pos_rate = pos_count / len(daily_df) * 100
        print(f"  {horizon}d horizon: {pos_count} positive days ({pos_rate:.2f}%)")
    
    # Add rolling features
    print("\n[4/7] Adding rolling features...")
    daily_df = add_rolling_features(daily_df, windows=[7, 14, 30])
    print(f"  Added rolling features for windows: 7d, 14d, 30d")
    
    # Add empirical features
    print("\n[5/7] Adding empirical features...")
    daily_df = add_empirical_features(daily_df, events_df)
    print(f"  Added cum_logM0_30d and dist_to_recent_large_90d")
    
    # Prepare feature matrix
    feature_cols = [
        "count_7d", "count_14d", "count_30d",
        "mean_mag_7d", "mean_mag_14d", "mean_mag_30d",
        "max_mag_7d", "max_mag_14d", "max_mag_30d",
        "cum_logM0_30d", "dist_to_recent_large_90d"
    ]
    
    # Train and evaluate models
    print("\n[6/7] Training and evaluating models...")
    results = {}
    
    for horizon in horizons:
        print(f"\n  Horizon: {horizon}d")
        label_col = f"label_{horizon}d"
        
        # Prepare data
        X = daily_df[feature_cols].copy()
        y = daily_df[label_col].copy()
        
        # Remove rows with NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Compute class weight
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"    Class distribution: {neg_count} negative, {pos_count} positive")
        print(f"    Positive class weight: {pos_weight:.2f}")
        
        # Split data (80/20 train/test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Logistic Regression
        print(f"    Training Logistic Regression...")
        logreg_model, logreg_proba, logreg_metrics = train_evaluate_model(
            X_train, y_train, X_test, y_test, model_type="logreg", pos_weight=pos_weight
        )
        print(f"      ROC AUC: {logreg_metrics['roc_auc']:.4f}, F1: {logreg_metrics['f1']:.4f}")
        
        # LightGBM
        print(f"    Training LightGBM...")
        lgbm_model, lgbm_proba, lgbm_metrics = train_evaluate_model(
            X_train, y_train, X_test, y_test, model_type="lgbm", pos_weight=pos_weight
        )
        print(f"      ROC AUC: {lgbm_metrics['roc_auc']:.4f}, F1: {lgbm_metrics['f1']:.4f}")
        
        # Cross-validation
        print(f"    Running cross-validation...")
        logreg_cv = cross_validate_model(X, y, model_type="logreg", n_splits=5, pos_weight=pos_weight)
        lgbm_cv = cross_validate_model(X, y, model_type="lgbm", n_splits=5, pos_weight=pos_weight)
        
        # Calibration plots
        logreg_cal_path = plot_calibration_curve(y_test, logreg_proba, "Logistic Regression", horizon, output_dir)
        lgbm_cal_path = plot_calibration_curve(y_test, lgbm_proba, "LightGBM", horizon, output_dir)
        
        # Store results
        results[f"{horizon}d"] = {
            "logreg": {
                "test_metrics": logreg_metrics,
                "cv_metrics": logreg_cv,
                "calibration_plot": logreg_cal_path
            },
            "lgbm": {
                "test_metrics": lgbm_metrics,
                "cv_metrics": lgbm_cv,
                "calibration_plot": lgbm_cal_path
            }
        }
    
    # Save results
    print("\n[7/7] Saving results...")
    provenance = compute_provenance(raw_file)
    
    output_data = {
        "experiment": "Tokyo Phase 1",
        "horizons": horizons,
        "features": feature_cols,
        "models": ["Logistic Regression", "LightGBM"],
        "results": results,
        "provenance": provenance
    }
    
    output_file = output_dir / "phase1_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  Results saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nArtifacts saved to: {output_dir}")
    print(f"  - phase1_results.json")
    for horizon in horizons:
        print(f"  - calibration_logistic_regression_{horizon}d.png")
        print(f"  - calibration_lightgbm_{horizon}d.png")
    
    print("\nBest performing model (by ROC AUC):")
    for horizon in horizons:
        logreg_auc = results[f"{horizon}d"]["logreg"]["test_metrics"]["roc_auc"]
        lgbm_auc = results[f"{horizon}d"]["lgbm"]["test_metrics"]["roc_auc"]
        best_model = "LightGBM" if lgbm_auc > logreg_auc else "Logistic Regression"
        best_auc = max(logreg_auc, lgbm_auc)
        print(f"  {horizon}d horizon: {best_model} (ROC AUC = {best_auc:.4f})")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
