
# File: experiments/tokyo/run_experiment.py

import pandas as pd, numpy as np, json, yaml, os
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
import lightgbm as lgb
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt

# Load raw payload and metadata
ARTIFACTS_PATH = Path('./artifacts')
RAW_PATH = Path('./data/raw/usgs_events')
META_PATH = Path('./data/metadata')
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

window_hash = sorted([f.stem for f in RAW_PATH.glob('*.json')])[-1]
with open(RAW_PATH / f'{window_hash}.json','r') as f: data = json.load(f)
with open(META_PATH / f'{window_hash}.yaml','r') as f: meta = yaml.safe_load(f)

# Parse features
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

# Daily aggregation
start = df['date'].min()
end = df['date'].max()
dates = pd.date_range(start, end, freq='D')
daily = df.groupby('date').agg(
    count=('id','count'),
    mean_mag=('mag','mean'),
    max_mag=('mag','max'),
    mean_depth=('depth_km','mean'),
).reindex(dates.date, fill_value=0)
daily.index = pd.to_datetime(daily.index)

# Multi-horizon labels
for horizon in [7,14,30]:
    label = []
    for i in range(len(daily)):
        window = daily.iloc[i+1:i+1+horizon]
        label.append(int((window['max_mag'] >= 5.0).any()))
    daily[f'label_next{horizon}'] = label + [0]*(len(daily)-len(label))

# Feature engineering
daily['count_7d'] = daily['count'].rolling(7).sum().shift(1).fillna(0)
daily['mean_mag_30d'] = daily['mean_mag'].rolling(30).mean().shift(1).fillna(0)
daily['max_mag_7d'] = daily['max_mag'].rolling(7).max().shift(1).fillna(0)

# Cumulative seismic moment
def Mw_to_M0(Mw): return 10 ** (1.5 * (Mw + 6.0))
df['M0'] = df['mag'].apply(Mw_to_M0)
df['logM0'] = np.log10(df['M0'])
df['day'] = pd.to_datetime(df['date'])
daily['cum_logM0_30d'] = df.groupby('day')['logM0'].sum().rolling(30).sum().shift(1).reindex(daily.index).fillna(0)

# Distance to recent M≥5 event
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

distances = []
for date in daily.index:
    past_events = df[(df['day'] < date) & (df['mag'] >= 5.0) & (df['day'] >= date - timedelta(days=90))]
    if past_events.empty:
        distances.append(500.0)
    else:
        dists = [haversine(lat1, lon1, lat2, lon2)
                 for lat1, lon1 in zip(df[df['day']==date]['lat'], df[df['day']==date]['lon'])
                 for lat2, lon2 in zip(past_events['lat'], past_events['lon'])]
        distances.append(min(dists) if dists else 500.0)
daily['dist_to_recent_large'] = distances

# Modeling
features = ['count_7d','mean_mag_30d','max_mag_7d','mean_depth','cum_logM0_30d','dist_to_recent_large']
train_cutoff = pd.to_datetime('2025-06-30')
train = daily[daily.index <= train_cutoff]
test = daily[daily.index > train_cutoff]

metrics = {}
for horizon in [7,14,30]:
    y_train = train[f'label_next{horizon}']
    y_test = test[f'label_next{horizon}']
    X_train = train[features]
    X_test = test[features]

    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:,1]
    pred = (prob >= 0.5).astype(int)

    metrics[f'logreg_next{horizon}'] = {
        'precision': precision_score(y_test, pred, zero_division=0),
        'recall': recall_score(y_test, pred, zero_division=0),
        'f1': f1_score(y_test, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, prob) if len(set(y_test))>1 else None,
        'brier': brier_score_loss(y_test, prob),
    }

# Save artifacts
daily.to_csv(ARTIFACTS_PATH / 'daily_features_labels.csv', index=False)
with open(ARTIFACTS_PATH / 'metrics.json','w') as f: json.dump(metrics,f,indent=2)
with open(ARTIFACTS_PATH / 'data_snapshot.yaml','w') as f: yaml.safe_dump(meta,f)
