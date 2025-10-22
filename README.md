
# GEO_EARTH Enhanced Earthquake Prediction

Empirical earthquake prediction experiment for the Tokyo region using multi-horizon forecasting, class-weighted models, and spatial triangulation.

## Project Overview

This project implements Phase 1 of an enhanced earthquake prediction system with the following features:

### Region and Time Window
- **Region**: Tokyo area (30.0–38.5°N, 136.0–145.0°E)
- **Time Window**: 2024-10-01 to 2025-09-30
- **Magnitude Threshold**: M ≥ 3.5 (training data)
- **Prediction Target**: Binary classification for M ≥ 5.0 events

### Prediction Horizons
- 7-day forecast
- 14-day forecast
- 30-day forecast

### Models
1. **Class-weighted Logistic Regression**
   - Handles class imbalance with computed weights
   - Baseline interpretable model

2. **LightGBM**
   - Gradient boosting with `scale_pos_weight`
   - Enhanced performance on imbalanced data

### Features

#### Rolling Aggregates
- Event counts (7d, 14d, 30d windows)
- Mean magnitude (7d, 14d, 30d windows)
- Maximum magnitude (7d, 14d, 30d windows)

#### Empirical Features
- **Cumulative Seismic Moment** (`cum_logM0_30d`): Log-transformed sum of seismic moment over 30 days
- **Distance to Recent Large Events** (`dist_to_recent_large_90d`): Minimum distance to M ≥ 5.0 events in last 90 days

### Evaluation Metrics
- Precision, Recall, F1-score
- ROC AUC
- Brier score (calibration)
- Precision @ top 10%
- Recall @ top 10%
- Calibration curves

### Spatial Analysis
- **Delaunay Triangulation**: Spatial tessellation over seismic stations
- **R-tree Fault Indexing**: Efficient nearest-fault queries
- **Triangle-day Aggregates**: Spatiotemporal event clustering

## Directory Structure

```
.
├── backend/
│   └── services/
│       └── ingest/
│           └── usgs_event_ingest.py      # USGS data fetcher with pagination
├── experiments/
│   └── tokyo/
│       ├── run_experiment.sh             # Main experiment runner
│       ├── experiment_driver.py          # Phase 1 driver script
│       ├── triangulation_integration.py  # Spatial triangulation module
│       └── artifacts/                    # Output artifacts
├── tests/
│   ├── triangulation/
│   │   └── test_triangulation_integration.py  # Unit tests
│   └── fixtures/                         # Test fixtures
├── data/
│   ├── raw/
│   │   └── usgs_events/                  # Raw USGS event data
│   └── metadata/                         # Query metadata
├── .github/
│   └── workflows/
│       └── ci.yml                        # CI/CD configuration
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Installation

### Prerequisites
- Python 3.10+
- System dependencies for geospatial libraries

### Install Dependencies

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libspatialindex-dev

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### 1. Fetch USGS Data

```bash
python backend/services/ingest/usgs_event_ingest.py \
  --starttime 2024-10-01T00:00:00Z \
  --endtime 2025-09-30T23:59:59Z \
  --minmag 3.5 \
  --minlatitude 30.0 --maxlatitude 38.5 \
  --minlongitude 136.0 --maxlongitude 145.0 \
  --limit 20000
```

This will:
- Fetch events from USGS FDSN web service
- Save raw data to `data/raw/usgs_events/{window_hash}.json`
- Save metadata to `data/metadata/{window_hash}.yaml`

### 2. Run Experiment

```bash
# Make script executable
chmod +x experiments/tokyo/run_experiment.sh

# Run full experiment pipeline
./experiments/tokyo/run_experiment.sh
```

Or run the driver directly:

```bash
python experiments/tokyo/experiment_driver.py \
  --raw_file data/raw/usgs_events/{window_hash}.json
```

### 3. View Results

Results are saved to `experiments/tokyo/artifacts/`:
- `phase1_results.json`: Complete metrics and provenance
- `calibration_logistic_regression_{horizon}d.png`: Calibration plots for LogReg
- `calibration_lightgbm_{horizon}d.png`: Calibration plots for LightGBM

### 4. Run Triangulation (Optional)

```bash
python experiments/tokyo/triangulation_integration.py
```

Outputs:
- `triangles.json`: Triangle metadata
- `triangles.parquet`: Triangle data (Parquet format)
- `triangle_fault_index.json`: Triangle-to-fault mapping
- `triangle_daily_agg.csv`: Daily event aggregates by triangle

## Testing

### Run Unit Tests

```bash
python tests/triangulation/test_triangulation_integration.py
```

### CI/CD

The project uses GitHub Actions for continuous integration:
- Runs on all PRs and pushes to `main`
- Uses `NO_LIVE_FETCH=1` environment variable for cache-only mode
- Validates code syntax and runs unit tests
- Uploads test artifacts

See `.github/workflows/ci.yml` for configuration.

## Provenance Tracking

All experiments include provenance metadata:
- **Raw file SHA256**: Ensures data integrity
- **Git commit**: Links results to code version
- **Timestamp**: Records execution time
- **Script version**: Tracks driver version

## Dependencies

### Core ML
- `lightgbm`: Gradient boosting framework
- `scikit-learn`: Machine learning utilities
- `pandas`, `numpy`, `scipy`: Data manipulation and scientific computing

### Geospatial
- `shapely`: Geometric operations
- `rtree`: Spatial indexing
- `pyproj`: Coordinate transformations

### Utilities
- `matplotlib`: Visualization
- `pyyaml`: Configuration files
- `requests`: HTTP requests

## Future Enhancements

### Phase 2 (Planned)
- Deep learning models (LSTM, Transformer)
- Attention mechanisms for spatial-temporal patterns
- Real-time prediction API
- Interactive dashboard

### Phase 3 (Planned)
- Multi-region expansion
- Ensemble methods
- Uncertainty quantification
- Operational deployment

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Commit changes (`git commit -m "Add my feature"`)
4. Push to branch (`git push origin feat/my-feature`)
5. Open a Pull Request

## License

This project is part of the GEO_EARTH research initiative.

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Note**: This is an experimental research project. Predictions should not be used for operational earthquake early warning systems without extensive validation and regulatory approval.
