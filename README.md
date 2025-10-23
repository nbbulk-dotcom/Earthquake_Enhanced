# Earthquake Enhanced System (GEO_EARTH / BRETT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 🌍 Overview

The **Earthquake Enhanced System** (also known as **GEO_EARTH** or **BRETT**) is a comprehensive, physics-informed earthquake prediction and analysis platform. This system implements advanced seismological analysis techniques, empirical feature engineering, and machine learning models to predict earthquake activity with multi-horizon forecasting capabilities.

### Key Features

- **Multi-Horizon Prediction**: 7-day, 14-day, and 30-day earthquake forecasting
- **Empirical Feature Engineering**: Physics-based features including seismic moment, distance metrics, and resonance analysis
- **Class-Weighted Models**: Handles severe class imbalance in earthquake prediction
- **Comprehensive Testing**: Full test coverage for all modules
- **Real USGS Data**: Uses actual earthquake data from USGS API
- **Reproducible Pipeline**: Complete data lineage from raw data to predictions

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Running Experiments](#running-experiments)
- [Running Tests](#running-tests)
- [Key Components](#key-components)
- [Configuration](#configuration)
- [Results and Artifacts](#results-and-artifacts)
- [For External AI Evaluation](#for-external-ai-evaluation)
- [Contributing](#contributing)
- [License](#license)

## 📁 Project Structure

```
Earthquake_Enhanced/
├── backend/
│   ├── experiments/
│   │   └── tokyo/
│   │       ├── __init__.py
│   │       ├── run_experiment.py          # Main Tokyo experiment
│   │       └── tests/
│   │           ├── __init__.py
│   │           └── test_tokyo_experiment.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── config.yaml                    # System configuration
│   │   ├── resonance.py                   # Resonance analysis
│   │   ├── tri_feature.py                 # Triangle spatial features
│   │   ├── correlation_engine.py          # Correlation analysis
│   │   ├── space_engine.py                # Spatial analysis
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_resonance.py
│   │       ├── test_correlation_engine.py
│   │       └── test_space_engine.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py
│   ├── api/
│   │   └── ...
│   └── utils/
│       └── __init__.py
├── data/
│   ├── raw/
│   │   └── usgs_events/
│   │       └── 2022_tokyo_07e0d7f2a.json  # 2022 Tokyo earthquake data
│   ├── metadata/
│   │   └── 2022_tokyo_07e0d7f2a.yaml      # Data metadata
│   └── processed/
├── artifacts/                              # Generated experiment outputs
├── config/
├── docs/
├── frontend/
├── scripts/
├── requirements.txt                        # Python dependencies
├── test_system.py
├── .gitignore
└── README.md                              # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/nbbulk-dotcom/Earthquake_Enhanced.git
cd Earthquake_Enhanced
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, lightgbm; print('All dependencies installed successfully!')"
```

## ⚡ Quick Start

### Run the Tokyo Experiment

```bash
# From the repository root
python backend/experiments/tokyo/run_experiment.py
```

This will:
1. Load the 2022 Tokyo earthquake data
2. Engineer empirical features
3. Create multi-horizon labels (7d, 14d, 30d)
4. Train class-weighted models (Logistic Regression + LightGBM)
5. Evaluate performance metrics
6. Generate visualizations
7. Save results to `artifacts/`

Expected output location: `artifacts/tokyo_experiment_results.json`

## 📊 Data Format

### USGS GeoJSON Format

The system uses standard USGS GeoJSON format for earthquake data:

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "generated": 1761222499000,
    "count": 586,
    "url": "https://earthquake.usgs.gov/fdsnws/event/1/query?..."
  },
  "features": [
    {
      "type": "Feature",
      "id": "us7000j12e",
      "properties": {
        "mag": 4.4,
        "place": "5 km ENE of Sakai, Japan",
        "time": 1672413006152,
        "status": "reviewed",
        "tsunami": 0
      },
      "geometry": {
        "type": "Point",
        "coordinates": [139.8576, 36.1139, 56.826]
      }
    }
  ]
}
```

### Data Location

- **Raw Data**: `data/raw/usgs_events/`
- **Metadata**: `data/metadata/`
- **Test Data**: 2022 Tokyo region data included in repository

### Data Coverage

- **Region**: Tokyo area (30.0°N - 38.5°N, 136.0°E - 145.0°E)
- **Time Period**: 2022-01-01 to 2022-12-31
- **Magnitude Range**: ≥ 3.5
- **Event Count**: 586 earthquakes

## 🧪 Running Experiments

### Tokyo Multi-Horizon Experiment

```bash
cd Earthquake_Enhanced
python backend/experiments/tokyo/run_experiment.py
```

**What it does:**
- Loads 2022 Tokyo earthquake data
- Computes rolling temporal features (7d, 30d, 90d windows)
- Calculates cumulative seismic moment
- Measures distance to recent large events (M≥5.0)
- Creates multi-horizon labels (7d, 14d, 30d)
- Trains class-weighted models
- Evaluates: Precision, Recall, F1, ROC-AUC, Brier Score
- Saves results and visualizations

**Expected Runtime:** 30-60 seconds

### Custom Experiments

To create your own experiment:

1. Create a new directory: `backend/experiments/your_experiment/`
2. Add `__init__.py` and `run_experiment.py`
3. Load data from `data/raw/usgs_events/`
4. Use configuration from `backend/features/config.yaml`
5. Save results to `artifacts/`

## 🧪 Running Tests

### Run All Tests

```bash
# From repository root
pytest -v
```

### Run Specific Test Suites

```bash
# Tokyo experiment tests
pytest backend/experiments/tokyo/tests/ -v

# Resonance engine tests
pytest backend/features/tests/test_resonance.py -v

# All feature tests
pytest backend/features/tests/ -v
```

### Run with Coverage

```bash
pytest --cov=backend --cov-report=html
```

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test full experiment pipeline
- **Edge Case Tests**: Test error handling and boundary conditions

## 🔧 Key Components

### 1. Tokyo Experiment (`backend/experiments/tokyo/run_experiment.py`)

Main experiment implementing multi-horizon earthquake prediction with:
- Empirical feature engineering
- Class-weighted models (Logistic Regression, LightGBM)
- Multi-horizon labeling (7d, 14d, 30d)
- Comprehensive evaluation metrics

**Key Functions:**
- `haversine_distance()`: Calculate great-circle distance
- `calculate_seismic_moment()`: Hanks & Kanamori (1979) formula
- `run_tokyo_experiment()`: Main pipeline

### 2. Resonance Engine (`backend/features/resonance.py`)

Physics-based resonance analysis:
- Strain-rate tensor calculations
- Crustal stress resonance
- Schumann resonance harmonics
- Seismic wave propagation modeling

### 3. Triangle Features (`backend/features/tri_feature.py`)

Geometric spatial pattern analysis:
- Triangle detection from earthquake triplets
- Area, perimeter, aspect ratio, compactness metrics
- Spatial clustering indicators

**Key Functions:**
- `find_earthquake_triangles()`: Detect triangle patterns
- `calculate_triangle_metrics()`: Compute geometric features
- `compute_triangle_features()`: Aggregate for daily prediction

### 4. Configuration (`backend/features/config.yaml`)

Centralized system configuration:
- Feature engineering parameters
- Model hyperparameters
- Experiment settings
- Data quality thresholds

## ⚙️ Configuration

Edit `backend/features/config.yaml` to customize:

```yaml
# Example: Modify prediction horizons
experiment:
  prediction_horizons: [7, 14, 30]  # Days
  magnitude_threshold: 5.0          # M≥5.0 for labeling

# Example: Adjust feature windows
features:
  temporal_windows: [7, 14, 30, 90]  # Days
  magnitude_thresholds: [3.5, 4.0, 4.5, 5.0, 5.5]
```

## 📈 Results and Artifacts

After running experiments, find outputs in `artifacts/`:

```
artifacts/
├── tokyo_experiment_results.json       # Metrics for all horizons
├── tokyo_daily_features.csv            # Processed daily features
└── tokyo_experiment_predictions.png    # Visualization
```

### Sample Results Structure

```json
{
  "7d": {
    "logistic_regression": {
      "precision": 0.xxx,
      "recall": 0.xxx,
      "f1": 0.xxx,
      "roc_auc": 0.xxx,
      "brier": 0.xxx
    },
    "lightgbm": { ... }
  },
  "14d": { ... },
  "30d": { ... }
}
```

## 🤖 For External AI Evaluation

### Accessing the Package

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nbbulk-dotcom/Earthquake_Enhanced.git
   cd Earthquake_Enhanced
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data integrity:**
   ```bash
   ls -lh data/raw/usgs_events/2022_tokyo_07e0d7f2a.json
   ls -lh data/metadata/2022_tokyo_07e0d7f2a.yaml
   ```

4. **Run the experiment:**
   ```bash
   python backend/experiments/tokyo/run_experiment.py
   ```

5. **Run tests:**
   ```bash
   pytest -v
   ```

### Evaluation Checklist

- ✅ **Data Integrity**: USGS GeoJSON with 586 Tokyo earthquakes (2022)
- ✅ **Reproducibility**: Fixed random seeds, deterministic pipeline
- ✅ **Feature Engineering**: Empirical features (seismic moment, distance, temporal)
- ✅ **Multi-Horizon**: 7d, 14d, 30d prediction windows
- ✅ **Class Weighting**: Handles severe class imbalance
- ✅ **Comprehensive Testing**: Unit tests, integration tests, edge cases
- ✅ **Documentation**: Complete README, inline code comments
- ✅ **Configuration**: Centralized config.yaml
- ✅ **Results**: JSON metrics, CSV features, PNG visualizations

### Key Files to Review

1. `backend/experiments/tokyo/run_experiment.py` - Main experiment logic
2. `backend/features/config.yaml` - System configuration
3. `data/raw/usgs_events/2022_tokyo_07e0d7f2a.json` - Test data
4. `backend/experiments/tokyo/tests/test_tokyo_experiment.py` - Test suite
5. `backend/features/tri_feature.py` - Spatial feature engineering
6. `backend/features/resonance.py` - Physics-based analysis

### Expected Behavior

The Tokyo experiment should:
- Load 586 events from 2022 Tokyo data
- Create features with 7d, 30d, 90d rolling windows
- Generate multi-horizon labels with class imbalance (~1-5% positive)
- Train models with class weighting
- Produce evaluation metrics (precision, recall, F1, ROC-AUC, Brier)
- Save results to `artifacts/`

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Maintain backward compatibility
- Use descriptive commit messages

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🔍 Technical Details

### Feature Engineering

**Temporal Features:**
- Rolling counts (7d, 30d, 90d)
- Rolling mean/max magnitude
- Rolling mean depth

**Empirical Features:**
- Cumulative seismic moment (log10 scale)
- Distance to nearest M≥5.0 event (90d window)
- Triangle spatial patterns (optional)

**Physics-Based Features:**
- Resonance analysis (Schumann frequencies)
- Strain-rate calculations
- Crustal stress indicators

### Model Architecture

**Logistic Regression:**
- Class weight: 'balanced'
- Max iterations: 1000
- Solver: lbfgs (default)

**LightGBM:**
- Scale positive weight: automatic
- Learning rate: 0.05
- Max depth: 6
- N estimators: 100

### Evaluation Metrics

- **Precision**: Positive predictive value
- **Recall**: True positive rate (sensitivity)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Brier Score**: Mean squared error of probabilistic predictions

---

**Built with ❤️ for advancing earthquake science and public safety**
