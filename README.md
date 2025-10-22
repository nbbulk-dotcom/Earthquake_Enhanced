# Earthquake Enhanced - Deterministic Measurement System

## Overview

This is a **deterministic, physics-based seismic stress measurement system** that calculates empirical quantities from real earthquake data. It does NOT predict earthquakes using probabilistic models.

## Key Principles

### 1. No Probabilistic Predictions
- No Logistic Regression, LightGBM, or ML classifiers
- No synthetic data generation
- No statistical forecasting

### 2. Deterministic Measurement Only
- Calculates physical quantities: seismic moment, strain rate, stress accumulation
- Uses real USGS earthquake catalog data
- Applies established physics formulas (Hanks & Kanamori, Kostrov)

### 3. Threshold-Based Alerts
- Reports when measured quantities exceed pre-defined physical thresholds
- Thresholds based on published seismological research
- Clear alert policies documented in `docs/ALERT_POLICY.md`

### 4. Complete Provenance
- Every data point tracked with checksums (SHA-256)
- Version control for all data sources
- Reproducible calculations with full audit trail
- Metadata includes: timestamp, source, version, parameters

## System Architecture

```
earthquake-enhanced/
├── backend/
│   ├── services/ingest/          # Data ingestion with provenance
│   ├── calculators/               # Physics calculations
│   ├── triangulation/             # Delaunay triangulation & fault indexing
│   ├── physics_engine/            # Core measurement engine
│   └── features/                  # Feature extraction utilities
├── tests/                         # Unit and integration tests
├── docs/                          # Documentation
├── experiments/                   # Regional experiments
└── scripts/                       # Utility scripts
```

## Core Components

### Data Ingestion (`backend/services/ingest/`)
- Fetches earthquake data from USGS API
- Validates data integrity
- Generates checksums for provenance
- Stores metadata with timestamps

### Physics Calculators (`backend/calculators/`)
- **Seismic Moment**: M₀ = μ × A × D (Hanks & Kanamori, 1979)
- **Strain Rate**: ε̇ = ΣM₀ / (2μVT) (Kostrov, 1974)
- Uses rigidity μ = 3.0×10¹⁰ Pa for crustal rocks

### Triangulation (`backend/triangulation/`)
- Delaunay triangulation of earthquake epicenters
- Calculates triangle areas and volumes
- R-tree spatial indexing for fault proximity

### Physics Engine (`backend/physics_engine/`)
- Orchestrates all calculations
- Applies threshold checks
- Generates alerts when thresholds exceeded
- Maintains calculation provenance

## Installation

```bash
# Clone repository
git clone https://github.com/nbbulk-dotcom/Earthquake_Enhanced.git
cd Earthquake_Enhanced

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Usage

### Run Historical Analysis
```bash
# Using Makefile
make run-historical

# Or directly
python backend/physics_engine/historical_runner.py
```

### Run Regional Experiment
```bash
cd experiments/tokyo
./run_experiment.sh
```

### Run Tests
```bash
make test
```

## Data Sources

All data sources are documented in `data_manifest.yaml`:
- USGS Earthquake Catalog (primary source)
- Fault line databases (for proximity calculations)
- Regional seismic networks (supplementary)

## Alert Thresholds

Documented in `docs/ALERT_POLICY.md`:
- Strain rate > 1.0×10⁻⁷ /year (elevated stress)
- Seismic moment accumulation > threshold for region
- Fault proximity < 10 km with elevated strain

## Reproducibility

See `docs/REPRODUCIBILITY.md` for:
- Data versioning procedures
- Checksum verification
- Calculation reproducibility
- Artifact publishing

## Contributing

See `docs/PR_CHECKLIST.md` for pull request requirements:
- All calculations must be deterministic
- Full provenance tracking required
- Unit tests for all physics calculations
- Documentation updates

## License

MIT License - See LICENSE file

## References

- Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. *Journal of Geophysical Research*, 84(B5), 2348-2350.
- Kostrov, V. V. (1974). Seismic moment and energy of earthquakes, and seismic flow of rock. *Izvestiya, Physics of the Solid Earth*, 1, 23-44.
- Aki, K., & Richards, P. G. (2002). *Quantitative Seismology* (2nd ed.). University Science Books.

## Contact

For questions about the deterministic measurement approach, please open an issue on GitHub.
