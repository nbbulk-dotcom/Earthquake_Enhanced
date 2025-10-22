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

### Resonance Analysis (`backend/features/`)
- **Modal Spectral Curves**: Eigendecomposition of cross-spectral density matrices
- **Resonant Frequency Detection**: Identifies coherent oscillations across station networks
- **Geometry-Based Estimates**: f_geom ≈ Vs / (2L) for characteristic length L
- **Quality Factor (Q)**: Measures sharpness of resonance peaks
- **Multi-Band Analysis**: Analyzes very low (0.001-0.1 Hz), low (0.1-1 Hz), and medium (1-10 Hz) frequencies
- **Integration with Triangulation**: Combines spatial strain rate with frequency-domain resonance

#### Physics References:
- Bendat & Piersol (2010): Random Data Analysis and Measurement Procedures
- Lacoss et al. (1969): Estimation of Seismic Noise Structure using Arrays
- Aki & Richards (2002): Quantitative Seismology (Chapter 5: Resonance)

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

# Run all tests including resonance module
pytest

# Run only resonance tests
pytest backend/features/tests/test_resonance.py -v

# Run resonance module standalone example
python backend/features/resonance.py

# Run triangle feature extraction example
python backend/features/tri_feature.py
```

#### Resonance Test Coverage
The resonance analysis module includes comprehensive tests:
- **Coherent Sine Wave Tests**: Verify peak detection with known frequencies
- **Geometry Calculation Tests**: Validate f_geom = Vs / (2L) formula
- **Q-Factor Tests**: Test half-power bandwidth calculation
- **Transfer Function Tests**: Verify frequency response computation
- **Integration Tests**: Full workflow from waveform to resonance features
- **Determinism Tests**: Ensure same input → same output

All tests use synthetic data with known properties to validate numerical accuracy.

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
