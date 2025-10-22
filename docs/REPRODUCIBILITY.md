# Reproducibility Guide

## Overview

The Earthquake Enhanced system is designed for complete reproducibility. Every calculation can be independently verified using the same input data and parameters.

## Provenance Tracking

### Data Provenance

Every data fetch includes:
1. **Source URL**: Complete API endpoint with parameters
2. **Timestamp**: UTC timestamp of data fetch
3. **Checksum**: SHA-256 hash of raw data
4. **Version**: API version and data format
5. **Record Count**: Number of events fetched

Example metadata:
```json
{
  "source": "USGS Earthquake Catalog",
  "source_url": "https://earthquake.usgs.gov/fdsnws/event/1/query?...",
  "fetch_timestamp": "2025-10-22T12:00:00Z",
  "checksum_sha256": "a1b2c3d4...",
  "record_count": 1234,
  "api_version": "1.0"
}
```

### Calculation Provenance

Every calculation includes:
1. **Physical Constants**: All constants used (μ, etc.)
2. **Formulas**: References to published papers
3. **Parameters**: All input parameters
4. **Timestamps**: When calculation was performed
5. **Software Version**: System version used

## Verification Procedure

### Step 1: Verify Data Integrity

```bash
# Verify checksum of raw data file
cd data/checksums
sha256sum -c usgs_events_*.json.sha256
```

Expected output:
```
usgs_events_2024-01-01_2024-12-31_a1b2c3d4.json: OK
```

### Step 2: Reproduce Calculations

```bash
# Run historical analysis with same parameters
python backend/physics_engine/historical_runner.py \
  --region california \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --min-magnitude 4.0
```

### Step 3: Compare Results

```bash
# Compare output checksums
cd output/california
sha256sum analysis_2024-01-01_2024-12-31.json
```

Results should match exactly if:
- Same input data (verified by checksum)
- Same parameters
- Same software version

## Deterministic Guarantees

### What is Deterministic

✓ **Seismic moment calculation**: M₀ = 10^(1.5×Mw + 9.1)
✓ **Strain rate calculation**: ε̇ = ΣM₀ / (2μVT)
✓ **Triangulation**: Delaunay triangulation is deterministic
✓ **Threshold checks**: Simple comparisons

### What is NOT Deterministic

✗ **Data fetch timing**: USGS catalog updates continuously
✗ **Floating point precision**: May vary slightly across systems
✗ **Random number generation**: NOT USED in this system

## Snapshot Manifests

For long-term reproducibility, create snapshot manifests:

```bash
python scripts/generate_snapshot_manifest.py \
  --data-dir data/raw \
  --output snapshots/manifest_2024-10-22.yaml
```

Manifest includes:
- All data file checksums
- Software version
- Configuration parameters
- Calculation results checksums

## Docker Reproducibility

For maximum reproducibility, use Docker:

```bash
# Build container with specific version
docker build -t earthquake-enhanced:2.0.0 .

# Run analysis in container
docker run -v $(pwd)/data:/app/data \
  earthquake-enhanced:2.0.0 \
  python backend/physics_engine/historical_runner.py
```

Container includes:
- Exact Python version
- Pinned dependencies
- Fixed system libraries

## Publishing Artifacts

For sharing reproducible results:

```bash
python scripts/publish_artifacts.py \
  --analysis-id california_2024 \
  --include-data \
  --include-checksums \
  --output artifacts/
```

Artifact bundle includes:
- Raw data files
- Checksums
- Configuration
- Results
- Provenance metadata
- Software version

## Verification Checklist

Before publishing results:

- [ ] All data checksums verified
- [ ] Calculation parameters documented
- [ ] Physical constants recorded
- [ ] Software version tagged
- [ ] Results checksums generated
- [ ] Provenance metadata complete
- [ ] Independent verification performed

## References

- **Data Integrity**: NIST SP 800-107, Recommendation for Applications Using Approved Hash Algorithms
- **Scientific Reproducibility**: Stodden et al. (2016), Enhancing reproducibility for computational methods
- **Provenance Standards**: W3C PROV Data Model

## Contact

For questions about reproducibility:
- Open an issue on GitHub
- Include: data checksums, parameters, software version
- We will help verify your results

**Last Updated**: 2025-10-22
