#!/bin/bash
# Tokyo Region Experiment Runner

set -e

echo "=========================================="
echo "Tokyo Region Seismic Stress Measurement"
echo "=========================================="
echo ""

# Configuration
REGION="tokyo"
START_DATE="2023-01-01"
END_DATE="2024-12-31"
MIN_MAGNITUDE=4.0

echo "Configuration:"
echo "  Region: $REGION"
echo "  Period: $START_DATE to $END_DATE"
echo "  Min Magnitude: $MIN_MAGNITUDE"
echo ""

# Create output directory
mkdir -p output

# Run analysis
echo "Running physics engine..."
python ../../backend/physics_engine/historical_runner.py \
  --region "$REGION" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --min-magnitude "$MIN_MAGNITUDE"

echo ""
echo "Experiment complete!"
echo "Results saved to: output/"
