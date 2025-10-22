
#!/bin/bash
# GEO_EARTH Tokyo Experiment Runner v1.0
# Usage: ./run_experiment.sh
# Runs ingest, processing, modeling; produces artifacts.

set -e

# Ingest (live fetch)
python backend/services/ingest/usgs_event_ingest.py \
  --starttime 2024-10-01T00:00:00Z \
  --endtime 2025-09-30T23:59:59Z \
  --minmag 3.5 \
  --minlatitude 30.0 --maxlatitude 38.5 \
  --minlongitude 136.0 --maxlongitude 145.0 \
  --limit 20000

# Processing + modeling (use latest raw_file from metadata)
latest_meta=$(ls -t data/metadata/*.yaml | head -1)
window_hash=$(yq e '.window_hash' $latest_meta)
python experiments/tokyo/experiment_driver.py --raw_file data/raw/usgs_events/${window_hash}.json

# Archive artifacts
git add artifacts/*
git commit -m "Tokyo Phase 1 artifacts $(date -u +%Y-%m-%dT%H:%M:%SZ)"
