
#!/usr/bin/env python3
"""
USGS FDSN Event Web Service Adapter
Fetches earthquake events with pagination and metadata tracking.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import requests
import yaml


def compute_window_hash(params):
    """Compute SHA256 hash of query parameters for unique identification."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()[:16]


def fetch_usgs_events(params, limit=20000):
    """
    Fetch events from USGS FDSN web service with pagination.
    
    Args:
        params: Dictionary of query parameters
        limit: Maximum number of events to fetch
    
    Returns:
        GeoJSON FeatureCollection with all events
    """
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    all_features = []
    offset = 1
    page_size = 20000  # USGS max per request
    
    print(f"Fetching events from USGS (limit={limit})...")
    
    while len(all_features) < limit:
        # Update pagination params
        current_params = params.copy()
        current_params["format"] = "geojson"
        current_params["offset"] = offset
        current_params["limit"] = min(page_size, limit - len(all_features))
        
        # Make request
        url = f"{base_url}?{urlencode(current_params)}"
        print(f"  Fetching page (offset={offset}, limit={current_params['limit']})...")
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            sys.exit(1)
        
        features = data.get("features", [])
        if not features:
            print(f"  No more events found. Total fetched: {len(all_features)}")
            break
        
        all_features.extend(features)
        print(f"  Fetched {len(features)} events. Total: {len(all_features)}")
        
        # Check if we've reached the end
        if len(features) < current_params["limit"]:
            print(f"  Reached end of results. Total fetched: {len(all_features)}")
            break
        
        offset += len(features)
    
    # Construct GeoJSON FeatureCollection
    result = {
        "type": "FeatureCollection",
        "metadata": {
            "generated": datetime.utcnow().isoformat() + "Z",
            "count": len(all_features),
            "status": 200
        },
        "features": all_features
    }
    
    return result


def save_events(data, params, output_dir="data/raw/usgs_events", metadata_dir="data/metadata"):
    """Save events and metadata."""
    output_path = Path(output_dir)
    metadata_path = Path(metadata_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path.mkdir(parents=True, exist_ok=True)
    
    # Compute window hash
    window_hash = compute_window_hash(params)
    
    # Save events
    events_file = output_path / f"{window_hash}.json"
    with open(events_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nEvents saved to: {events_file}")
    
    # Save metadata
    metadata = {
        "window_hash": window_hash,
        "query_params": params,
        "event_count": data["metadata"]["count"],
        "generated_at": data["metadata"]["generated"],
        "file_path": str(events_file)
    }
    
    metadata_file = metadata_path / f"{window_hash}.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"Metadata saved to: {metadata_file}")
    
    return window_hash


def main():
    parser = argparse.ArgumentParser(description="USGS Event Ingest Service")
    parser.add_argument("--starttime", required=True, help="Start time (ISO8601)")
    parser.add_argument("--endtime", required=True, help="End time (ISO8601)")
    parser.add_argument("--minmag", type=float, required=True, help="Minimum magnitude")
    parser.add_argument("--minlatitude", type=float, required=True, help="Minimum latitude")
    parser.add_argument("--maxlatitude", type=float, required=True, help="Maximum latitude")
    parser.add_argument("--minlongitude", type=float, required=True, help="Minimum longitude")
    parser.add_argument("--maxlongitude", type=float, required=True, help="Maximum longitude")
    parser.add_argument("--limit", type=int, default=20000, help="Maximum events to fetch")
    
    args = parser.parse_args()
    
    # Build query parameters
    params = {
        "starttime": args.starttime,
        "endtime": args.endtime,
        "minmagnitude": args.minmag,
        "minlatitude": args.minlatitude,
        "maxlatitude": args.maxlatitude,
        "minlongitude": args.minlongitude,
        "maxlongitude": args.maxlongitude,
        "orderby": "time"
    }
    
    print("=" * 80)
    print("USGS Event Ingest")
    print("=" * 80)
    print(f"\nQuery parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"  limit: {args.limit}")
    
    # Fetch events
    data = fetch_usgs_events(params, limit=args.limit)
    
    # Save events and metadata
    window_hash = save_events(data, params)
    
    print("\n" + "=" * 80)
    print("INGEST COMPLETE")
    print("=" * 80)
    print(f"\nWindow hash: {window_hash}")
    print(f"Total events: {data['metadata']['count']}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
