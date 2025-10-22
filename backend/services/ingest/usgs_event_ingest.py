"""
USGS Event Ingest Service with Complete Provenance Tracking

Fetches earthquake data from USGS API and maintains full provenance:
- SHA-256 checksums for all data
- Timestamp tracking
- Version control
- Parameter logging
"""

import hashlib
import json
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path


class USGSEventIngest:
    """Ingest earthquake events from USGS with provenance tracking."""
    
    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.checksum_dir = self.data_dir / "checksums"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.checksum_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_events(
        self,
        starttime: str,
        endtime: str,
        minmagnitude: float = 2.5,
        minlatitude: Optional[float] = None,
        maxlatitude: Optional[float] = None,
        minlongitude: Optional[float] = None,
        maxlongitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fetch earthquake events from USGS API.
        
        Args:
            starttime: ISO format start time (e.g., "2024-01-01")
            endtime: ISO format end time
            minmagnitude: Minimum magnitude threshold
            minlatitude: Minimum latitude (optional)
            maxlatitude: Maximum latitude (optional)
            minlongitude: Minimum longitude (optional)
            maxlongitude: Maximum longitude (optional)
        
        Returns:
            Dictionary with events and provenance metadata
        """
        # Build parameters
        params = {
            "format": "geojson",
            "starttime": starttime,
            "endtime": endtime,
            "minmagnitude": minmagnitude,
            "orderby": "time-asc",
        }
        
        if minlatitude is not None:
            params["minlatitude"] = minlatitude
        if maxlatitude is not None:
            params["maxlatitude"] = maxlatitude
        if minlongitude is not None:
            params["minlongitude"] = minlongitude
        if maxlongitude is not None:
            params["maxlongitude"] = maxlongitude
        
        # Fetch data
        fetch_timestamp = datetime.now(timezone.utc).isoformat()
        response = requests.get(self.BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        # Calculate checksum
        data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        checksum = hashlib.sha256(data_bytes).hexdigest()
        
        # Create provenance metadata
        metadata = {
            "source": "USGS Earthquake Catalog",
            "source_url": response.url,
            "fetch_timestamp": fetch_timestamp,
            "parameters": params,
            "checksum_sha256": checksum,
            "record_count": len(data.get("features", [])),
            "api_version": "1.0",
        }
        
        # Save raw data
        filename = f"usgs_events_{starttime}_{endtime}_{checksum[:8]}.json"
        raw_path = self.raw_dir / filename
        with open(raw_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save checksum
        checksum_path = self.checksum_dir / f"{filename}.sha256"
        with open(checksum_path, 'w') as f:
            f.write(f"{checksum}  {filename}\n")
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{filename}.meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Fetched {metadata['record_count']} events")
        print(f"Checksum: {checksum}")
        print(f"Saved to: {raw_path}")
        
        return {
            "data": data,
            "metadata": metadata,
            "files": {
                "raw": str(raw_path),
                "checksum": str(checksum_path),
                "metadata": str(metadata_path),
            }
        }
    
    def verify_checksum(self, filepath: str) -> bool:
        """Verify the SHA-256 checksum of a data file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        
        calculated = hashlib.sha256(data).hexdigest()
        
        checksum_file = Path(filepath).name + ".sha256"
        checksum_path = self.checksum_dir / checksum_file
        
        if not checksum_path.exists():
            print(f"Warning: Checksum file not found: {checksum_path}")
            return False
        
        with open(checksum_path, 'r') as f:
            stored = f.read().split()[0]
        
        if calculated == stored:
            print(f"✓ Checksum verified: {filepath}")
            return True
        else:
            print(f"✗ Checksum mismatch: {filepath}")
            print(f"  Expected: {stored}")
            print(f"  Got: {calculated}")
            return False


if __name__ == "__main__":
    # Example usage
    ingest = USGSEventIngest()
    
    # Fetch recent events for California
    result = ingest.fetch_events(
        starttime="2024-01-01",
        endtime="2024-12-31",
        minmagnitude=4.0,
        minlatitude=32.0,
        maxlatitude=42.0,
        minlongitude=-125.0,
        maxlongitude=-114.0,
    )
    
    print(f"\nFetched {result['metadata']['record_count']} events")
    print(f"Files saved:")
    for key, path in result['files'].items():
        print(f"  {key}: {path}")
