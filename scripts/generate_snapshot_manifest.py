#!/usr/bin/env python3
"""
Generate snapshot manifest for reproducibility.

Creates a YAML manifest with checksums of all data files,
configuration, and results for long-term reproducibility.
"""

import hashlib
import yaml
from pathlib import Path
from datetime import datetime, timezone
import argparse


def calculate_checksum(filepath: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_manifest(data_dir: str, output_file: str):
    """Generate snapshot manifest."""
    data_path = Path(data_dir)
    
    manifest = {
        "snapshot_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system_version": "2.0.0",
        "files": {},
    }
    
    # Process all JSON files in data directory
    for json_file in data_path.rglob("*.json"):
        rel_path = str(json_file.relative_to(data_path))
        checksum = calculate_checksum(json_file)
        
        manifest["files"][rel_path] = {
            "checksum_sha256": checksum,
            "size_bytes": json_file.stat().st_size,
            "modified": datetime.fromtimestamp(
                json_file.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }
    
    # Save manifest
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    
    print(f"Manifest generated: {output_file}")
    print(f"Files cataloged: {len(manifest['files'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate snapshot manifest")
    parser.add_argument("--data-dir", default="data/raw", help="Data directory")
    parser.add_argument("--output", default="snapshots/manifest.yaml", help="Output file")
    
    args = parser.parse_args()
    generate_manifest(args.data_dir, args.output)
