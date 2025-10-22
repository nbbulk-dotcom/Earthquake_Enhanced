#!/usr/bin/env python3
"""
Publish reproducible artifacts.

Creates a bundle with all necessary files for reproducing
an analysis: data, checksums, configuration, results.
"""

import shutil
import json
from pathlib import Path
from datetime import datetime, timezone
import argparse
import tarfile


def publish_artifacts(
    analysis_id: str,
    include_data: bool = True,
    include_checksums: bool = True,
    output_dir: str = "artifacts"
):
    """
    Publish artifact bundle for an analysis.
    
    Args:
        analysis_id: Unique identifier for the analysis
        include_data: Include raw data files
        include_checksums: Include checksum files
        output_dir: Output directory for artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary bundle directory
    bundle_dir = output_path / f"bundle_{analysis_id}"
    bundle_dir.mkdir(exist_ok=True)
    
    print(f"Creating artifact bundle: {analysis_id}")
    
    # Copy configuration
    config_src = Path("backend/physics_engine/config.json")
    if config_src.exists():
        shutil.copy(config_src, bundle_dir / "config.json")
        print("  ✓ Configuration")
    
    # Copy data manifest
    manifest_src = Path("data_manifest.yaml")
    if manifest_src.exists():
        shutil.copy(manifest_src, bundle_dir / "data_manifest.yaml")
        print("  ✓ Data manifest")
    
    # Copy data files if requested
    if include_data:
        data_dir = bundle_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        raw_data = Path("data/raw")
        if raw_data.exists():
            for data_file in raw_data.glob("*.json"):
                shutil.copy(data_file, data_dir / data_file.name)
            print(f"  ✓ Data files ({len(list(data_dir.glob('*.json')))} files)")
    
    # Copy checksums if requested
    if include_checksums:
        checksum_dir = bundle_dir / "checksums"
        checksum_dir.mkdir(exist_ok=True)
        
        checksums_src = Path("data/checksums")
        if checksums_src.exists():
            for checksum_file in checksums_src.glob("*.sha256"):
                shutil.copy(checksum_file, checksum_dir / checksum_file.name)
            print(f"  ✓ Checksums ({len(list(checksum_dir.glob('*.sha256')))} files)")
    
    # Create metadata
    metadata = {
        "analysis_id": analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system_version": "2.0.0",
        "includes_data": include_data,
        "includes_checksums": include_checksums,
    }
    
    with open(bundle_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ Metadata")
    
    # Create tarball
    tarball_path = output_path / f"{analysis_id}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=analysis_id)
    
    # Clean up temporary directory
    shutil.rmtree(bundle_dir)
    
    print(f"\n✓ Artifact bundle created: {tarball_path}")
    print(f"  Size: {tarball_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish reproducible artifacts")
    parser.add_argument("--analysis-id", required=True, help="Analysis identifier")
    parser.add_argument("--include-data", action="store_true", help="Include raw data")
    parser.add_argument("--include-checksums", action="store_true", help="Include checksums")
    parser.add_argument("--output", default="artifacts", help="Output directory")
    
    args = parser.parse_args()
    
    publish_artifacts(
        args.analysis_id,
        args.include_data,
        args.include_checksums,
        args.output
    )
