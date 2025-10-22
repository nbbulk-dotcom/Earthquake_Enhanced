"""
Historical Data Runner

Runs the physics engine on historical earthquake data:
1. Fetches data from USGS
2. Processes with physics engine
3. Generates reports
4. Saves results with provenance
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

from backend.services.ingest.usgs_event_ingest import USGSEventIngest
from backend.physics_engine.engine_driver import PhysicsEngine


def load_config(config_path: str = "backend/physics_engine/config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_historical_analysis(
    region: str,
    start_date: str,
    end_date: str,
    min_magnitude: float = 4.0,
    config_path: str = "backend/physics_engine/config.json"
):
    """
    Run historical analysis for a specific region and time period.
    
    Args:
        region: Region name (e.g., "california", "japan")
        start_date: Start date (ISO format: YYYY-MM-DD)
        end_date: End date (ISO format: YYYY-MM-DD)
        min_magnitude: Minimum magnitude threshold
        config_path: Path to configuration file
    """
    print("=" * 80)
    print("EARTHQUAKE ENHANCED - HISTORICAL ANALYSIS")
    print("Deterministic Physics-Based Measurement System")
    print("=" * 80)
    print()
    
    # Load configuration
    config = load_config(config_path)
    region_config = config["regions"].get(region)
    
    if not region_config:
        print(f"Error: Region '{region}' not found in configuration")
        return
    
    print(f"Region: {region}")
    print(f"Time Period: {start_date} to {end_date}")
    print(f"Minimum Magnitude: {min_magnitude}")
    print()
    
    # Initialize components
    ingest = USGSEventIngest()
    engine = PhysicsEngine(config)
    
    # Fetch data
    print("Fetching earthquake data from USGS...")
    result = ingest.fetch_events(
        starttime=start_date,
        endtime=end_date,
        minmagnitude=min_magnitude,
        minlatitude=region_config["bounds"]["min_lat"],
        maxlatitude=region_config["bounds"]["max_lat"],
        minlongitude=region_config["bounds"]["min_lon"],
        maxlongitude=region_config["bounds"]["max_lon"],
    )
    
    print(f"✓ Fetched {result['metadata']['record_count']} events")
    print(f"✓ Checksum: {result['metadata']['checksum_sha256'][:16]}...")
    print()
    
    # Extract events
    features = result["data"]["features"]
    events = []
    
    for feature in features:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        
        events.append({
            "magnitude": props["mag"],
            "longitude": coords[0],
            "latitude": coords[1],
            "depth_km": coords[2],
            "time": props["time"],
            "place": props.get("place", ""),
        })
    
    # Process with physics engine
    print("Running physics calculations...")
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    
    physics_results = engine.process_earthquake_data(events, start_dt, end_dt)
    
    # Display results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    print(f"Event Statistics:")
    print(f"  Total Events: {physics_results['event_statistics']['num_events']}")
    print(f"  Magnitude Range: {physics_results['event_statistics']['magnitude_range']}")
    print(f"  Depth Range: {physics_results['event_statistics']['depth_range_km']} km")
    print()
    
    print(f"Seismic Moment:")
    print(f"  Total: {physics_results['seismic_moment']['total_Nm']:.2e} N⋅m")
    print(f"  Equivalent Magnitude: {physics_results['seismic_moment']['equivalent_magnitude']:.2f}")
    print()
    
    print(f"Spatial Analysis:")
    print(f"  Triangles: {physics_results['spatial_analysis']['num_triangles']}")
    print(f"  Total Area: {physics_results['spatial_analysis']['total_area_km2']:.2f} km²")
    print(f"  Total Volume: {physics_results['spatial_analysis']['total_volume_km3']:.2f} km³")
    print()
    
    print(f"Strain Rate:")
    print(f"  Calculated: {physics_results['strain_rate']['value_per_year']:.2e} /year")
    print(f"  Threshold: {physics_results['strain_rate']['threshold_per_year']:.2e} /year")
    print(f"  Status: {'⚠ EXCEEDS THRESHOLD' if physics_results['strain_rate']['exceeds_threshold'] else '✓ Within normal range'}")
    print()
    
    # Display alerts
    if physics_results["alerts"]:
        print("ALERTS:")
        for alert in physics_results["alerts"]:
            print(f"  [{alert['severity']}] {alert['type']}")
            print(f"    {alert['message']}")
        print()
    else:
        print("No alerts generated.")
        print()
    
    # Save results
    output_dir = Path("output") / region
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"analysis_{start_date}_{end_date}.json"
    engine.save_results(physics_results, str(output_file))
    
    print(f"✓ Results saved to: {output_file}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    # Default: California, last year
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=365)
    
    run_historical_analysis(
        region="california",
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        min_magnitude=4.0
    )
