"""
Physics Engine Driver - Core Deterministic Measurement System

Orchestrates all physics calculations:
1. Seismic moment calculation
2. Strain rate calculation (Kostrov formula)
3. Threshold checking
4. Alert generation
5. Provenance tracking
"""

import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from backend.calculators.seismic_moment import SeismicMomentCalculator
from backend.triangulation.triangulate import EarthquakeTriangulation


class PhysicsEngine:
    """Core deterministic physics measurement engine."""
    
    # Physical constants
    RIGIDITY_MU = 3.0e10  # Pa (shear modulus for crustal rocks)
    SECONDS_PER_YEAR = 365.25 * 24 * 3600
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize physics engine with configuration.
        
        Args:
            config: Configuration dictionary with thresholds and parameters
        """
        self.config = config
        self.moment_calc = SeismicMomentCalculator()
        self.triangulation = EarthquakeTriangulation()
        
        # Load thresholds
        self.thresholds = config.get("thresholds", {})
        self.strain_rate_threshold = self.thresholds.get("strain_rate_per_year", 1.0e-7)
        self.fault_proximity_km = self.thresholds.get("fault_proximity_km", 10.0)
    
    def calculate_strain_rate(
        self,
        seismic_moments: np.ndarray,
        volume_m3: float,
        time_span_years: float
    ) -> float:
        """
        Calculate strain rate using Kostrov formula.
        
        Formula: ε̇ = ΣM₀ / (2μVT)
        
        Reference: Kostrov, V. V. (1974). Seismic moment and energy of 
        earthquakes, and seismic flow of rock. Izvestiya, Physics of the 
        Solid Earth, 1, 23-44.
        
        Args:
            seismic_moments: Array of seismic moments (N⋅m)
            volume_m3: Volume of the region (m³)
            time_span_years: Time span in years
        
        Returns:
            Strain rate (per year)
        """
        total_moment = np.sum(seismic_moments)
        time_span_seconds = time_span_years * self.SECONDS_PER_YEAR
        
        # Kostrov formula
        strain_rate_per_second = total_moment / (2 * self.RIGIDITY_MU * volume_m3 * time_span_seconds)
        
        # Convert to per year
        strain_rate_per_year = strain_rate_per_second * self.SECONDS_PER_YEAR
        
        return strain_rate_per_year
    
    def process_earthquake_data(
        self,
        events: List[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Process earthquake data and calculate all physics quantities.
        
        Args:
            events: List of earthquake event dictionaries
            start_time: Start of time period
            end_time: End of time period
        
        Returns:
            Dictionary with all calculated quantities and alerts
        """
        if not events:
            return {
                "error": "No events to process",
                "num_events": 0
            }
        
        # Extract data
        magnitudes = np.array([e["magnitude"] for e in events])
        longitudes = np.array([e["longitude"] for e in events])
        latitudes = np.array([e["latitude"] for e in events])
        depths = np.array([e["depth_km"] for e in events])
        
        # Calculate seismic moments
        seismic_moments = self.moment_calc.magnitude_to_moment(magnitudes)
        total_moment = np.sum(seismic_moments)
        
        # Perform triangulation
        tri_result = self.triangulation.triangulate(
            longitudes, latitudes, depths, magnitudes
        )
        
        # Calculate time span
        time_span_years = (end_time - start_time).total_seconds() / self.SECONDS_PER_YEAR
        
        # Calculate strain rate
        volume_m3 = tri_result["total_volume_m3"]
        strain_rate = self.calculate_strain_rate(
            seismic_moments, volume_m3, time_span_years
        )
        
        # Check thresholds and generate alerts
        alerts = []
        
        if strain_rate > self.strain_rate_threshold:
            alerts.append({
                "type": "ELEVATED_STRAIN_RATE",
                "severity": "WARNING",
                "message": f"Strain rate {strain_rate:.2e}/year exceeds threshold {self.strain_rate_threshold:.2e}/year",
                "value": strain_rate,
                "threshold": self.strain_rate_threshold,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        
        # Compile results
        results = {
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "time_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_years": time_span_years,
            },
            "event_statistics": {
                "num_events": len(events),
                "magnitude_range": [float(magnitudes.min()), float(magnitudes.max())],
                "depth_range_km": [float(depths.min()), float(depths.max())],
            },
            "seismic_moment": {
                "total_Nm": float(total_moment),
                "equivalent_magnitude": float(self.moment_calc.moment_to_magnitude(total_moment)),
            },
            "spatial_analysis": {
                "num_triangles": tri_result["num_triangles"],
                "total_area_km2": tri_result["total_area_m2"] / 1e6,
                "total_volume_km3": tri_result["total_volume_m3"] / 1e9,
            },
            "strain_rate": {
                "value_per_year": float(strain_rate),
                "threshold_per_year": self.strain_rate_threshold,
                "exceeds_threshold": strain_rate > self.strain_rate_threshold,
            },
            "alerts": alerts,
            "provenance": {
                "rigidity_Pa": self.RIGIDITY_MU,
                "calculation_method": "Kostrov (1974)",
                "magnitude_formula": "Hanks & Kanamori (1979)",
            }
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save calculation results with provenance."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    config = {
        "thresholds": {
            "strain_rate_per_year": 1.0e-7,
            "fault_proximity_km": 10.0,
        }
    }
    
    engine = PhysicsEngine(config)
    
    # Sample events
    events = [
        {"magnitude": 5.0, "longitude": -118.0, "latitude": 34.0, "depth_km": 10.0},
        {"magnitude": 5.5, "longitude": -118.5, "latitude": 34.5, "depth_km": 15.0},
        {"magnitude": 4.8, "longitude": -117.5, "latitude": 33.8, "depth_km": 8.0},
    ]
    
    from datetime import datetime, timezone
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    
    results = engine.process_earthquake_data(events, start, end)
    
    print("Physics Engine Results:")
    print(f"  Events: {results['event_statistics']['num_events']}")
    print(f"  Strain Rate: {results['strain_rate']['value_per_year']:.2e}/year")
    print(f"  Alerts: {len(results['alerts'])}")
