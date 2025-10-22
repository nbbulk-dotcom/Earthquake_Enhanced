"""
Feature extraction utilities.

Note: These are for descriptive analysis only, NOT for ML prediction.
"""

import numpy as np
from typing import List, Dict, Any


def calculate_event_rate(timestamps: List[float], window_days: float = 30.0) -> float:
    """
    Calculate event rate (events per day).
    
    Args:
        timestamps: List of Unix timestamps
        window_days: Time window in days
    
    Returns:
        Event rate (events/day)
    """
    if not timestamps:
        return 0.0
    
    time_span_seconds = (max(timestamps) - min(timestamps))
    time_span_days = time_span_seconds / (24 * 3600)
    
    if time_span_days == 0:
        return 0.0
    
    return len(timestamps) / time_span_days


def calculate_epicenter_density(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    area_km2: float
) -> float:
    """
    Calculate epicenter density (events per km²).
    
    Args:
        longitudes: Array of longitudes
        latitudes: Array of latitudes
        area_km2: Total area in km²
    
    Returns:
        Density (events/km²)
    """
    num_events = len(longitudes)
    return num_events / area_km2 if area_km2 > 0 else 0.0


def extract_descriptive_statistics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract descriptive statistics from events.
    
    Note: These are for analysis only, NOT for prediction.
    
    Args:
        events: List of earthquake event dictionaries
    
    Returns:
        Dictionary of descriptive statistics
    """
    if not events:
        return {}
    
    magnitudes = np.array([e["magnitude"] for e in events])
    depths = np.array([e["depth_km"] for e in events])
    
    return {
        "num_events": len(events),
        "magnitude_stats": {
            "min": float(magnitudes.min()),
            "max": float(magnitudes.max()),
            "mean": float(magnitudes.mean()),
            "median": float(np.median(magnitudes)),
            "std": float(magnitudes.std()),
        },
        "depth_stats": {
            "min": float(depths.min()),
            "max": float(depths.max()),
            "mean": float(depths.mean()),
            "median": float(np.median(depths)),
            "std": float(depths.std()),
        },
    }
