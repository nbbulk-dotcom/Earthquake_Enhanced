"""
Triangle Feature Computation Module

This module computes geometric triangle features from earthquake triplets
to identify spatial patterns that may precede larger events.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from math import radians, sin, cos, sqrt, atan2
from typing import List, Tuple, Dict
import yaml
from pathlib import Path


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points on earth.
    
    Args:
        lon1, lat1: Coordinates of first point (degrees)
        lon2, lat2: Coordinates of second point (degrees)
    
    Returns:
        Distance in kilometers
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r


def calculate_triangle_area(p1: Tuple[float, float], 
                           p2: Tuple[float, float], 
                           p3: Tuple[float, float]) -> float:
    """
    Calculate the area of a triangle using Heron's formula.
    
    Args:
        p1, p2, p3: Triangle vertices as (lon, lat) tuples
    
    Returns:
        Area in square kilometers
    """
    # Calculate side lengths
    a = haversine_distance(p1[0], p1[1], p2[0], p2[1])
    b = haversine_distance(p2[0], p2[1], p3[0], p3[1])
    c = haversine_distance(p3[0], p3[1], p1[0], p1[1])
    
    # Heron's formula
    s = (a + b + c) / 2
    area = sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
    
    return area


def calculate_triangle_metrics(p1: Tuple[float, float], 
                              p2: Tuple[float, float], 
                              p3: Tuple[float, float]) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for a triangle.
    
    Args:
        p1, p2, p3: Triangle vertices as (lon, lat) tuples
    
    Returns:
        Dictionary of triangle metrics
    """
    # Side lengths
    a = haversine_distance(p1[0], p1[1], p2[0], p2[1])
    b = haversine_distance(p2[0], p2[1], p3[0], p3[1])
    c = haversine_distance(p3[0], p3[1], p1[0], p1[1])
    
    # Area
    area = calculate_triangle_area(p1, p2, p3)
    
    # Perimeter
    perimeter = a + b + c
    
    # Aspect ratio (elongation measure)
    max_side = max(a, b, c)
    min_side = min(a, b, c)
    aspect_ratio = max_side / min_side if min_side > 0 else float('inf')
    
    # Compactness (circularity)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # Centroid
    centroid_lon = (p1[0] + p2[0] + p3[0]) / 3
    centroid_lat = (p1[1] + p2[1] + p3[1]) / 3
    
    return {
        'area_km2': area,
        'perimeter_km': perimeter,
        'aspect_ratio': aspect_ratio,
        'compactness': compactness,
        'max_side_km': max_side,
        'min_side_km': min_side,
        'centroid_lon': centroid_lon,
        'centroid_lat': centroid_lat
    }


def find_earthquake_triangles(events_df: pd.DataFrame, 
                              time_window_days: int = 30,
                              min_magnitude: float = 3.5,
                              max_distance_km: float = 500,
                              min_distance_km: float = 10) -> pd.DataFrame:
    """
    Find all valid earthquake triangles within specified constraints.
    
    Args:
        events_df: DataFrame with columns ['time', 'lat', 'lon', 'mag']
        time_window_days: Maximum time span for triangle formation (days)
        min_magnitude: Minimum magnitude for triangle vertices
        max_distance_km: Maximum distance between any two vertices
        min_distance_km: Minimum distance between any two vertices
    
    Returns:
        DataFrame of triangles with their metrics
    """
    # Filter by magnitude
    events = events_df[events_df['mag'] >= min_magnitude].copy()
    events = events.sort_values('time').reset_index(drop=True)
    
    triangles = []
    
    # Iterate through all possible triplets
    for i, event1 in events.iterrows():
        # Only consider events within time window
        time_mask = (events['time'] >= event1['time']) & \
                    (events['time'] <= event1['time'] + pd.Timedelta(days=time_window_days))
        
        candidates = events[time_mask & (events.index > i)].copy()
        
        if len(candidates) < 2:
            continue
        
        # Check all pairs within candidates
        for idx_pair in combinations(candidates.index, 2):
            event2 = events.loc[idx_pair[0]]
            event3 = events.loc[idx_pair[1]]
            
            p1 = (event1['lon'], event1['lat'])
            p2 = (event2['lon'], event2['lat'])
            p3 = (event3['lon'], event3['lat'])
            
            # Check distance constraints
            d12 = haversine_distance(p1[0], p1[1], p2[0], p2[1])
            d23 = haversine_distance(p2[0], p2[1], p3[0], p3[1])
            d31 = haversine_distance(p3[0], p3[1], p1[0], p1[1])
            
            if (min_distance_km <= d12 <= max_distance_km and
                min_distance_km <= d23 <= max_distance_km and
                min_distance_km <= d31 <= max_distance_km):
                
                # Calculate triangle metrics
                metrics = calculate_triangle_metrics(p1, p2, p3)
                
                # Store triangle information
                triangle = {
                    'event1_id': event1.get('id', f'e{i}'),
                    'event2_id': event2.get('id', f'e{idx_pair[0]}'),
                    'event3_id': event3.get('id', f'e{idx_pair[1]}'),
                    'time_start': event1['time'],
                    'time_end': event3['time'],
                    'time_span_days': (event3['time'] - event1['time']).total_seconds() / 86400,
                    'mag_sum': event1['mag'] + event2['mag'] + event3['mag'],
                    'mag_mean': (event1['mag'] + event2['mag'] + event3['mag']) / 3,
                    'mag_max': max(event1['mag'], event2['mag'], event3['mag']),
                    **metrics
                }
                
                triangles.append(triangle)
    
    return pd.DataFrame(triangles)


def compute_triangle_features(events_df: pd.DataFrame,
                              daily_df: pd.DataFrame,
                              config_path: str = 'backend/features/config.yaml') -> pd.DataFrame:
    """
    Compute triangle-based features for daily earthquake prediction.
    
    Args:
        events_df: DataFrame with individual earthquake events
        daily_df: DataFrame with daily aggregated features
        config_path: Path to configuration file
    
    Returns:
        Updated daily_df with triangle features
    """
    # Load configuration
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        tri_config = config.get('triangulation', {})
    else:
        tri_config = {
            'min_magnitude': 3.5,
            'time_window': 30,
            'max_distance_km': 500,
            'min_distance_km': 10
        }
    
    # Find all triangles
    triangles = find_earthquake_triangles(
        events_df,
        time_window_days=tri_config.get('time_window', 30),
        min_magnitude=tri_config.get('min_magnitude', 3.5),
        max_distance_km=tri_config.get('max_distance_km', 500),
        min_distance_km=tri_config.get('min_distance_km', 10)
    )
    
    # Aggregate triangle features by day
    daily_df = daily_df.copy()
    daily_df['tri_count_30d'] = 0
    daily_df['tri_mean_area_30d'] = 0.0
    daily_df['tri_mean_compactness_30d'] = 0.0
    daily_df['tri_max_mag_30d'] = 0.0
    
    for i, date in enumerate(daily_df.index):
        # Get triangles formed in past 30 days
        past_30_days = date - pd.Timedelta(days=30)
        
        recent_triangles = triangles[
            (triangles['time_end'] >= past_30_days) & 
            (triangles['time_end'] < date)
        ]
        
        if len(recent_triangles) > 0:
            daily_df.loc[date, 'tri_count_30d'] = len(recent_triangles)
            daily_df.loc[date, 'tri_mean_area_30d'] = recent_triangles['area_km2'].mean()
            daily_df.loc[date, 'tri_mean_compactness_30d'] = recent_triangles['compactness'].mean()
            daily_df.loc[date, 'tri_max_mag_30d'] = recent_triangles['mag_max'].max()
    
    return daily_df


if __name__ == '__main__':
    # Example usage
    print("Triangle Feature Computation Module")
    print("=" * 60)
    
    # This would typically be called from the main experiment script
    # Example:
    # triangles = find_earthquake_triangles(events_df)
    # print(f"Found {len(triangles)} triangles")
    # daily_with_tri = compute_triangle_features(events_df, daily_df)
    
    print("Module loaded successfully.")
    print("Use find_earthquake_triangles() and compute_triangle_features()")
    print("to compute triangle-based spatial features.")
