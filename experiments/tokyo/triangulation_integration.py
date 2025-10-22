
#!/usr/bin/env python3
"""
Delaunay Triangulation Integration Module
Creates spatial triangulation with R-tree fault indexing for earthquake analysis.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rtree import index
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon
from pyproj import Geod


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km."""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def compute_triangle_area(vertices):
    """Compute triangle area in km² using geodesic calculations."""
    geod = Geod(ellps="WGS84")
    lons = [v[0] for v in vertices]
    lats = [v[1] for v in vertices]
    area, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area) / 1e6  # Convert m² to km²


def create_delaunay_triangulation(stations, max_edge_km=200):
    """
    Create Delaunay triangulation from station coordinates.
    
    Args:
        stations: List of (lat, lon) tuples
        max_edge_km: Maximum edge length in km (filters out large triangles)
    
    Returns:
        List of triangle dictionaries with metadata
    """
    if len(stations) < 3:
        raise ValueError("Need at least 3 stations for triangulation")
    
    # Convert to numpy array (lon, lat for Delaunay)
    points = np.array([(lon, lat) for lat, lon in stations])
    
    # Compute Delaunay triangulation
    tri = Delaunay(points)
    
    triangles = []
    for simplex in tri.simplices:
        # Get vertices (lon, lat)
        vertices = points[simplex]
        
        # Compute edge lengths
        edge_lengths = []
        for i in range(3):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 3]
            dist = haversine_distance(v1[1], v1[0], v2[1], v2[0])
            edge_lengths.append(dist)
        
        max_edge = max(edge_lengths)
        
        # Filter by max edge length
        if max_edge > max_edge_km:
            continue
        
        # Compute centroid
        centroid_lon = vertices[:, 0].mean()
        centroid_lat = vertices[:, 1].mean()
        
        # Compute area
        area_km2 = compute_triangle_area([(v[0], v[1]) for v in vertices])
        
        triangle = {
            "id": len(triangles),
            "vertices": vertices.tolist(),
            "centroid": [centroid_lon, centroid_lat],
            "area_km2": area_km2,
            "max_edge_km": max_edge,
            "edge_lengths_km": edge_lengths
        }
        
        triangles.append(triangle)
    
    return triangles


def build_fault_rtree(faults):
    """
    Build R-tree spatial index for fault lines.
    
    Args:
        faults: List of fault dictionaries with 'geometry' (LineString coordinates)
    
    Returns:
        R-tree index
    """
    idx = index.Index()
    
    for i, fault in enumerate(faults):
        coords = fault["geometry"]["coordinates"]
        # Compute bounding box
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bbox = (min(lons), min(lats), max(lons), max(lats))
        idx.insert(i, bbox, obj=fault)
    
    return idx


def map_triangles_to_faults(triangles, fault_index):
    """
    Map each triangle to nearest fault using R-tree index.
    
    Args:
        triangles: List of triangle dictionaries
        fault_index: R-tree index of faults
    
    Returns:
        Dictionary mapping triangle_id to nearest fault info
    """
    triangle_fault_map = {}
    
    for triangle in triangles:
        centroid = triangle["centroid"]
        point = Point(centroid[0], centroid[1])
        
        # Query nearest faults
        nearest = list(fault_index.nearest((centroid[0], centroid[1], centroid[0], centroid[1]), 1, objects=True))
        
        if nearest:
            fault = nearest[0].object
            triangle_fault_map[triangle["id"]] = {
                "fault_id": fault.get("id", "unknown"),
                "fault_name": fault.get("name", "unknown"),
                "distance_km": 0.0  # Placeholder (would need actual distance calculation)
            }
        else:
            triangle_fault_map[triangle["id"]] = None
    
    return triangle_fault_map


def aggregate_events_by_triangle(events_df, triangles):
    """
    Aggregate earthquake events by triangle and day.
    
    Args:
        events_df: DataFrame with columns [time, latitude, longitude, mag]
        triangles: List of triangle dictionaries
    
    Returns:
        DataFrame with triangle-day aggregates
    """
    # Create Shapely polygons for triangles
    triangle_polygons = []
    for tri in triangles:
        vertices = tri["vertices"]
        poly = Polygon([(v[0], v[1]) for v in vertices])
        triangle_polygons.append(poly)
    
    # Assign events to triangles
    events_df["triangle_id"] = -1
    
    for idx, row in events_df.iterrows():
        point = Point(row["longitude"], row["latitude"])
        for tri_id, poly in enumerate(triangle_polygons):
            if poly.contains(point):
                events_df.at[idx, "triangle_id"] = tri_id
                break
    
    # Filter events assigned to triangles
    assigned_events = events_df[events_df["triangle_id"] >= 0].copy()
    
    # Create daily aggregates
    assigned_events["date"] = assigned_events["time"].dt.date
    
    daily_agg = assigned_events.groupby(["triangle_id", "date"]).agg({
        "mag": ["count", "sum", "mean", "max"]
    }).reset_index()
    
    daily_agg.columns = ["triangle_id", "date", "count", "sum_mag", "mean_mag", "max_mag"]
    
    # Compute sum of log(M0)
    def Mw_to_M0(Mw):
        return 10 ** (1.5 * Mw + 9.1)
    
    assigned_events["logM0"] = assigned_events["mag"].apply(lambda m: np.log10(Mw_to_M0(m)))
    logM0_agg = assigned_events.groupby(["triangle_id", "date"])["logM0"].sum().reset_index()
    logM0_agg.columns = ["triangle_id", "date", "sum_logM0"]
    
    daily_agg = daily_agg.merge(logM0_agg, on=["triangle_id", "date"], how="left")
    
    return daily_agg


def save_triangulation_artifacts(triangles, triangle_fault_map, daily_agg, output_dir="experiments/tokyo/artifacts"):
    """Save triangulation artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save triangles as JSON
    triangles_json = output_path / "triangles.json"
    with open(triangles_json, "w") as f:
        json.dump(triangles, f, indent=2)
    
    print(f"Triangles saved to: {triangles_json}")
    
    # Save triangles as Parquet
    triangles_df = pd.DataFrame(triangles)
    triangles_parquet = output_path / "triangles.parquet"
    triangles_df.to_parquet(triangles_parquet, index=False)
    
    print(f"Triangles (Parquet) saved to: {triangles_parquet}")
    
    # Save triangle-fault mapping
    fault_map_json = output_path / "triangle_fault_index.json"
    with open(fault_map_json, "w") as f:
        json.dump(triangle_fault_map, f, indent=2)
    
    print(f"Triangle-fault mapping saved to: {fault_map_json}")
    
    # Save daily aggregates
    daily_agg_csv = output_path / "triangle_daily_agg.csv"
    daily_agg.to_csv(daily_agg_csv, index=False)
    
    print(f"Triangle daily aggregates saved to: {daily_agg_csv}")


def main():
    """Example usage of triangulation module."""
    # Example stations (Tokyo region)
    stations = [
        (35.6762, 139.6503),  # Tokyo
        (35.4437, 139.6380),  # Yokohama
        (35.0116, 135.7681),  # Kyoto
        (34.6937, 135.5023),  # Osaka
        (36.2048, 138.2529),  # Nagano
        (36.5651, 136.6562),  # Kanazawa
    ]
    
    print("Creating Delaunay triangulation...")
    triangles = create_delaunay_triangulation(stations, max_edge_km=200)
    print(f"Created {len(triangles)} triangles")
    
    # Example faults (placeholder)
    faults = [
        {
            "id": "fault_1",
            "name": "Sagami Trough",
            "geometry": {
                "coordinates": [[139.0, 35.0], [139.5, 35.5], [140.0, 36.0]]
            }
        }
    ]
    
    print("\nBuilding fault R-tree index...")
    fault_index = build_fault_rtree(faults)
    
    print("Mapping triangles to faults...")
    triangle_fault_map = map_triangles_to_faults(triangles, fault_index)
    
    # Example events (placeholder)
    events_data = {
        "time": pd.date_range("2024-10-01", periods=100, freq="D"),
        "latitude": np.random.uniform(34.5, 36.5, 100),
        "longitude": np.random.uniform(138.0, 140.0, 100),
        "mag": np.random.uniform(3.5, 6.0, 100)
    }
    events_df = pd.DataFrame(events_data)
    
    print("\nAggregating events by triangle...")
    daily_agg = aggregate_events_by_triangle(events_df, triangles)
    print(f"Created {len(daily_agg)} triangle-day records")
    
    print("\nSaving artifacts...")
    save_triangulation_artifacts(triangles, triangle_fault_map, daily_agg)
    
    print("\nTriangulation integration complete!")


if __name__ == "__main__":
    main()
