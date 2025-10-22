"""
R-tree Spatial Index for Fault Proximity Calculations

Uses R-tree data structure for efficient spatial queries:
- Find nearest fault to earthquake epicenter
- Calculate distance to fault
- Identify earthquakes near active faults
"""

from rtree import index
import numpy as np
from typing import List, Tuple, Dict, Optional
from shapely.geometry import Point, LineString
import pyproj


class FaultIndex:
    """Spatial index for fault line proximity calculations."""
    
    def __init__(self):
        """Initialize R-tree spatial index."""
        self.idx = index.Index()
        self.faults = {}
        self.fault_counter = 0
        
        # Transformer for distance calculations
        self.geod = pyproj.Geod(ellps="WGS84")
    
    def add_fault(
        self,
        fault_id: str,
        coordinates: List[Tuple[float, float]],
        name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a fault line to the spatial index.
        
        Args:
            fault_id: Unique identifier for the fault
            coordinates: List of (lon, lat) tuples defining the fault line
            name: Optional fault name
            metadata: Optional metadata dictionary
        """
        # Create LineString
        line = LineString(coordinates)
        bounds = line.bounds  # (minx, miny, maxx, maxy)
        
        # Add to R-tree index
        self.idx.insert(self.fault_counter, bounds)
        
        # Store fault data
        self.faults[self.fault_counter] = {
            "fault_id": fault_id,
            "name": name or fault_id,
            "geometry": line,
            "coordinates": coordinates,
            "metadata": metadata or {},
        }
        
        self.fault_counter += 1
    
    def find_nearest_fault(
        self,
        longitude: float,
        latitude: float,
        max_distance_km: float = 100.0
    ) -> Optional[Dict]:
        """
        Find the nearest fault to a given point.
        
        Args:
            longitude: Point longitude
            latitude: Point latitude
            max_distance_km: Maximum search distance in kilometers
        
        Returns:
            Dictionary with nearest fault info and distance, or None
        """
        point = Point(longitude, latitude)
        
        # Search in bounding box
        search_buffer = max_distance_km / 111.0  # Rough degrees conversion
        search_bounds = (
            longitude - search_buffer,
            latitude - search_buffer,
            longitude + search_buffer,
            latitude + search_buffer,
        )
        
        nearest_fault = None
        min_distance = float('inf')
        
        # Query R-tree for candidates
        for fault_idx in self.idx.intersection(search_bounds):
            fault_data = self.faults[fault_idx]
            fault_line = fault_data["geometry"]
            
            # Calculate distance using geodesic
            # Project point to line and calculate distance
            nearest_point_on_line = fault_line.interpolate(
                fault_line.project(point)
            )
            
            # Calculate geodesic distance
            _, _, distance_m = self.geod.inv(
                longitude,
                latitude,
                nearest_point_on_line.x,
                nearest_point_on_line.y
            )
            
            distance_km = distance_m / 1000.0
            
            if distance_km < min_distance and distance_km <= max_distance_km:
                min_distance = distance_km
                nearest_fault = {
                    "fault_id": fault_data["fault_id"],
                    "fault_name": fault_data["name"],
                    "distance_km": distance_km,
                    "nearest_point_lon": nearest_point_on_line.x,
                    "nearest_point_lat": nearest_point_on_line.y,
                    "metadata": fault_data["metadata"],
                }
        
        return nearest_fault
    
    def find_earthquakes_near_faults(
        self,
        earthquake_coords: List[Tuple[float, float]],
        max_distance_km: float = 10.0
    ) -> List[Dict]:
        """
        Find earthquakes within specified distance of any fault.
        
        Args:
            earthquake_coords: List of (lon, lat) tuples
            max_distance_km: Maximum distance threshold
        
        Returns:
            List of dictionaries with earthquake index and nearest fault info
        """
        results = []
        
        for idx, (lon, lat) in enumerate(earthquake_coords):
            nearest = self.find_nearest_fault(lon, lat, max_distance_km)
            if nearest:
                results.append({
                    "earthquake_index": idx,
                    "longitude": lon,
                    "latitude": lat,
                    "nearest_fault": nearest,
                })
        
        return results


if __name__ == "__main__":
    # Example usage
    fault_idx = FaultIndex()
    
    # Add San Andreas Fault (simplified)
    san_andreas = [
        (-116.0, 33.0),
        (-117.0, 34.0),
        (-118.0, 35.0),
        (-119.0, 36.0),
    ]
    fault_idx.add_fault(
        "san_andreas",
        san_andreas,
        name="San Andreas Fault",
        metadata={"type": "strike-slip", "length_km": 1200}
    )
    
    # Find nearest fault to a point
    test_lon, test_lat = -117.5, 34.5
    nearest = fault_idx.find_nearest_fault(test_lon, test_lat)
    
    if nearest:
        print(f"Nearest fault to ({test_lon}, {test_lat}):")
        print(f"  Fault: {nearest['fault_name']}")
        print(f"  Distance: {nearest['distance_km']:.2f} km")
