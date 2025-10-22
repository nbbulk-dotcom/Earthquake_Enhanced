"""
Delaunay Triangulation for Earthquake Epicenters

Creates triangular mesh from earthquake locations to calculate:
- Triangle areas
- Volumes (using depth)
- Spatial distribution of seismic activity
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple, Dict, Any
import pyproj


class EarthquakeTriangulation:
    """Perform Delaunay triangulation on earthquake epicenters."""
    
    def __init__(self, projection: str = "epsg:3857"):
        """
        Initialize triangulation calculator.
        
        Args:
            projection: Coordinate reference system for calculations
                       Default: Web Mercator (epsg:3857)
        """
        self.projection = projection
        self.transformer = pyproj.Transformer.from_crs(
            "epsg:4326",  # WGS84 (lat/lon)
            projection,
            always_xy=True
        )
    
    def project_coordinates(
        self, 
        longitudes: np.ndarray, 
        latitudes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project geographic coordinates to planar coordinates.
        
        Args:
            longitudes: Array of longitude values
            latitudes: Array of latitude values
        
        Returns:
            Tuple of (x, y) projected coordinates in meters
        """
        x, y = self.transformer.transform(longitudes, latitudes)
        return np.array(x), np.array(y)
    
    def triangulate(
        self,
        longitudes: np.ndarray,
        latitudes: np.ndarray,
        depths: np.ndarray,
        magnitudes: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Perform Delaunay triangulation on earthquake locations.
        
        Args:
            longitudes: Array of longitude values
            latitudes: Array of latitude values
            depths: Array of depth values (km)
            magnitudes: Array of magnitudes
        
        Returns:
            Dictionary containing triangulation results
        """
        # Project to planar coordinates
        x, y = self.project_coordinates(longitudes, latitudes)
        
        # Create 2D points for triangulation
        points_2d = np.column_stack([x, y])
        
        # Perform Delaunay triangulation
        tri = Delaunay(points_2d)
        
        # Calculate triangle properties
        triangles = []
        for simplex in tri.simplices:
            # Get vertices
            v0, v1, v2 = simplex
            
            # Calculate area using cross product
            p0 = points_2d[v0]
            p1 = points_2d[v1]
            p2 = points_2d[v2]
            
            area = 0.5 * abs(
                (p1[0] - p0[0]) * (p2[1] - p0[1]) - 
                (p2[0] - p0[0]) * (p1[1] - p0[1])
            )
            
            # Calculate average depth
            avg_depth = np.mean([depths[v0], depths[v1], depths[v2]])
            
            # Calculate volume (area × depth)
            volume = area * avg_depth * 1000  # Convert km to m
            
            # Calculate centroid
            centroid_x = np.mean([x[v0], x[v1], x[v2]])
            centroid_y = np.mean([y[v0], y[v1], y[v2]])
            
            # Back-project centroid to lat/lon
            centroid_lon, centroid_lat = self.transformer.transform(
                centroid_x, centroid_y, direction="INVERSE"
            )
            
            triangles.append({
                "vertices": simplex.tolist(),
                "area_m2": area,
                "volume_m3": volume,
                "avg_depth_km": avg_depth,
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                "vertex_magnitudes": [
                    magnitudes[v0], magnitudes[v1], magnitudes[v2]
                ],
            })
        
        return {
            "num_points": len(longitudes),
            "num_triangles": len(triangles),
            "triangles": triangles,
            "total_area_m2": sum(t["area_m2"] for t in triangles),
            "total_volume_m3": sum(t["volume_m3"] for t in triangles),
        }


if __name__ == "__main__":
    # Example usage
    tri_calc = EarthquakeTriangulation()
    
    # Sample earthquake data (California region)
    lons = np.array([-118.5, -119.0, -118.0, -117.5, -119.5])
    lats = np.array([34.0, 34.5, 34.2, 33.8, 34.8])
    depths = np.array([10.0, 15.0, 8.0, 12.0, 20.0])
    mags = np.array([4.5, 5.0, 4.2, 4.8, 5.2])
    
    result = tri_calc.triangulate(lons, lats, depths, mags)
    
    print(f"Triangulation Results:")
    print(f"  Points: {result['num_points']}")
    print(f"  Triangles: {result['num_triangles']}")
    print(f"  Total Area: {result['total_area_m2']/1e6:.2f} km²")
    print(f"  Total Volume: {result['total_volume_m3']/1e9:.2f} km³")
