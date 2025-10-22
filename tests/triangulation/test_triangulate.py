"""
Unit tests for Delaunay triangulation.

Tests spatial analysis and triangle calculations.
"""

import pytest
import numpy as np
from backend.triangulation.triangulate import EarthquakeTriangulation


class TestEarthquakeTriangulation:
    """Test suite for earthquake triangulation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tri = EarthquakeTriangulation()
    
    def test_project_coordinates(self):
        """Test coordinate projection."""
        lons = np.array([-118.0, -119.0])
        lats = np.array([34.0, 35.0])
        
        x, y = self.tri.project_coordinates(lons, lats)
        
        assert len(x) == 2
        assert len(y) == 2
        assert x[0] != lons[0]  # Should be projected
        assert y[0] != lats[0]
    
    def test_triangulate_minimum_points(self):
        """Test triangulation with minimum points (3)."""
        lons = np.array([-118.0, -119.0, -118.5])
        lats = np.array([34.0, 35.0, 34.5])
        depths = np.array([10.0, 15.0, 12.0])
        mags = np.array([4.0, 4.5, 4.2])
        
        result = self.tri.triangulate(lons, lats, depths, mags)
        
        assert result["num_points"] == 3
        assert result["num_triangles"] == 1
        assert result["total_area_m2"] > 0
        assert result["total_volume_m3"] > 0
    
    def test_triangulate_multiple_triangles(self):
        """Test triangulation with multiple triangles."""
        lons = np.array([-118.0, -119.0, -118.5, -117.5])
        lats = np.array([34.0, 35.0, 34.5, 34.2])
        depths = np.array([10.0, 15.0, 12.0, 8.0])
        mags = np.array([4.0, 4.5, 4.2, 4.1])
        
        result = self.tri.triangulate(lons, lats, depths, mags)
        
        assert result["num_points"] == 4
        assert result["num_triangles"] >= 1
        assert len(result["triangles"]) == result["num_triangles"]
    
    def test_triangle_properties(self):
        """Test that triangle properties are calculated correctly."""
        lons = np.array([-118.0, -119.0, -118.5])
        lats = np.array([34.0, 35.0, 34.5])
        depths = np.array([10.0, 15.0, 12.0])
        mags = np.array([4.0, 4.5, 4.2])
        
        result = self.tri.triangulate(lons, lats, depths, mags)
        triangle = result["triangles"][0]
        
        assert "vertices" in triangle
        assert "area_m2" in triangle
        assert "volume_m3" in triangle
        assert "avg_depth_km" in triangle
        assert "centroid_lon" in triangle
        assert "centroid_lat" in triangle
        assert len(triangle["vertices"]) == 3
        assert triangle["area_m2"] > 0
        assert triangle["volume_m3"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
