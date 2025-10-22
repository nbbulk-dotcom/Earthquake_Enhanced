
#!/usr/bin/env python3
"""
Unit and integration tests for triangulation module.
Uses synthetic fixtures for fast CI execution.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.tokyo.triangulation_integration import (
    haversine_distance,
    create_delaunay_triangulation,
    compute_triangle_area,
    aggregate_events_by_triangle
)


class TestHaversineDistance(unittest.TestCase):
    """Test haversine distance calculation."""
    
    def test_same_point(self):
        """Distance between same point should be 0."""
        dist = haversine_distance(35.6762, 139.6503, 35.6762, 139.6503)
        self.assertAlmostEqual(dist, 0.0, places=2)
    
    def test_known_distance(self):
        """Test known distance (Tokyo to Yokohama ~30 km)."""
        dist = haversine_distance(35.6762, 139.6503, 35.4437, 139.6380)
        self.assertAlmostEqual(dist, 26.0, delta=5.0)  # Allow 5km tolerance


class TestDelaunayTriangulation(unittest.TestCase):
    """Test Delaunay triangulation creation."""
    
    def test_minimum_stations(self):
        """Test with minimum 3 stations."""
        stations = [
            (35.0, 139.0),
            (36.0, 139.0),
            (35.5, 140.0)
        ]
        triangles = create_delaunay_triangulation(stations, max_edge_km=500)
        self.assertGreaterEqual(len(triangles), 1)
    
    def test_insufficient_stations(self):
        """Test error with < 3 stations."""
        stations = [(35.0, 139.0), (36.0, 139.0)]
        with self.assertRaises(ValueError):
            create_delaunay_triangulation(stations)
    
    def test_max_edge_filter(self):
        """Test max edge length filtering."""
        # Create stations with one very far point
        stations = [
            (35.0, 139.0),
            (35.1, 139.1),
            (35.2, 139.2),
            (40.0, 145.0)  # Very far point
        ]
        triangles_unfiltered = create_delaunay_triangulation(stations, max_edge_km=1000)
        triangles_filtered = create_delaunay_triangulation(stations, max_edge_km=50)
        
        # Filtered should have fewer triangles
        self.assertLessEqual(len(triangles_filtered), len(triangles_unfiltered))
    
    def test_triangle_metadata(self):
        """Test triangle metadata fields."""
        stations = [
            (35.0, 139.0),
            (36.0, 139.0),
            (35.5, 140.0)
        ]
        triangles = create_delaunay_triangulation(stations, max_edge_km=500)
        
        for tri in triangles:
            self.assertIn("id", tri)
            self.assertIn("vertices", tri)
            self.assertIn("centroid", tri)
            self.assertIn("area_km2", tri)
            self.assertIn("max_edge_km", tri)
            self.assertIn("edge_lengths_km", tri)
            
            # Check data types
            self.assertIsInstance(tri["id"], int)
            self.assertIsInstance(tri["vertices"], list)
            self.assertIsInstance(tri["centroid"], list)
            self.assertIsInstance(tri["area_km2"], float)
            self.assertIsInstance(tri["max_edge_km"], float)
            
            # Check centroid is within bounds
            self.assertGreaterEqual(tri["centroid"][0], 138.5)
            self.assertLessEqual(tri["centroid"][0], 140.5)


class TestEventAggregation(unittest.TestCase):
    """Test event aggregation by triangle."""
    
    def test_event_assignment(self):
        """Test events are correctly assigned to triangles."""
        # Create simple triangulation
        stations = [
            (35.0, 139.0),
            (36.0, 139.0),
            (35.5, 140.0)
        ]
        triangles = create_delaunay_triangulation(stations, max_edge_km=500)
        
        # Create synthetic events within triangle bounds
        events_data = {
            "time": pd.date_range("2024-10-01", periods=10, freq="D"),
            "latitude": np.random.uniform(35.0, 36.0, 10),
            "longitude": np.random.uniform(139.0, 140.0, 10),
            "mag": np.random.uniform(3.5, 5.0, 10)
        }
        events_df = pd.DataFrame(events_data)
        
        # Aggregate
        daily_agg = aggregate_events_by_triangle(events_df, triangles)
        
        # Check output
        self.assertIsInstance(daily_agg, pd.DataFrame)
        self.assertIn("triangle_id", daily_agg.columns)
        self.assertIn("date", daily_agg.columns)
        self.assertIn("count", daily_agg.columns)
        self.assertIn("sum_logM0", daily_agg.columns)
    
    def test_empty_events(self):
        """Test with no events."""
        stations = [
            (35.0, 139.0),
            (36.0, 139.0),
            (35.5, 140.0)
        ]
        triangles = create_delaunay_triangulation(stations, max_edge_km=500)
        
        # Empty events
        events_df = pd.DataFrame(columns=["time", "latitude", "longitude", "mag"])
        
        daily_agg = aggregate_events_by_triangle(events_df, triangles)
        
        # Should return empty DataFrame with correct columns
        self.assertEqual(len(daily_agg), 0)
        self.assertIn("triangle_id", daily_agg.columns)


class TestTriangleArea(unittest.TestCase):
    """Test triangle area calculation."""
    
    def test_small_triangle(self):
        """Test area of small triangle."""
        vertices = [
            (139.0, 35.0),
            (139.1, 35.0),
            (139.05, 35.1)
        ]
        area = compute_triangle_area(vertices)
        
        # Should be positive and reasonable (< 1000 km²)
        self.assertGreater(area, 0)
        self.assertLess(area, 1000)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHaversineDistance))
    suite.addTests(loader.loadTestsFromTestCase(TestDelaunayTriangulation))
    suite.addTests(loader.loadTestsFromTestCase(TestEventAggregation))
    suite.addTests(loader.loadTestsFromTestCase(TestTriangleArea))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
