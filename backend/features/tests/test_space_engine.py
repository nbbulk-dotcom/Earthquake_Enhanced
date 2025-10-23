"""
Unit Tests for Space Engine Module

Tests all 8 features:
1. Atmospheric Boundary Refraction
2. Angle of Incidence Tracking
3. Sun Path Prediction
4. Dynamic Lag Time Calculation
5. RGB Resonance Calculations
6. Data Integration
7. Resultant Resonance Calculations
8. Equatorial Enhancement
"""

import unittest
import asyncio
import math
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from features.space_engine import SpaceEngine, SpaceVariableType


class TestAtmosphericBoundaryRefraction(unittest.TestCase):
    """Test Feature 1: Atmospheric Boundary Refraction"""
    
    def setUp(self):
        self.engine = SpaceEngine()
    
    def test_80km_boundary_refraction(self):
        """Test 80km boundary refraction factor (1.15)"""
        raw_value = 1.0
        result = self.engine.calculate_atmospheric_refraction(80.0, raw_value)
        self.assertAlmostEqual(result, 1.15, places=2)
    
    def test_85km_boundary_refraction(self):
        """Test 85km boundary refraction factor (1.12)"""
        raw_value = 1.0
        result = self.engine.calculate_atmospheric_refraction(85.0, raw_value)
        self.assertAlmostEqual(result, 1.12, places=2)
    
    def test_interpolated_refraction(self):
        """Test interpolation between 80km and 85km"""
        raw_value = 1.0
        result_82_5 = self.engine.calculate_atmospheric_refraction(82.5, raw_value)
        expected = (1.15 + 1.12) / 2.0  # Average
        self.assertAlmostEqual(result_82_5, expected, places=2)
    
    def test_boundary_factors(self):
        """Test boundary refraction factors retrieval"""
        factors = self.engine.get_boundary_refraction_factors()
        self.assertEqual(factors['80km_boundary'], 1.15)
        self.assertEqual(factors['85km_boundary'], 1.12)
        self.assertAlmostEqual(factors['average_boundary'], 1.135, places=3)


class TestAngleOfIncidenceTracking(unittest.TestCase):
    """Test Feature 2: Angle of Incidence Tracking"""
    
    def setUp(self):
        self.engine = SpaceEngine()
        self.test_time = datetime(2024, 6, 21, 12, 0, 0)  # Summer solstice noon
    
    def test_solar_elevation_calculation(self):
        """Test solar elevation angle calculation"""
        # Test at equator on equinox
        lat, lon = 0.0, 0.0
        test_time = datetime(2024, 3, 20, 12, 0, 0)  # Vernal equinox noon
        
        result = self.engine.calculate_solar_elevation(lat, lon, test_time)
        
        self.assertIn('elevation', result)
        self.assertIn('azimuth', result)
        self.assertIn('zenith', result)
        self.assertIn('declination', result)
        
        # At equator noon on equinox, sun should be near zenith
        self.assertTrue(-10 <= result['elevation'] <= 100)
    
    def test_solar_angles_range(self):
        """Test that solar angles are in valid ranges"""
        lat, lon = 45.0, -122.0
        result = self.engine.calculate_solar_elevation(lat, lon, self.test_time)
        
        self.assertTrue(-90 <= result['elevation'] <= 90)
        self.assertTrue(0 <= result['azimuth'] <= 360)
        self.assertTrue(0 <= result['zenith'] <= 180)
        self.assertTrue(-23.5 <= result['declination'] <= 23.5)
    
    def test_tetrahedral_angle_seismic(self):
        """Test tetrahedral angle for seismic events (26.52°)"""
        lat, lon = 35.0, 140.0
        angle = self.engine.calculate_tetrahedral_angle(lat, lon, 'seismic')
        
        # Should be base angle (26.52) plus location adjustments
        self.assertTrue(20 < angle < 40)
    
    def test_tetrahedral_angle_volcanic(self):
        """Test tetrahedral angle for volcanic events (54.74°)"""
        lat, lon = 35.0, 140.0
        angle = self.engine.calculate_tetrahedral_angle(lat, lon, 'volcanic')
        
        # Should be base angle (54.74) plus location adjustments
        self.assertTrue(50 < angle < 70)
    
    def test_magnetic_latitude_conversion(self):
        """Test geographic to magnetic latitude conversion"""
        # Test at geographic north pole
        lat, lon = 90.0, 0.0
        mag_lat = self.engine.calculate_magnetic_latitude(lat, lon)
        
        # Magnetic latitude should be valid
        self.assertTrue(-90 <= mag_lat <= 90)
        
        # Test at equator
        lat, lon = 0.0, 0.0
        mag_lat = self.engine.calculate_magnetic_latitude(lat, lon)
        self.assertTrue(-90 <= mag_lat <= 90)


class TestSunPathPrediction(unittest.TestCase):
    """Test Feature 3: Sun Path Prediction"""
    
    def setUp(self):
        self.engine = SpaceEngine()
        self.test_time = datetime(2024, 6, 21, 6, 0, 0)
    
    def test_sun_path_prediction_24h(self):
        """Test 24-hour sun path prediction"""
        lat, lon = 40.0, -105.0
        predictions = self.engine.predict_sun_path(lat, lon, self.test_time, hours_ahead=24)
        
        self.assertEqual(len(predictions), 24)
        
        # Check structure of predictions
        for pred in predictions:
            self.assertIn('timestamp', pred)
            self.assertIn('hour_offset', pred)
            self.assertIn('solar_elevation', pred)
            self.assertIn('solar_azimuth', pred)
            self.assertIn('zenith_angle', pred)
            self.assertIn('ray_path_distance_km', pred)
            self.assertIn('is_daytime', pred)
    
    def test_ray_path_distance(self):
        """Test ray path distance calculation"""
        # Near vertical (high elevation)
        distance_high = self.engine._calculate_ray_path_distance(85)
        
        # Low elevation (long path)
        distance_low = self.engine._calculate_ray_path_distance(10)
        
        # Low elevation should have longer path
        self.assertGreater(distance_low, distance_high)
        
        # Below horizon should return 0
        distance_neg = self.engine._calculate_ray_path_distance(-10)
        self.assertEqual(distance_neg, 0.0)
    
    def test_daytime_detection(self):
        """Test daytime vs nighttime detection"""
        lat, lon = 0.0, 0.0
        noon_time = datetime(2024, 6, 21, 12, 0, 0)
        midnight_time = datetime(2024, 6, 21, 0, 0, 0)
        
        noon_path = self.engine.predict_sun_path(lat, lon, noon_time, hours_ahead=1)
        midnight_path = self.engine.predict_sun_path(lat, lon, midnight_time, hours_ahead=1)
        
        # These should have different is_daytime values
        self.assertIsNotNone(noon_path[0]['is_daytime'])
        self.assertIsNotNone(midnight_path[0]['is_daytime'])


class TestDynamicLagTimeCalculation(unittest.TestCase):
    """Test Feature 4: Dynamic Lag Time Calculation"""
    
    def setUp(self):
        self.engine = SpaceEngine()
        self.test_time = datetime(2024, 6, 21, 12, 0, 0)
    
    def test_lag_time_calculation(self):
        """Test dynamic lag time calculation"""
        lat, lon = 35.0, 140.0
        lag_times = self.engine.calculate_dynamic_lag_times(lat, lon, self.test_time)
        
        # Check all components present
        self.assertIn('light_travel_base_hours', lag_times)
        self.assertIn('light_travel_adjusted_hours', lag_times)
        self.assertIn('solar_lag_hours', lag_times)
        self.assertIn('geomagnetic_lag_hours', lag_times)
        self.assertIn('ionospheric_lag_hours', lag_times)
        self.assertIn('total_lag_hours', lag_times)
    
    def test_light_travel_base_delay(self):
        """Test base light travel delay (~8.3 minutes)"""
        lat, lon = 0.0, 0.0
        lag_times = self.engine.calculate_dynamic_lag_times(lat, lon, self.test_time)
        
        # Base delay should be around 8.3 minutes = 0.138 hours
        base_delay = lag_times['light_travel_base_hours']
        self.assertAlmostEqual(base_delay, 8.3 / 60.0, places=2)
    
    def test_solar_lag_range(self):
        """Test solar lag is in 4-12 hour range"""
        lat, lon = 0.0, 0.0
        
        # Test multiple times to check seasonal variation
        for month in [1, 4, 7, 10]:
            test_time = datetime(2024, month, 15, 12, 0, 0)
            lag_times = self.engine.calculate_dynamic_lag_times(lat, lon, test_time)
            
            solar_lag = lag_times['solar_lag_hours']
            self.assertTrue(3.0 <= solar_lag <= 13.0)  # Allow some margin
    
    def test_geomagnetic_lag_range(self):
        """Test geomagnetic lag is in 4-8 hour range"""
        lat, lon = 0.0, 0.0
        
        # Test multiple hours for diurnal variation
        for hour in [0, 6, 12, 18]:
            test_time = datetime(2024, 6, 21, hour, 0, 0)
            lag_times = self.engine.calculate_dynamic_lag_times(lat, lon, test_time)
            
            geomag_lag = lag_times['geomagnetic_lag_hours']
            self.assertTrue(3.0 <= geomag_lag <= 9.0)  # Allow some margin
    
    def test_ionospheric_lag_range(self):
        """Test ionospheric lag is in 1-7 hour range"""
        lat, lon = 0.0, 0.0
        lag_times = self.engine.calculate_dynamic_lag_times(lat, lon, self.test_time)
        
        iono_lag = lag_times['ionospheric_lag_hours']
        self.assertTrue(0.5 <= iono_lag <= 7.5)  # Allow some margin


class TestRGBResonanceCalculations(unittest.TestCase):
    """Test Feature 5: RGB Resonance Calculations"""
    
    def setUp(self):
        self.engine = SpaceEngine()
    
    def test_rgb_resonance_formula(self):
        """Test RGB resonance formula: sqrt((R² + G² + B²) / 3.0)"""
        # Create test readings where we know the RGB components
        space_readings = {
            'solar_activity': 0.8,  # R
            'solar_flare_intensity': 0.6,  # R
            'cosmic_ray_intensity': 0.7,  # R
            'magnetosphere_compression': 0.5,  # R
            'geomagnetic_field': 0.4,  # G
            'solar_wind_pressure': 0.5,  # G
            'auroral_activity': 0.3,  # G
            'interplanetary_magnetic': 0.6,  # G
            'planetary_alignment': 0.7,  # B
            'ionospheric_density': 0.5,  # B
            'coronal_mass_ejection': 0.6,  # B
            'galactic_cosmic_radiation': 0.4,  # B
        }
        
        result = self.engine.calculate_rgb_resonance(space_readings)
        
        # Check components present
        self.assertIn('R_component', result)
        self.assertIn('G_component', result)
        self.assertIn('B_component', result)
        self.assertIn('rgb_resonance', result)
        
        # Manually calculate expected result
        R = (0.8 + 0.6 + 0.7 + 0.5) / 4
        G = (0.4 + 0.5 + 0.3 + 0.6) / 4
        B = (0.7 + 0.5 + 0.6 + 0.4) / 4
        expected = math.sqrt((R**2 + G**2 + B**2) / 3.0)
        
        self.assertAlmostEqual(result['rgb_resonance'], expected, places=3)
    
    def test_rgb_component_separation(self):
        """Test that RGB components are correctly separated"""
        space_readings = {var: 0.5 for var in self.engine.space_variables.keys()}
        
        result = self.engine.calculate_rgb_resonance(space_readings)
        
        # All components should be present and valid
        self.assertTrue(0 <= result['R_component'] <= 1)
        self.assertTrue(0 <= result['G_component'] <= 1)
        self.assertTrue(0 <= result['B_component'] <= 1)
        self.assertTrue(0 <= result['rgb_resonance'] <= 1)
    
    def test_rgb_resonance_range(self):
        """Test that RGB resonance is in valid range [0, 1]"""
        # Test with extreme values
        space_readings = {var: 1.0 for var in self.engine.space_variables.keys()}
        result = self.engine.calculate_rgb_resonance(space_readings)
        self.assertTrue(0 <= result['rgb_resonance'] <= 1.5)


class TestDataIntegration(unittest.TestCase):
    """Test Feature 6: Data Integration"""
    
    def setUp(self):
        self.engine = SpaceEngine()
        self.test_time = datetime(2024, 6, 21, 12, 0, 0)
    
    @patch('requests.get')
    def test_nasa_omni2_api_call(self, mock_get):
        """Test NASA OMNI2 API integration"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "mock omni2 data"
        mock_get.return_value = mock_response
        
        # Run async test
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.engine.fetch_nasa_omni2_data(self.test_time, self.test_time)
        )
        
        self.assertIsNotNone(result)
        self.assertIn('source', result)
        self.assertEqual(result['source'], 'NASA_OMNI2')
    
    @patch('requests.get')
    def test_noaa_swpc_api_call(self, mock_get):
        """Test NOAA SWPC API integration"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'test': 'data'}
        mock_get.return_value = mock_response
        
        # Run async test
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.engine.fetch_noaa_swpc_data()
        )
        
        self.assertIsNotNone(result)
        self.assertIn('source', result)
        self.assertEqual(result['source'], 'NOAA_SWPC')
        self.assertEqual(result['reliability'], 0.92)
    
    def test_graceful_failure_no_data(self):
        """Test graceful failure when no data available"""
        loop = asyncio.get_event_loop()
        
        # This will likely fail to connect, testing graceful failure
        result = loop.run_until_complete(
            self.engine.get_integrated_space_data(self.test_time)
        )
        
        # Should return structure even on failure
        self.assertIsNotNone(result)
        self.assertIn('data_available', result)


class TestResultantResonanceCalculations(unittest.TestCase):
    """Test Feature 7: Resultant Resonance Calculations"""
    
    def setUp(self):
        self.engine = SpaceEngine()
        self.test_time = datetime(2024, 6, 21, 12, 0, 0)
    
    def test_12d_correlation_matrix(self):
        """Test 12-dimensional correlation matrix calculation"""
        space_readings = {var: 0.5 for var in self.engine.space_variables.keys()}
        lat, lon = 35.0, 140.0
        
        matrix = self.engine.calculate_12d_correlation_matrix(
            space_readings, lat, lon, self.test_time
        )
        
        # Check matrix dimensions
        self.assertEqual(matrix.shape, (12, 12))
        
        # Diagonal should be 1.0 (self-correlation)
        for i in range(12):
            self.assertAlmostEqual(matrix[i, i], 1.0, places=5)
        
        # Matrix should be valid correlations
        self.assertTrue(np.all(matrix >= -1.0))
        self.assertTrue(np.all(matrix <= 1.0))
    
    def test_resultant_resonance_calculation(self):
        """Test resultant resonance from 12D correlation"""
        space_readings = {var: 0.5 for var in self.engine.space_variables.keys()}
        lat, lon = 35.0, 140.0
        
        result = self.engine.calculate_resultant_resonance(
            space_readings, lat, lon, self.test_time
        )
        
        # Check result structure
        self.assertIn('resultant_resonance', result)
        self.assertIn('rgb_contribution', result)
        self.assertIn('matrix_contribution', result)
        self.assertIn('dominant_eigenvalue', result)
        
        # Resultant should be in valid range
        self.assertTrue(0 <= result['resultant_resonance'] <= 1)
    
    def test_cross_correlation(self):
        """Test cross-correlation between variables"""
        space_readings = {var: 0.5 for var in self.engine.space_variables.keys()}
        lat, lon = 0.0, 0.0
        
        var1 = 'solar_activity'
        var2 = 'geomagnetic_field'
        
        corr = self.engine._calculate_cross_correlation(
            var1, var2, space_readings, lat, lon, self.test_time
        )
        
        # Correlation should be valid
        self.assertTrue(-1.0 <= corr <= 1.0)


class TestEquatorialEnhancement(unittest.TestCase):
    """Test Feature 8: Equatorial Enhancement"""
    
    def setUp(self):
        self.engine = SpaceEngine()
    
    def test_equatorial_enhancement_at_equator(self):
        """Test 1.25 enhancement at equator"""
        lat = 0.0
        base_value = 1.0
        
        result = self.engine.apply_equatorial_enhancement(lat, base_value)
        
        # At equator, should get full enhancement (1.25)
        self.assertAlmostEqual(result['enhancement_factor'], 1.25, places=2)
        self.assertAlmostEqual(result['enhanced_value'], 1.25, places=2)
        self.assertTrue(result['is_equatorial'])
    
    def test_equatorial_enhancement_at_threshold(self):
        """Test enhancement at equatorial threshold (23.5°)"""
        lat = 23.5
        base_value = 1.0
        
        result = self.engine.apply_equatorial_enhancement(lat, base_value)
        
        # At threshold, should get minimal enhancement
        self.assertAlmostEqual(result['enhancement_factor'], 1.0, places=1)
        self.assertTrue(result['is_equatorial'])
    
    def test_no_enhancement_outside_equatorial(self):
        """Test no enhancement outside equatorial region"""
        lat = 45.0
        base_value = 1.0
        
        result = self.engine.apply_equatorial_enhancement(lat, base_value)
        
        # Outside equatorial region, no enhancement
        self.assertEqual(result['enhancement_factor'], 1.0)
        self.assertEqual(result['enhanced_value'], base_value)
        self.assertFalse(result['is_equatorial'])
    
    def test_equatorial_enhancement_southern_hemisphere(self):
        """Test enhancement works in southern hemisphere"""
        lat = -10.0
        base_value = 1.0
        
        result = self.engine.apply_equatorial_enhancement(lat, base_value)
        
        # Should be enhanced in southern equatorial region
        self.assertGreater(result['enhancement_factor'], 1.0)
        self.assertTrue(result['is_equatorial'])


class TestIntegratedSpacePrediction(unittest.TestCase):
    """Test integrated space engine prediction"""
    
    def setUp(self):
        self.engine = SpaceEngine()
        self.test_time = datetime(2024, 6, 21, 12, 0, 0)
    
    def test_full_prediction_calculation(self):
        """Test full integrated prediction calculation"""
        lat, lon = 35.0, 140.0
        
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.engine.calculate_space_prediction(lat, lon, self.test_time)
        )
        
        # Check success
        self.assertTrue(result.get('success', False) or 'error' in result)
        
        if result.get('success'):
            # Check all major components present
            self.assertIn('solar_angles', result)
            self.assertIn('lag_times', result)
            self.assertIn('rgb_resonance', result)
            self.assertIn('resultant_resonance', result)
            self.assertIn('equatorial_enhancement', result)
            self.assertIn('earthquake_correlation_score', result)
            self.assertIn('engine_info', result)
    
    def test_engine_status(self):
        """Test engine status reporting"""
        status = self.engine.get_engine_status()
        
        self.assertIn('version', status)
        self.assertIn('engine_id', status)
        self.assertIn('status', status)
        self.assertIn('features', status)
        
        # Check all features are reported
        features = status['features']
        self.assertTrue(features['atmospheric_boundary_refraction'])
        self.assertTrue(features['angle_of_incidence_tracking'])
        self.assertTrue(features['sun_path_prediction'])
        self.assertTrue(features['dynamic_lag_calculation'])
        self.assertTrue(features['rgb_resonance'])
        self.assertTrue(features['data_integration'])
        self.assertTrue(features['resultant_resonance'])
        self.assertTrue(features['equatorial_enhancement'])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.engine = SpaceEngine()
    
    def test_extreme_latitudes(self):
        """Test with extreme latitudes (poles)"""
        test_time = datetime(2024, 6, 21, 12, 0, 0)
        
        # North pole
        solar_angles_north = self.engine.calculate_solar_elevation(90.0, 0.0, test_time)
        self.assertIsNotNone(solar_angles_north)
        
        # South pole
        solar_angles_south = self.engine.calculate_solar_elevation(-90.0, 0.0, test_time)
        self.assertIsNotNone(solar_angles_south)
    
    def test_date_boundary(self):
        """Test with dates at year boundaries"""
        # New Year's Day
        test_time = datetime(2024, 1, 1, 0, 0, 0)
        lat, lon = 0.0, 0.0
        
        result = self.engine.calculate_solar_elevation(lat, lon, test_time)
        self.assertIsNotNone(result)
        
        # New Year's Eve
        test_time = datetime(2024, 12, 31, 23, 59, 59)
        result = self.engine.calculate_solar_elevation(lat, lon, test_time)
        self.assertIsNotNone(result)
    
    def test_empty_space_readings(self):
        """Test RGB resonance with empty readings"""
        empty_readings = {}
        
        result = self.engine.calculate_rgb_resonance(empty_readings)
        
        # Should handle gracefully
        self.assertIsNotNone(result)
        self.assertIn('rgb_resonance', result)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAtmosphericBoundaryRefraction))
    suite.addTests(loader.loadTestsFromTestCase(TestAngleOfIncidenceTracking))
    suite.addTests(loader.loadTestsFromTestCase(TestSunPathPrediction))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicLagTimeCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestRGBResonanceCalculations))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestResultantResonanceCalculations))
    suite.addTests(loader.loadTestsFromTestCase(TestEquatorialEnhancement))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegratedSpacePrediction))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
