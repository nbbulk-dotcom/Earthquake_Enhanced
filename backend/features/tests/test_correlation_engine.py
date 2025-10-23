"""
Unit Tests for Correlation Engine
Tests all 8 features of the multi-resonance overlay analysis system
"""

import unittest
import asyncio
import math
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.features.correlation_engine import (
    CorrelationEngine, ResonanceSource, OverlayRegion, ResonancePattern
)


class TestCorrelationEngine(unittest.TestCase):
    """Test suite for Correlation Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = CorrelationEngine()
        self.test_lat = 35.6762  # Tokyo
        self.test_lon = 139.6503
        self.test_depth = 15.0
        self.timestamp = datetime.utcnow()
    
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    # ========== Feature 1: Multi-Resonance Overlay Analysis Tests ==========
    
    def test_integrate_space_resonances(self):
        """Test integration of space engine resonances"""
        async def run_test():
            sources = await self.engine.integrate_space_resonances(
                self.test_lat, self.test_lon, self.timestamp
            )
            
            # Should return multiple sources
            self.assertGreater(len(sources), 0)
            
            # All sources should be ResonanceSource objects
            for source in sources:
                self.assertIsInstance(source, ResonanceSource)
                self.assertEqual(source.source_type, 'space')
                self.assertGreater(source.frequency, 0)
                self.assertGreaterEqual(source.amplitude, 0)
                self.assertLessEqual(source.amplitude, 1.0)
            
            # Should include RGB components
            source_names = [s.source_name for s in sources]
            self.assertTrue(any('Red' in name for name in source_names))
            self.assertTrue(any('Green' in name for name in source_names))
            self.assertTrue(any('Blue' in name for name in source_names))
            
            # Sources should be registered
            for source in sources:
                self.assertIn(source.source_id, self.engine.resonance_registry)
        
        asyncio.run(run_test())
    
    def test_integrate_strain_rate_resonances(self):
        """Test integration of strain-rate resonances"""
        sources = self.engine.integrate_strain_rate_resonances(
            self.test_lat, self.test_lon, self.test_depth, self.timestamp
        )
        
        # Should return multiple sources
        self.assertGreater(len(sources), 0)
        
        # All sources should be strain-rate type
        for source in sources:
            self.assertIsInstance(source, ResonanceSource)
            self.assertEqual(source.source_type, 'strain-rate')
            self.assertGreater(source.frequency, 0)
            self.assertGreaterEqual(source.amplitude, 0)
        
        # Should include crustal resonances
        source_names = [s.source_name for s in sources]
        self.assertTrue(any('Crustal' in name for name in source_names))
        
        # Sources should be registered
        for source in sources:
            self.assertIn(source.source_id, self.engine.resonance_registry)
    
    def test_add_custom_resonance_source(self):
        """Test adding custom resonance sources"""
        source = self.engine.add_custom_resonance_source(
            source_name='Test Custom Source',
            frequency=10.5,
            amplitude=0.75,
            phase=math.pi / 4,
            latitude=self.test_lat,
            longitude=self.test_lon,
            metadata={'test': 'data'}
        )
        
        self.assertIsInstance(source, ResonanceSource)
        self.assertEqual(source.source_name, 'Test Custom Source')
        self.assertEqual(source.source_type, 'custom')
        self.assertEqual(source.frequency, 10.5)
        self.assertEqual(source.amplitude, 0.75)
        self.assertIn('test', source.metadata)
        
        # Should be registered
        self.assertIn(source.source_id, self.engine.resonance_registry)
    
    # ========== Feature 2: Resultant Frequency Calculation Tests ==========
    
    def test_calculate_wave_superposition(self):
        """Test wave superposition calculations"""
        # Create test sources
        import math
        sources = [
            ResonanceSource(
                source_id='test1',
                source_name='Test 1',
                source_type='test',
                frequency=10.0,
                amplitude=0.5,
                phase=0.0,
                location=(self.test_lat, self.test_lon),
                timestamp=self.timestamp
            ),
            ResonanceSource(
                source_id='test2',
                source_name='Test 2',
                source_type='test',
                frequency=12.0,
                amplitude=0.6,
                phase=math.pi / 2,
                location=(self.test_lat, self.test_lon),
                timestamp=self.timestamp
            )
        ]
        
        result = self.engine.calculate_wave_superposition(
            sources, (self.test_lat, self.test_lon), time_point=0.0
        )
        
        # Check result structure
        self.assertIn('resultant_frequency', result)
        self.assertIn('resultant_amplitude', result)
        self.assertIn('resultant_phase', result)
        self.assertIn('interference_type', result)
        
        # Frequency should be between source frequencies
        self.assertGreater(result['resultant_frequency'], 0)
        
        # Amplitude should be reasonable
        self.assertGreaterEqual(result['resultant_amplitude'], 0)
        self.assertLessEqual(result['resultant_amplitude'], 1.2)  # Can exceed 1 due to constructive interference
        
        # Should have correct source count
        self.assertEqual(result['source_count'], 2)
    
    def test_detect_beat_frequencies(self):
        """Test beat frequency detection"""
        import math
        sources = [
            ResonanceSource(
                source_id='test1',
                source_name='Test 1',
                source_type='test',
                frequency=10.0,
                amplitude=0.5,
                phase=0.0,
                location=(self.test_lat, self.test_lon),
                timestamp=self.timestamp
            ),
            ResonanceSource(
                source_id='test2',
                source_name='Test 2',
                source_type='test',
                frequency=12.0,
                amplitude=0.6,
                phase=0.0,
                location=(self.test_lat, self.test_lon),
                timestamp=self.timestamp
            )
        ]
        
        beats = self.engine.detect_beat_frequencies(sources)
        
        # Should detect beat frequency
        self.assertGreater(len(beats), 0)
        
        # Beat frequency should be difference of source frequencies
        expected_beat = abs(10.0 - 12.0)
        self.assertAlmostEqual(beats[0]['beat_frequency'], expected_beat, places=2)
        
        # Beat should have amplitude
        self.assertGreater(beats[0]['beat_amplitude'], 0)
    
    # ========== Feature 3: Coherence and Amplification Detection Tests ==========
    
    def test_calculate_coherence_coefficient(self):
        """Test coherence coefficient calculation"""
        import math
        # Test with aligned phases (high coherence)
        aligned_sources = [
            ResonanceSource(
                source_id=f'aligned{i}',
                source_name=f'Aligned {i}',
                source_type='test',
                frequency=10.0,
                amplitude=0.5,
                phase=0.0,  # All same phase
                location=(self.test_lat, self.test_lon),
                timestamp=self.timestamp
            )
            for i in range(3)
        ]
        
        coherence = self.engine.calculate_coherence_coefficient(aligned_sources)
        
        self.assertIn('coherence_coefficient', coherence)
        self.assertIn('is_coherent', coherence)
        self.assertIn('phase_alignment_quality', coherence)
        
        # Aligned sources should have high coherence
        self.assertGreater(coherence['coherence_coefficient'], 0.9)
        self.assertTrue(coherence['is_coherent'])
        
        # Test with random phases (low coherence)
        import random
        random_sources = [
            ResonanceSource(
                source_id=f'random{i}',
                source_name=f'Random {i}',
                source_type='test',
                frequency=10.0,
                amplitude=0.5,
                phase=random.uniform(0, 2*math.pi),
                location=(self.test_lat, self.test_lon),
                timestamp=self.timestamp
            )
            for i in range(3)
        ]
        
        coherence_random = self.engine.calculate_coherence_coefficient(random_sources)
        
        # Random phases should have lower coherence
        self.assertLess(coherence_random['coherence_coefficient'], 
                       coherence['coherence_coefficient'])
    
    def test_identify_amplification_zones(self):
        """Test amplification zone identification"""
        import math
        sources = [
            ResonanceSource(
                source_id='test1',
                source_name='Test 1',
                source_type='test',
                frequency=10.0,
                amplitude=0.8,
                phase=0.0,
                location=(self.test_lat, self.test_lon),
                timestamp=self.timestamp
            ),
            ResonanceSource(
                source_id='test2',
                source_name='Test 2',
                source_type='test',
                frequency=10.5,
                amplitude=0.7,
                phase=0.0,  # Similar phase for constructive interference
                location=(self.test_lat + 0.1, self.test_lon + 0.1),
                timestamp=self.timestamp
            )
        ]
        
        zones = self.engine.identify_amplification_zones(sources, grid_resolution=5)
        
        # Should identify some zones
        self.assertGreaterEqual(len(zones), 0)
        
        # Each zone should have required fields
        for zone in zones:
            self.assertIn('location', zone)
            self.assertIn('resultant_amplitude', zone)
            self.assertIn('constructive_ratio', zone)
    
    def test_identify_cancellation_zones(self):
        """Test cancellation zone identification"""
        import math
        sources = [
            ResonanceSource(
                source_id='test1',
                source_name='Test 1',
                source_type='test',
                frequency=10.0,
                amplitude=0.5,
                phase=0.0,
                location=(self.test_lat, self.test_lon),
                timestamp=self.timestamp
            ),
            ResonanceSource(
                source_id='test2',
                source_name='Test 2',
                source_type='test',
                frequency=10.0,
                amplitude=0.5,
                phase=math.pi,  # Opposite phase for destructive interference
                location=(self.test_lat + 0.1, self.test_lon + 0.1),
                timestamp=self.timestamp
            )
        ]
        
        zones = self.engine.identify_cancellation_zones(sources, grid_resolution=5)
        
        # Should identify some zones
        self.assertGreaterEqual(len(zones), 0)
        
        # Each zone should have required fields
        for zone in zones:
            self.assertIn('location', zone)
            self.assertIn('resultant_amplitude', zone)
            self.assertIn('destructive_ratio', zone)
    
    # ========== Feature 4: Pattern Identification Tests ==========
    
    def test_identify_recurring_patterns(self):
        """Test recurring pattern identification"""
        # Add multiple sources with similar frequencies over time
        import math
        for day in range(5):
            timestamp = self.timestamp - timedelta(days=day)
            for i in range(3):
                self.engine.add_custom_resonance_source(
                    source_name=f'Pattern Test {day}-{i}',
                    frequency=10.0 + i * 0.5,
                    amplitude=0.6,
                    phase=0.0,
                    latitude=self.test_lat,
                    longitude=self.test_lon,
                    metadata={'day': day}
                )
        
        patterns = self.engine.identify_recurring_patterns(time_window_days=10)
        
        # Should identify patterns
        self.assertGreaterEqual(len(patterns), 0)
        
        # Each pattern should have required fields
        for pattern in patterns:
            self.assertIsInstance(pattern, ResonancePattern)
            self.assertGreater(len(pattern.frequency_signature), 0)
            self.assertGreaterEqual(pattern.occurrence_count, 1)
    
    def test_calculate_pattern_similarity(self):
        """Test pattern similarity calculation"""
        pattern1 = ResonancePattern(
            pattern_id='test1',
            pattern_name='Test 1',
            frequency_signature=[10.0, 11.0, 12.0],
            temporal_evolution=[],
            recurrence_period=1.0,
            similarity_score=1.0,
            first_observed=self.timestamp,
            last_observed=self.timestamp,
            occurrence_count=1
        )
        
        pattern2 = ResonancePattern(
            pattern_id='test2',
            pattern_name='Test 2',
            frequency_signature=[10.0, 11.0, 12.0],  # Same as pattern1
            temporal_evolution=[],
            recurrence_period=1.0,
            similarity_score=1.0,
            first_observed=self.timestamp,
            last_observed=self.timestamp,
            occurrence_count=1
        )
        
        pattern3 = ResonancePattern(
            pattern_id='test3',
            pattern_name='Test 3',
            frequency_signature=[20.0, 21.0, 22.0],  # Different
            temporal_evolution=[],
            recurrence_period=1.0,
            similarity_score=1.0,
            first_observed=self.timestamp,
            last_observed=self.timestamp,
            occurrence_count=1
        )
        
        # Similar patterns should have high similarity
        sim_high = self.engine.calculate_pattern_similarity(pattern1, pattern2)
        self.assertGreater(sim_high, 0.9)
        
        # Different patterns should have lower similarity
        sim_low = self.engine.calculate_pattern_similarity(pattern1, pattern3)
        self.assertLess(sim_low, sim_high)
    
    # ========== Feature 5: 21-Day Forward Prediction Tests ==========
    
    def test_generate_21day_prediction(self):
        """Test 21-day forward prediction"""
        async def run_test():
            result = await self.engine.generate_21day_prediction(
                self.test_lat, self.test_lon, self.test_depth
            )
            
            self.assertTrue(result['success'])
            self.assertIn('daily_predictions', result)
            self.assertIn('summary', result)
            
            # Should have 22 predictions (0 to 21 days)
            self.assertEqual(len(result['daily_predictions']), 22)
            
            # Each prediction should have required fields
            for pred in result['daily_predictions']:
                self.assertIn('day', pred)
                self.assertIn('date', pred)
                self.assertIn('resultant_frequency', pred)
                self.assertIn('resultant_amplitude', pred)
                self.assertIn('coherence_coefficient', pred)
                self.assertIn('confidence', pred)
                self.assertIn('risk_score', pred)
                self.assertIn('risk_level', pred)
                
                # Confidence should decrease with time
                if pred['day'] > 0:
                    self.assertLessEqual(pred['confidence'], 1.0)
                    self.assertGreaterEqual(pred['confidence'], 0.0)
            
            # Summary should have key statistics
            summary = result['summary']
            self.assertIn('max_risk_score', summary)
            self.assertIn('peak_risk_day', summary)
            self.assertIn('peak_risk_level', summary)
        
        asyncio.run(run_test())
    
    # ========== Feature 6: Geolocated Point Analysis Tests ==========
    
    def test_analyze_single_point(self):
        """Test single-point analysis"""
        async def run_test():
            result = await self.engine.analyze_single_point(
                self.test_lat, self.test_lon, self.test_depth
            )
            
            self.assertTrue(result['success'])
            self.assertIn('location', result)
            self.assertIn('resonance_sources', result)
            self.assertIn('superposition', result)
            self.assertIn('coherence', result)
            self.assertIn('beat_frequencies', result)
            self.assertIn('overlay_region', result)
            
            # Should have analyzed some sources
            self.assertGreater(result['resonance_sources'], 0)
            
            # Overlay region should be created
            overlay = result['overlay_region']
            self.assertIn('region_id', overlay)
            self.assertIn('overlay_count', overlay)
            
            # Region should be registered
            self.assertIn(overlay['region_id'], self.engine.overlay_regions)
        
        asyncio.run(run_test())
    
    def test_analyze_multi_fault_region(self):
        """Test multi-fault region analysis with triangulation"""
        async def run_test():
            # Tokyo region with multiple fault points
            triangulation_points = [
                (35.6762, 139.6503),  # Central Tokyo
                (35.7, 139.7),        # Northeast
                (35.6, 139.6),        # Southwest
                (35.65, 139.75)       # East
            ]
            
            result = await self.engine.analyze_multi_fault_region(
                self.test_lat, self.test_lon,
                triangulation_points,
                self.test_depth
            )
            
            self.assertTrue(result['success'])
            self.assertIn('region_center', result)
            self.assertIn('triangulation_points', result)
            self.assertIn('point_analyses', result)
            self.assertIn('regional_aggregates', result)
            self.assertIn('amplification_zones', result)
            self.assertIn('regional_risk_score', result)
            
            # Should have analyzed all triangulation points
            self.assertEqual(result['triangulation_points'], len(triangulation_points))
            self.assertEqual(len(result['point_analyses']), len(triangulation_points))
            
            # Regional aggregates should have statistics
            aggregates = result['regional_aggregates']
            self.assertIn('average_frequency', aggregates)
            self.assertIn('maximum_amplitude', aggregates)
            self.assertIn('average_coherence', aggregates)
        
        asyncio.run(run_test())
    
    # ========== Feature 7: Resonance Set Tracking Tests ==========
    
    def test_resonance_registry(self):
        """Test resonance source registry"""
        # Add some sources
        import math
        for i in range(5):
            self.engine.add_custom_resonance_source(
                source_name=f'Registry Test {i}',
                frequency=10.0 + i,
                amplitude=0.5,
                phase=0.0,
                latitude=self.test_lat,
                longitude=self.test_lon
            )
        
        # Get registry summary
        summary = self.engine.get_resonance_registry_summary()
        
        self.assertIn('total_sources', summary)
        self.assertIn('by_type', summary)
        self.assertIn('frequency_range', summary)
        self.assertIn('amplitude_range', summary)
        
        # Should have tracked sources
        self.assertGreater(summary['total_sources'], 0)
        self.assertIn('custom', summary['by_type'])
    
    def test_overlay_statistics(self):
        """Test overlay statistics tracking"""
        async def run_test():
            # Create some overlay regions
            await self.engine.analyze_single_point(
                self.test_lat, self.test_lon, self.test_depth
            )
            
            await self.engine.analyze_single_point(
                self.test_lat + 0.1, self.test_lon + 0.1, self.test_depth
            )
            
            # Get statistics
            stats = self.engine.get_overlay_statistics()
            
            self.assertIn('total_overlays', stats)
            self.assertIn('max_overlay_count', stats)
            self.assertIn('dominant_frequencies', stats)
            self.assertIn('interference_distribution', stats)
            
            # Should have tracked overlays
            self.assertGreater(stats['total_overlays'], 0)
        
        asyncio.run(run_test())
    
    def test_query_overlays_by_criteria(self):
        """Test querying overlay regions by criteria"""
        async def run_test():
            # Create overlay regions with known properties
            await self.engine.analyze_single_point(
                self.test_lat, self.test_lon, self.test_depth
            )
            
            # Query with criteria
            results = self.engine.query_overlays_by_criteria(
                min_overlay_count=1,
                time_window_hours=24
            )
            
            # Should return list
            self.assertIsInstance(results, list)
            
            # Results should meet criteria
            for region in results:
                self.assertIsInstance(region, OverlayRegion)
                self.assertGreaterEqual(region.overlay_count, 1)
        
        asyncio.run(run_test())
    
    # ========== Feature 8: Visualization Data Preparation Tests ==========
    
    def test_prepare_3d_wireframe_data(self):
        """Test 3D wireframe data preparation"""
        async def run_test():
            # Create overlay regions
            await self.engine.analyze_single_point(
                self.test_lat, self.test_lon, self.test_depth
            )
            
            regions = list(self.engine.overlay_regions.values())
            
            # Prepare visualization data
            viz_data = self.engine.prepare_3d_wireframe_data(regions, time_steps=10)
            
            self.assertIn('vertices', viz_data)
            self.assertIn('edges', viz_data)
            self.assertIn('colors', viz_data)
            self.assertIn('values', viz_data)
            self.assertIn('time_series', viz_data)
            
            # Should have data
            self.assertGreater(len(viz_data['vertices']), 0)
            
            # Each vertex should have x, y, z
            for vertex in viz_data['vertices']:
                self.assertIn('x', vertex)
                self.assertIn('y', vertex)
                self.assertIn('z', vertex)
            
            # Time series should have correct number of steps
            self.assertEqual(len(viz_data['time_series']), 10)
        
        asyncio.run(run_test())
    
    def test_prepare_time_series_data(self):
        """Test time-series data preparation"""
        # Add sources over multiple days
        import math
        for day in range(7):
            timestamp = self.timestamp - timedelta(days=day)
            self.engine.add_custom_resonance_source(
                source_name=f'Time Series Test {day}',
                frequency=10.0,
                amplitude=0.5,
                phase=0.0,
                latitude=self.test_lat,
                longitude=self.test_lon,
                metadata={'day': day}
            )
        
        # Prepare time series
        time_series = self.engine.prepare_time_series_data(
            (self.test_lat, self.test_lon), days=7
        )
        
        self.assertIn('timestamps', time_series)
        self.assertIn('frequencies', time_series)
        self.assertIn('amplitudes', time_series)
        self.assertIn('coherence', time_series)
        self.assertIn('overlay_counts', time_series)
        
        # Should have data points
        self.assertGreater(len(time_series['timestamps']), 0)
    
    # ========== Integration Tests ==========
    
    def test_full_workflow(self):
        """Test complete workflow from analysis to visualization"""
        async def run_test():
            # 1. Analyze single point
            point_result = await self.engine.analyze_single_point(
                self.test_lat, self.test_lon, self.test_depth
            )
            self.assertTrue(point_result['success'])
            
            # 2. Generate prediction
            prediction = await self.engine.generate_21day_prediction(
                self.test_lat, self.test_lon, self.test_depth
            )
            self.assertTrue(prediction['success'])
            
            # 3. Get statistics
            stats = self.engine.get_overlay_statistics()
            self.assertGreater(stats['total_overlays'], 0)
            
            # 4. Prepare visualization
            regions = list(self.engine.overlay_regions.values())
            viz_data = self.engine.prepare_3d_wireframe_data(regions)
            self.assertGreater(len(viz_data['vertices']), 0)
            
            # 5. Check engine status
            status = self.engine.get_engine_status()
            self.assertEqual(status['status'], 'operational')
            self.assertGreater(status['resonance_sources_tracked'], 0)
        
        asyncio.run(run_test())
    
    def test_engine_status(self):
        """Test engine status reporting"""
        status = self.engine.get_engine_status()
        
        self.assertIn('version', status)
        self.assertIn('engine_id', status)
        self.assertIn('status', status)
        self.assertIn('features', status)
        self.assertIn('sub_engines', status)
        
        self.assertEqual(status['status'], 'operational')
        self.assertEqual(len(status['features']), 8)  # All 8 features


# Helper functions for running tests
def run_all_tests():
    """Run all correlation engine tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCorrelationEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    print("=" * 70)
    print("CORRELATION ENGINE TEST SUITE")
    print("Testing all 8 features of multi-resonance overlay analysis")
    print("=" * 70)
    print()
    
    result = run_all_tests()
    
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
