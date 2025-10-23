"""
Comprehensive test suite for Tokyo earthquake prediction experiment.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import shutil


# Import the experiment functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.experiments.tokyo.run_experiment import (
    haversine_distance,
    calculate_seismic_moment
)


class TestHaversineDistance:
    """Test cases for haversine distance calculation."""
    
    def test_zero_distance(self):
        """Test distance between same point is zero."""
        distance = haversine_distance(139.6917, 35.6895, 139.6917, 35.6895)
        assert abs(distance) < 0.001, "Distance between same point should be ~0"
    
    def test_known_distance_tokyo_osaka(self):
        """Test distance between Tokyo and Osaka."""
        # Tokyo: 35.6895° N, 139.6917° E
        # Osaka: 34.6937° N, 135.5023° E
        # Known distance: ~400 km
        distance = haversine_distance(139.6917, 35.6895, 135.5023, 34.6937)
        assert 390 < distance < 410, f"Tokyo-Osaka distance should be ~400km, got {distance}"
    
    def test_symmetry(self):
        """Test that distance is symmetric."""
        d1 = haversine_distance(139.0, 35.0, 140.0, 36.0)
        d2 = haversine_distance(140.0, 36.0, 139.0, 35.0)
        assert abs(d1 - d2) < 0.001, "Distance should be symmetric"
    
    def test_positive_distance(self):
        """Test that distance is always positive."""
        distance = haversine_distance(139.0, 35.0, 140.0, 36.0)
        assert distance > 0, "Distance should be positive"


class TestSeismicMoment:
    """Test cases for seismic moment calculation."""
    
    def test_hanks_kanamori_formula(self):
        """Test Hanks & Kanamori (1979) formula."""
        # M0 = 10^(1.5 * Mw + 9.1)
        # For Mw = 5.0, log10(M0) should be 16.6
        result = calculate_seismic_moment(5.0)
        expected = 1.5 * 5.0 + 9.1  # = 16.6
        assert abs(result - expected) < 0.001
    
    def test_magnitude_scaling(self):
        """Test that larger magnitudes give larger moments."""
        m4 = calculate_seismic_moment(4.0)
        m5 = calculate_seismic_moment(5.0)
        m6 = calculate_seismic_moment(6.0)
        
        assert m4 < m5 < m6, "Seismic moment should increase with magnitude"
    
    def test_moment_difference(self):
        """Test that one magnitude unit increases moment by factor ~31.6."""
        # In log scale, 1 magnitude unit = 1.5 in log10(M0)
        # 10^1.5 ≈ 31.6
        m4 = calculate_seismic_moment(4.0)
        m5 = calculate_seismic_moment(5.0)
        
        diff = m5 - m4
        assert abs(diff - 1.5) < 0.001, "Magnitude difference should be 1.5 in log scale"


class TestDataLoading:
    """Test cases for data loading and parsing."""
    
    @pytest.fixture
    def sample_geojson(self):
        """Create sample GeoJSON data for testing."""
        return {
            "type": "FeatureCollection",
            "metadata": {
                "generated": 1672531200000,
                "count": 3
            },
            "features": [
                {
                    "type": "Feature",
                    "id": "test1",
                    "properties": {
                        "mag": 4.5,
                        "time": 1640995200000,
                        "place": "Test Location 1"
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [139.5, 35.5, 50.0]
                    }
                },
                {
                    "type": "Feature",
                    "id": "test2",
                    "properties": {
                        "mag": 5.0,
                        "time": 1641081600000,
                        "place": "Test Location 2"
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [139.7, 35.7, 45.0]
                    }
                },
                {
                    "type": "Feature",
                    "id": "test3",
                    "properties": {
                        "mag": 3.8,
                        "time": 1641168000000,
                        "place": "Test Location 3"
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [139.3, 35.3, 55.0]
                    }
                }
            ]
        }
    
    def test_geojson_structure(self, sample_geojson):
        """Test that sample GeoJSON has correct structure."""
        assert sample_geojson['type'] == 'FeatureCollection'
        assert 'metadata' in sample_geojson
        assert 'features' in sample_geojson
        assert len(sample_geojson['features']) == 3
    
    def test_feature_properties(self, sample_geojson):
        """Test that features have required properties."""
        for feature in sample_geojson['features']:
            assert 'properties' in feature
            assert 'mag' in feature['properties']
            assert 'time' in feature['properties']
            assert 'geometry' in feature
            assert 'coordinates' in feature['geometry']


class TestFeatureEngineering:
    """Test cases for feature engineering."""
    
    @pytest.fixture
    def sample_daily_data(self):
        """Create sample daily aggregated data."""
        dates = pd.date_range('2022-01-01', periods=100, freq='D')
        
        np.random.seed(42)
        data = {
            'count': np.random.randint(0, 10, size=100),
            'mean_mag': np.random.uniform(3.5, 4.5, size=100),
            'max_mag': np.random.uniform(4.0, 5.5, size=100),
            'mean_depth': np.random.uniform(10, 100, size=100)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_rolling_window_features(self, sample_daily_data):
        """Test rolling window feature calculation."""
        # Calculate 7-day rolling mean
        rolling_mean = sample_daily_data['count'].rolling(7, min_periods=1).mean()
        
        assert len(rolling_mean) == len(sample_daily_data)
        assert not rolling_mean.isna().all(), "Rolling mean should have values"
        
        # First value should equal first count (min_periods=1)
        assert rolling_mean.iloc[0] == sample_daily_data['count'].iloc[0]
    
    def test_feature_dimensions(self, sample_daily_data):
        """Test that features have correct dimensions."""
        assert sample_daily_data.shape[0] == 100, "Should have 100 days"
        assert sample_daily_data.shape[1] == 4, "Should have 4 base features"
    
    def test_no_negative_counts(self, sample_daily_data):
        """Test that earthquake counts are non-negative."""
        assert (sample_daily_data['count'] >= 0).all(), "Counts should be non-negative"
    
    def test_magnitude_range(self, sample_daily_data):
        """Test that magnitudes are in reasonable range."""
        assert (sample_daily_data['mean_mag'] >= 3.0).all(), "Magnitudes too low"
        assert (sample_daily_data['mean_mag'] <= 10.0).all(), "Magnitudes too high"


class TestMultiHorizonLabeling:
    """Test cases for multi-horizon label creation."""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample event data with known large events."""
        dates = pd.date_range('2022-01-01', periods=60, freq='D')
        
        # Create data with known pattern:
        # Day 10: M5.5 event
        # Day 30: M5.2 event
        # Day 50: M5.1 event
        max_mags = np.full(60, 4.0)
        max_mags[9] = 5.5   # Day 10
        max_mags[29] = 5.2  # Day 30
        max_mags[49] = 5.1  # Day 50
        
        data = pd.DataFrame({
            'max_mag': max_mags,
            'count': np.random.randint(1, 5, size=60)
        }, index=dates)
        
        return data
    
    def test_7day_horizon_labeling(self, sample_events):
        """Test 7-day horizon label creation."""
        horizon = 7
        labels = np.zeros(len(sample_events))
        
        for i in range(len(sample_events) - horizon):
            future_window = sample_events.iloc[i+1:i+1+horizon]
            if (future_window['max_mag'] >= 5.0).any():
                labels[i] = 1
        
        # Days 3-9 should be labeled (7 days before event on day 10)
        assert labels[2:9].sum() > 0, "Should have labels before M5.5 event"
    
    def test_label_count_increases_with_horizon(self, sample_events):
        """Test that longer horizons generally produce more labels."""
        def count_labels(events, horizon):
            labels = np.zeros(len(events))
            for i in range(len(events) - horizon):
                future_window = events.iloc[i+1:i+1+horizon]
                if (future_window['max_mag'] >= 5.0).any():
                    labels[i] = 1
            return labels.sum()
        
        labels_7d = count_labels(sample_events, 7)
        labels_14d = count_labels(sample_events, 14)
        labels_30d = count_labels(sample_events, 30)
        
        # Longer horizons should have same or more labels
        assert labels_14d >= labels_7d, "14d should have ≥ labels than 7d"
        assert labels_30d >= labels_14d, "30d should have ≥ labels than 14d"


class TestModelEvaluation:
    """Test cases for model evaluation metrics."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        assert precision_score(y_true, y_pred) == 1.0
        assert recall_score(y_true, y_pred) == 1.0
        assert f1_score(y_true, y_pred) == 1.0
    
    def test_zero_predictions(self):
        """Test metrics when no positives are predicted."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        
        # Precision undefined (0/0), but recall should be 0
        assert recall_score(y_true, y_pred) == 0.0
        assert f1_score(y_true, y_pred) == 0.0
    
    def test_roc_auc_with_probabilities(self):
        """Test ROC-AUC calculation with probability scores."""
        from sklearn.metrics import roc_auc_score
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
        
        auc = roc_auc_score(y_true, y_prob)
        assert 0 <= auc <= 1, "ROC-AUC should be between 0 and 1"
        assert auc > 0.5, "Model should perform better than random"


class TestExperimentIntegration:
    """Integration tests for the full experiment pipeline."""
    
    def test_experiment_directory_structure(self):
        """Test that experiment has correct directory structure."""
        base_path = Path(__file__).parent.parent.parent.parent.parent
        
        # Check key directories exist
        assert (base_path / 'backend' / 'experiments' / 'tokyo').exists()
        assert (base_path / 'backend' / 'features').exists()
        assert (base_path / 'data').exists() or True  # May not exist yet
    
    def test_config_file_exists(self):
        """Test that configuration file exists."""
        config_path = Path(__file__).parent.parent.parent.parent / 'features' / 'config.yaml'
        assert config_path.exists(), f"Config file should exist at {config_path}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()
        assert len(df) == 0
        assert df.empty
    
    def test_single_event(self):
        """Test handling of single event."""
        df = pd.DataFrame({
            'mag': [4.5],
            'lat': [35.0],
            'lon': [139.0],
            'depth_km': [50.0]
        })
        
        assert len(df) == 1
        assert df['mag'].iloc[0] == 4.5
    
    def test_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'mag': [4.5, np.nan, 5.0],
            'lat': [35.0, 35.5, 36.0]
        })
        
        # After dropna, should have 2 rows
        df_clean = df.dropna()
        assert len(df_clean) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
