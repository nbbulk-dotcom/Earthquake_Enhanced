"""
Unit tests for seismic moment calculator.

Tests deterministic physics calculations against known values.
"""

import pytest
import numpy as np
from backend.calculators.seismic_moment import (
    SeismicMomentCalculator,
    calculate_cumulative_moment
)


class TestSeismicMomentCalculator:
    """Test suite for seismic moment calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calc = SeismicMomentCalculator()
    
    def test_magnitude_to_moment_single(self):
        """Test magnitude to moment conversion for single value."""
        # Magnitude 6.0 should give ~1.26e18 N⋅m
        magnitude = 6.0
        moment = self.calc.magnitude_to_moment(magnitude)
        
        expected = 10 ** (1.5 * 6.0 + 9.1)  # 1.26e18
        assert abs(moment - expected) < 1e15
    
    def test_magnitude_to_moment_array(self):
        """Test magnitude to moment conversion for array."""
        magnitudes = np.array([4.0, 5.0, 6.0])
        moments = self.calc.magnitude_to_moment(magnitudes)
        
        assert len(moments) == 3
        assert moments[0] < moments[1] < moments[2]
    
    def test_moment_to_magnitude_single(self):
        """Test moment to magnitude conversion."""
        moment = 1.26e18  # Should give ~6.0
        magnitude = self.calc.moment_to_magnitude(moment)
        
        assert abs(magnitude - 6.0) < 0.01
    
    def test_roundtrip_conversion(self):
        """Test that magnitude -> moment -> magnitude is consistent."""
        original_mag = 5.5
        moment = self.calc.magnitude_to_moment(original_mag)
        recovered_mag = self.calc.moment_to_magnitude(moment)
        
        assert abs(original_mag - recovered_mag) < 1e-10
    
    def test_rupture_area_from_magnitude(self):
        """Test rupture area calculation."""
        magnitude = 6.0
        area_m2 = self.calc.rupture_area_from_magnitude(magnitude)
        
        # Magnitude 6.0 should give ~100 km² = 1e8 m²
        expected_km2 = 10 ** (6.0 - 4.0)  # 100 km²
        expected_m2 = expected_km2 * 1e6
        
        assert abs(area_m2 - expected_m2) < 1e6
    
    def test_average_slip_calculation(self):
        """Test average slip calculation."""
        moment = 1.0e18  # N⋅m
        area = 1.0e8     # m²
        
        slip = self.calc.average_slip_from_moment(moment, area)
        
        # D = M₀ / (μ × A) = 1e18 / (3e10 × 1e8) = 0.333 m
        expected = moment / (self.calc.RIGIDITY_MU * area)
        
        assert abs(slip - expected) < 1e-6
    
    def test_calculate_all_parameters(self):
        """Test comprehensive parameter calculation."""
        magnitude = 5.0
        params = self.calc.calculate_all_parameters(magnitude)
        
        assert "magnitude" in params
        assert "seismic_moment_Nm" in params
        assert "rupture_area_m2" in params
        assert "average_slip_m" in params
        assert params["magnitude"] == magnitude
        assert params["seismic_moment_Nm"] > 0
        assert params["rupture_area_m2"] > 0
        assert params["average_slip_m"] > 0
    
    def test_cumulative_moment(self):
        """Test cumulative moment calculation."""
        magnitudes = [4.0, 4.5, 5.0]
        total = calculate_cumulative_moment(magnitudes)
        
        # Total should be sum of individual moments
        individual_moments = [self.calc.magnitude_to_moment(m) for m in magnitudes]
        expected = sum(individual_moments)
        
        assert abs(total - expected) < 1e10
    
    def test_physical_constants(self):
        """Test that physical constants are correct."""
        # Rigidity should be 3.0×10¹⁰ Pa
        assert self.calc.RIGIDITY_MU == 3.0e10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
