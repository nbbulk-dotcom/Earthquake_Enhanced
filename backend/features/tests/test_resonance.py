"""
Unit tests for the Resonance Engine module.
"""

import pytest
import numpy as np
from backend.features.resonance import ResonanceEngine


class TestResonanceEngine:
    """Test cases for ResonanceEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ResonanceEngine()
    
    def test_initialization(self):
        """Test that ResonanceEngine initializes correctly."""
        assert self.engine is not None
        assert self.engine.version == "RESONANCE-ENGINE-V1.0.0"
        assert self.engine.engine_id == "RESONANCE_ENGINE_SEISMIC"
    
    def test_schumann_resonance_constants(self):
        """Test Schumann resonance frequency constants."""
        assert self.engine.SCHUMANN_BASE_FREQ == 7.83
        assert len(self.engine.SCHUMANN_HARMONICS) == 5
        assert self.engine.SCHUMANN_HARMONICS[0] == 7.83
    
    def test_earth_constants(self):
        """Test Earth physical constants."""
        assert self.engine.EARTH_RADIUS_KM == 6371.0
        assert self.engine.CRUSTAL_THICKNESS_KM == 35.0
        assert self.engine.POISSON_RATIO == 0.25
    
    def test_seismic_wave_velocities(self):
        """Test seismic wave velocity constants."""
        assert self.engine.VP_CRUST == 6.1  # P-wave
        assert self.engine.VS_CRUST == 3.5  # S-wave
        assert self.engine.VP_CRUST > self.engine.VS_CRUST  # P-waves are faster
    
    def test_elastic_moduli(self):
        """Test elastic moduli values are physically reasonable."""
        assert self.engine.ELASTIC_MODULUS_PA > 0
        assert self.engine.SHEAR_MODULUS_PA > 0
        assert self.engine.BULK_MODULUS_PA > 0
        assert self.engine.ELASTIC_MODULUS_PA > self.engine.SHEAR_MODULUS_PA
    
    def test_engine_methods_exist(self):
        """Test that expected methods exist on the engine."""
        # Check for key methods (add more as they're implemented)
        assert hasattr(self.engine, '__init__')
        
        # You can add more method checks here as the engine expands
        # For example:
        # assert hasattr(self.engine, 'calculate_strain_rate')
        # assert hasattr(self.engine, 'detect_resonance')


class TestResonanceCalculations:
    """Test cases for resonance calculation methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ResonanceEngine()
    
    def test_frequency_range(self):
        """Test that Schumann frequencies are in expected range."""
        for freq in self.engine.SCHUMANN_HARMONICS:
            assert 0 < freq < 100, f"Frequency {freq} Hz out of expected range"
    
    def test_harmonic_progression(self):
        """Test that harmonics increase monotonically."""
        harmonics = self.engine.SCHUMANN_HARMONICS
        for i in range(len(harmonics) - 1):
            assert harmonics[i] < harmonics[i+1], \
                "Harmonics should increase monotonically"


class TestPhysicalConstraints:
    """Test that physical constraints and relationships hold."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ResonanceEngine()
    
    def test_poisson_ratio_bounds(self):
        """Test Poisson's ratio is within physical bounds."""
        # For most materials: -1 < ν < 0.5
        # For crustal rocks: typically 0.20-0.30
        assert -1 < self.engine.POISSON_RATIO < 0.5
        assert 0.15 < self.engine.POISSON_RATIO < 0.35  # Reasonable for crust
    
    def test_wave_velocity_relationship(self):
        """Test P-wave and S-wave velocity relationship."""
        # For homogeneous isotropic media: VP > VS
        # Typically VP/VS ≈ 1.73 (√3) for Poisson's ratio ≈ 0.25
        ratio = self.engine.VP_CRUST / self.engine.VS_CRUST
        assert 1.4 < ratio < 2.0, f"VP/VS ratio {ratio} is outside typical range"
    
    def test_moduli_relationships(self):
        """Test relationships between elastic moduli."""
        E = self.engine.ELASTIC_MODULUS_PA
        G = self.engine.SHEAR_MODULUS_PA
        K = self.engine.BULK_MODULUS_PA
        nu = self.engine.POISSON_RATIO
        
        # E = 2G(1 + ν) should approximately hold
        E_from_G = 2 * G * (1 + nu)
        relative_error = abs(E - E_from_G) / E
        assert relative_error < 0.1, "Elastic moduli relationship E = 2G(1+ν) violated"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
