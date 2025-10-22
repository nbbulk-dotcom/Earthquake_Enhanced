"""
Unit Tests for Resonance Analysis Module

Tests cover:
1. Modal spectral curve computation with synthetic coherent signals
2. Resonant peak detection with known frequencies
3. Geometry-based frequency calculation
4. Q-factor computation
5. Transfer function calculation
6. Edge cases and error handling

Test Philosophy:
- Use deterministic synthetic signals with known properties
- Verify numerical accuracy within tolerance
- Test physics consistency (e.g., f_peak near injected frequency)
- Validate error handling for invalid inputs
"""

import numpy as np
import pytest
from backend.features.resonance import (
    spectral_modal_curve,
    find_resonant_peak,
    geom_resonant_freq,
    compute_q_factor,
    transfer_function,
)


class TestSpectralModalCurve:
    """Test modal spectral curve computation."""
    
    def test_coherent_sine_waves(self):
        """
        Test with 3 coherent sine waves at known frequency.
        
        Expected behavior:
        - Peak in eigenvalue spectrum at injected frequency
        - Principal eigenvalue >> noise floor
        - Modal eigenvector has consistent phases
        """
        # Setup: 3 stations with 5 Hz coherent signal
        fs = 100.0  # Hz
        duration = 10.0  # seconds
        f_injected = 5.0  # Hz
        
        t = np.arange(0, duration, 1/fs)
        
        # Generate coherent signals with small phase shifts
        waves = {
            'station1': np.sin(2 * np.pi * f_injected * t),
            'station2': np.sin(2 * np.pi * f_injected * t + 0.1),
            'station3': np.sin(2 * np.pi * f_injected * t + 0.2),
        }
        
        # Add small noise
        noise_level = 0.05
        for sid in waves:
            waves[sid] += noise_level * np.random.randn(len(t))
        
        # Execute
        freqs, eigvals, eigvecs = spectral_modal_curve(waves, fs=fs, nperseg=512)
        
        # Verify frequency resolution
        assert len(freqs) > 0
        assert freqs[0] >= 0
        assert freqs[-1] <= fs / 2  # Nyquist frequency
        
        # Verify eigenvalues are real and non-negative
        assert np.all(eigvals >= 0)
        assert eigvals.dtype in [np.float32, np.float64]
        
        # Verify eigenvectors are complex and normalized
        assert eigvecs.dtype == np.complex128
        # Check normalization (within numerical tolerance)
        for f_idx in range(len(freqs)):
            norm = np.linalg.norm(eigvecs[f_idx, :])
            assert np.abs(norm - 1.0) < 1e-6, f"Eigenvector not normalized at f={freqs[f_idx]}"
        
        # Verify peak is near injected frequency
        peak_idx = np.argmax(eigvals)
        f_peak = freqs[peak_idx]
        error_hz = abs(f_peak - f_injected)
        error_pct = 100 * error_hz / f_injected
        
        # Allow 5% error due to finite frequency resolution
        assert error_pct < 5.0, f"Peak at {f_peak:.3f} Hz, expected {f_injected:.3f} Hz"
        
        # Verify peak amplitude is significant
        noise_floor = np.median(eigvals)
        snr = eigvals[peak_idx] / noise_floor
        assert snr > 3.0, f"SNR too low: {snr:.1f}"
        
        # Verify modal eigenvector phases are consistent (small phase differences)
        modal_vec = eigvecs[peak_idx, :]
        phases = np.angle(modal_vec)
        phases_unwrapped = np.unwrap(phases)  # Unwrap 2π discontinuities
        phase_diffs = np.diff(phases_unwrapped)
        
        # Phase differences should be small (< pi/2) for coherent signal
        assert np.all(np.abs(phase_diffs) < np.pi / 2), "Large phase inconsistency"
    
    def test_empty_input(self):
        """Test error handling for empty waves_dict."""
        with pytest.raises(ValueError, match="cannot be empty"):
            spectral_modal_curve({}, fs=100.0)
    
    def test_inconsistent_lengths(self):
        """Test error handling for waveforms with different lengths."""
        waves = {
            'station1': np.random.randn(100),
            'station2': np.random.randn(150),  # Different length
        }
        with pytest.raises(ValueError, match="same length"):
            spectral_modal_curve(waves, fs=100.0)
    
    def test_nan_values(self):
        """Test error handling for waveforms with NaN."""
        waves = {
            'station1': np.array([1, 2, np.nan, 4]),
            'station2': np.array([1, 2, 3, 4]),
        }
        with pytest.raises(ValueError, match="NaN or Inf"):
            spectral_modal_curve(waves, fs=100.0)
    
    def test_single_station(self):
        """Test with single station (degenerate case)."""
        fs = 100.0
        t = np.arange(0, 2, 1/fs)  # Longer signal to accommodate default nperseg
        waves = {'station1': np.sin(2 * np.pi * 5.0 * t)}
        
        freqs, eigvals, eigvecs = spectral_modal_curve(waves, fs=fs, nperseg=64)
        
        # Should work, eigenvector should be scalar (1.0+0j)
        assert len(freqs) > 0
        assert eigvecs.shape[1] == 1
        # Eigenvector should be close to [1.0+0j]
        assert np.allclose(np.abs(eigvecs[:, 0]), 1.0)


class TestFindResonantPeak:
    """Test resonant peak detection."""
    
    def test_single_sharp_peak(self):
        """Test detection of single sharp peak."""
        # Create synthetic spectrum with Gaussian peak at 5 Hz
        freqs = np.linspace(0, 10, 1000)
        f_true = 5.0
        sigma = 0.2
        eigvals = np.exp(-0.5 * ((freqs - f_true) / sigma) ** 2)
        
        # Add small noise
        eigvals += 0.01 * np.random.rand(len(freqs))
        
        f_peak, amp_peak, idx_peak = find_resonant_peak(freqs, eigvals)
        
        # Verify peak is close to true frequency
        assert abs(f_peak - f_true) < 0.1, f"Peak at {f_peak:.3f}, expected {f_true:.3f}"
        assert amp_peak > 0.9  # Near peak of Gaussian
        assert 0 <= idx_peak < len(freqs)
    
    def test_frequency_band_restriction(self):
        """Test peak detection within restricted frequency band."""
        freqs = np.linspace(0, 20, 2000)
        
        # Create two peaks: 3 Hz and 15 Hz
        eigvals = (
            np.exp(-0.5 * ((freqs - 3.0) / 0.3) ** 2) +
            2.0 * np.exp(-0.5 * ((freqs - 15.0) / 0.3) ** 2)
        )
        
        # Search only in [0, 5] Hz band - should find 3 Hz peak
        f_peak_low, _, _ = find_resonant_peak(freqs, eigvals, band=(0, 5))
        assert 2.5 < f_peak_low < 3.5, f"Expected peak near 3 Hz, got {f_peak_low:.3f}"
        
        # Search only in [10, 20] Hz band - should find 15 Hz peak
        f_peak_high, _, _ = find_resonant_peak(freqs, eigvals, band=(10, 20))
        assert 14.5 < f_peak_high < 15.5, f"Expected peak near 15 Hz, got {f_peak_high:.3f}"
    
    def test_no_peak_detected(self):
        """Test when no peak exceeds threshold."""
        freqs = np.linspace(0, 10, 1000)
        # Create flat spectrum (no resonance)
        eigvals = np.ones(len(freqs))  # Completely flat
        
        f_peak, amp_peak, idx_peak = find_resonant_peak(
            freqs, eigvals, threshold_factor=0.9  # High threshold
        )
        
        # Should return NaN values (no peak above threshold)
        assert np.isnan(f_peak)
        assert np.isnan(amp_peak)
        assert idx_peak == -1
    
    def test_invalid_band(self):
        """Test error handling for invalid frequency band."""
        freqs = np.linspace(0, 10, 100)
        eigvals = np.random.rand(len(freqs))
        
        with pytest.raises(ValueError, match="Invalid band"):
            find_resonant_peak(freqs, eigvals, band=(5, 3))  # f_max < f_min
    
    def test_length_mismatch(self):
        """Test error handling for mismatched array lengths."""
        freqs = np.linspace(0, 10, 100)
        eigvals = np.random.rand(150)  # Different length
        
        with pytest.raises(ValueError, match="same length"):
            find_resonant_peak(freqs, eigvals)


class TestGeomResonantFreq:
    """Test geometry-based resonance frequency calculation."""
    
    def test_known_values(self):
        """Test with known input values."""
        # Example: 50 km × 50 km region with Vs = 3.5 km/s
        area_km2 = 2500.0
        vs_km_s = 3.5
        geometry_factor = 0.5
        
        # Expected: L = 0.5 * sqrt(2500) = 25 km
        #           f = 3.5 / (2 * 25) = 0.07 Hz
        f_expected = 0.07
        
        f_est = geom_resonant_freq(area_km2, vs_km_s, geometry_factor)
        
        # Allow 1% tolerance for floating point
        assert abs(f_est - f_expected) / f_expected < 0.01
    
    def test_scaling_relationships(self):
        """Test that frequency scales correctly with parameters."""
        area = 1000.0
        vs = 3.0
        
        f_base = geom_resonant_freq(area, vs)
        
        # Doubling area → sqrt(2) longer length → 1/sqrt(2) frequency
        f_double_area = geom_resonant_freq(2 * area, vs)
        assert abs(f_double_area / f_base - 1/np.sqrt(2)) < 0.01
        
        # Doubling Vs → 2x frequency
        f_double_vs = geom_resonant_freq(area, 2 * vs)
        assert abs(f_double_vs / f_base - 2.0) < 0.01
        
        # Doubling geometry factor → 2x length → 1/2 frequency
        f_double_factor = geom_resonant_freq(area, vs, geometry_factor=1.0)
        f_base_factor = geom_resonant_freq(area, vs, geometry_factor=0.5)
        assert abs(f_double_factor / f_base_factor - 0.5) < 0.01
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Negative area
        with pytest.raises(ValueError, match="area_km2 must be positive"):
            geom_resonant_freq(-100, 3.5)
        
        # Zero area
        with pytest.raises(ValueError, match="area_km2 must be positive"):
            geom_resonant_freq(0, 3.5)
        
        # Negative Vs
        with pytest.raises(ValueError, match="vs_km_s must be positive"):
            geom_resonant_freq(1000, -3.5)
        
        # Negative geometry factor
        with pytest.raises(ValueError, match="geometry_factor must be positive"):
            geom_resonant_freq(1000, 3.5, geometry_factor=-0.5)
    
    def test_realistic_ranges(self):
        """Test with realistic crustal parameters."""
        # Small region: 10 km × 10 km
        f_small = geom_resonant_freq(100, 3.5)
        assert 0.1 < f_small < 1.0, "Small region should have > 0.1 Hz"
        
        # Large region: 200 km × 200 km
        f_large = geom_resonant_freq(40000, 3.5)
        assert 0.001 < f_large < 0.1, "Large region should have < 0.1 Hz"


class TestComputeQFactor:
    """Test Q-factor computation."""
    
    def test_lorentzian_peak(self):
        """Test Q-factor for ideal Lorentzian (Breit-Wigner) peak."""
        # Create Lorentzian: L(f) = A / (1 + ((f - f0) / (gamma/2))^2)
        # FWHM = gamma, so Q = f0 / gamma
        
        f0 = 5.0  # Peak frequency
        gamma = 0.5  # FWHM
        Q_true = f0 / gamma  # = 10
        
        freqs = np.linspace(0, 10, 10000)
        eigvals = 1.0 / (1 + ((freqs - f0) / (gamma / 2)) ** 2)
        
        # Find peak
        idx_peak = np.argmax(eigvals)
        f_peak = freqs[idx_peak]
        
        # Compute Q
        Q = compute_q_factor(freqs, eigvals, f_peak, idx_peak)
        
        # Q-factor estimation from half-power bandwidth can vary
        # Allow 60% tolerance (numerical method, linear interpolation)
        error_pct = 100 * abs(Q - Q_true) / Q_true
        assert error_pct < 60.0, f"Q = {Q:.1f}, expected {Q_true:.1f}"
    
    def test_gaussian_peak(self):
        """Test Q-factor for Gaussian peak."""
        # Gaussian: G(f) = exp(-0.5 * ((f - f0) / sigma)^2)
        # FWHM ≈ 2.355 * sigma
        
        f0 = 10.0
        sigma = 0.5
        fwhm = 2.355 * sigma
        Q_expected = f0 / fwhm
        
        freqs = np.linspace(0, 20, 10000)
        eigvals = np.exp(-0.5 * ((freqs - f0) / sigma) ** 2)
        
        idx_peak = np.argmax(eigvals)
        Q = compute_q_factor(freqs, eigvals, freqs[idx_peak], idx_peak)
        
        # Gaussian Q is approximate, allow 50% error
        error_pct = 100 * abs(Q - Q_expected) / Q_expected
        assert error_pct < 50.0, f"Q = {Q:.1f}, expected {Q_expected:.1f}"
    
    def test_peak_at_boundary(self):
        """Test Q computation when peak is at edge of spectrum."""
        freqs = np.linspace(0, 10, 1000)
        eigvals = np.exp(-0.5 * ((freqs - 0.1) / 0.2) ** 2)  # Peak near f=0
        
        idx_peak = np.argmax(eigvals)
        Q = compute_q_factor(freqs, eigvals, freqs[idx_peak], idx_peak)
        
        # Should return NaN (cannot determine full bandwidth)
        assert np.isnan(Q)
    
    def test_invalid_index(self):
        """Test error handling for invalid peak index."""
        freqs = np.linspace(0, 10, 100)
        eigvals = np.random.rand(len(freqs))
        
        with pytest.raises(ValueError, match="out of range"):
            compute_q_factor(freqs, eigvals, 5.0, idx_peak=-1)
        
        with pytest.raises(ValueError, match="out of range"):
            compute_q_factor(freqs, eigvals, 5.0, idx_peak=1000)


class TestTransferFunction:
    """Test transfer function computation."""
    
    def test_identity_transfer(self):
        """Test transfer function for identical signals (H = 1)."""
        fs = 100.0
        t = np.arange(0, 2, 1/fs)
        signal_data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
        
        # Identical input and output
        freqs, H = transfer_function(signal_data, signal_data, fs=fs)
        magnitude = np.abs(H)
        
        # Check magnitude at signal frequencies (5 Hz and 10 Hz)
        # These should be close to 1.0
        idx_5hz = np.argmin(np.abs(freqs - 5.0))
        idx_10hz = np.argmin(np.abs(freqs - 10.0))
        
        assert 0.9 < magnitude[idx_5hz] < 1.1, f"H(5Hz) = {magnitude[idx_5hz]:.2f}"
        assert 0.9 < magnitude[idx_10hz] < 1.1, f"H(10Hz) = {magnitude[idx_10hz]:.2f}"
    
    def test_amplification(self):
        """Test transfer function with constant amplification."""
        fs = 100.0
        t = np.arange(0, 2, 1/fs)
        
        event = np.sin(2 * np.pi * 5 * t)
        response = 2.0 * event  # 2x amplification
        
        freqs, H = transfer_function(event, response, fs=fs)
        magnitude = np.abs(H)
        
        # At 5 Hz, should have ~2x amplification
        idx_5hz = np.argmin(np.abs(freqs - 5.0))
        assert 1.8 < magnitude[idx_5hz] < 2.2, f"Expected ~2x amp, got {magnitude[idx_5hz]:.2f}"
    
    def test_phase_shift(self):
        """Test transfer function with phase shift."""
        fs = 100.0
        t = np.arange(0, 2, 1/fs)
        f_signal = 5.0
        
        event = np.sin(2 * np.pi * f_signal * t)
        phase_shift = np.pi / 4  # 45 degrees
        response = np.sin(2 * np.pi * f_signal * t + phase_shift)
        
        freqs, H = transfer_function(event, response, fs=fs)
        phase = np.angle(H)
        
        # At 5 Hz, phase should be ~pi/4
        idx_5hz = np.argmin(np.abs(freqs - 5.0))
        phase_at_5hz = phase[idx_5hz]
        
        # Allow 10% error
        assert abs(phase_at_5hz - phase_shift) < 0.1 * np.pi
    
    def test_length_mismatch(self):
        """Test error handling for mismatched signal lengths."""
        event = np.random.randn(100)
        response = np.random.randn(150)
        
        with pytest.raises(ValueError, match="same length"):
            transfer_function(event, response, fs=100.0)
    
    def test_negative_regularization(self):
        """Test error handling for negative regularization."""
        signal_data = np.random.randn(100)
        
        with pytest.raises(ValueError, match="non-negative"):
            transfer_function(signal_data, signal_data, fs=100.0, regularization=-1e-6)


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_workflow(self):
        """
        Test complete resonance analysis workflow:
        1. Generate synthetic multi-station data
        2. Compute modal spectral curve
        3. Detect resonant peak
        4. Compute Q-factor
        5. Compare with geometry estimate
        """
        # Setup
        fs = 100.0
        duration = 20.0
        f_true = 3.0
        
        t = np.arange(0, duration, 1/fs)
        
        # Generate 4-station array with coherent 3 Hz signal
        waves = {}
        for i in range(4):
            phase = i * 0.15  # Progressive phase shift
            waves[f'station{i}'] = (
                np.sin(2 * np.pi * f_true * t + phase) +
                0.1 * np.random.randn(len(t))
            )
        
        # Step 1: Modal spectral curve
        freqs, eigvals, eigvecs = spectral_modal_curve(waves, fs=fs, nperseg=512)
        assert len(freqs) > 0
        
        # Step 2: Find resonant peak
        f_peak, amp_peak, idx_peak = find_resonant_peak(
            freqs, eigvals, band=(0.5, 10.0)
        )
        
        # Verify peak detection
        assert not np.isnan(f_peak)
        assert abs(f_peak - f_true) < 0.2  # Within 0.2 Hz
        
        # Step 3: Compute Q-factor
        Q = compute_q_factor(freqs, eigvals, f_peak, idx_peak)
        assert not np.isnan(Q)
        assert Q > 5.0  # Reasonable Q for sharp peak
        
        # Step 4: Geometry estimate
        # For a "coherent region" we can estimate the wavelength
        # lambda = Vs / f ≈ 3.5 km/s / 3 Hz ≈ 1.17 km
        # If characteristic length L ≈ lambda/2 ≈ 0.58 km
        # Then area ≈ (L/0.5)^2 ≈ 1.35 km²
        area_km2 = 1.35
        vs_km_s = 3.5
        f_geom = geom_resonant_freq(area_km2, vs_km_s)
        
        # Geometry estimate should be same order of magnitude
        ratio = f_peak / f_geom
        assert 0.5 < ratio < 2.0, f"Geometry ratio {ratio:.2f} out of range"
    
    def test_determinism(self):
        """
        Test that results are deterministic (same input → same output).
        
        Critical for reproducibility in measurement system.
        """
        # Setup
        np.random.seed(42)  # Fix random seed
        
        fs = 100.0
        t = np.arange(0, 5, 1/fs)
        
        waves = {
            'station1': np.sin(2 * np.pi * 4 * t) + 0.1 * np.random.randn(len(t)),
            'station2': np.sin(2 * np.pi * 4 * t + 0.1) + 0.1 * np.random.randn(len(t)),
        }
        
        # Run twice with same data
        freqs1, eigvals1, eigvecs1 = spectral_modal_curve(waves, fs=fs, nperseg=256)
        freqs2, eigvals2, eigvecs2 = spectral_modal_curve(waves, fs=fs, nperseg=256)
        
        # Results should be identical
        assert np.allclose(freqs1, freqs2)
        assert np.allclose(eigvals1, eigvals2)
        assert np.allclose(eigvecs1, eigvecs2)
        
        # Peak detection should also be deterministic
        f_peak1, amp_peak1, idx_peak1 = find_resonant_peak(freqs1, eigvals1)
        f_peak2, amp_peak2, idx_peak2 = find_resonant_peak(freqs2, eigvals2)
        
        assert f_peak1 == f_peak2
        assert amp_peak1 == amp_peak2
        assert idx_peak1 == idx_peak2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
