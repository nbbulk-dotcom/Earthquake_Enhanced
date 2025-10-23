
import numpy as np
from backend.features.resonance import spectral_modal_curve, find_resonant_peak, geom_resonant_freq

def test_geom_resonant_freq():
    area = 4.0  # km^2 -> L = 2 km
    vs = 4.0
    f = geom_resonant_freq(area, vs_km_s=vs, mode_factor=2.0)
    assert abs(f - 1.0) < 1e-6

def test_spectral_modal_curve_detects_peak():
    fs = 100.0
    t = np.arange(0, 20.0, 1.0 / fs)
    f0 = 2.5  # Hz injected tone
    s1 = np.sin(2 * np.pi * f0 * t)
    s2 = 0.8 * np.sin(2 * np.pi * f0 * t + 0.2)
    s3 = 0.6 * np.sin(2 * np.pi * f0 * t - 0.1)
    waves = {"S1": s1, "S2": s2, "S3": s3}
    freqs, eigvals, eigvecs = spectral_modal_curve(waves, fs=fs, nperseg=256)
    assert freqs.size > 0
    f_peak, amp_peak, idx = find_resonant_peak(freqs, eigvals)
    assert f_peak is not None
    assert abs(f_peak - f0) < 0.5
    assert amp_peak > 0.0
    assert eigvecs.shape[0] == 3
    assert idx is not None
