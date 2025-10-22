"""
Resonance Analysis Module for Earthquake Measurement System

This module implements frequency-domain modal spectral analysis to detect
resonant frequencies in seismic networks. The approach uses cross-spectral
density matrices and eigendecomposition to identify coherent oscillations
across multiple stations.

Physics Background:
-------------------
Seismic resonance occurs when crustal structures respond to stress at natural
frequencies determined by geometry and material properties. Modal spectral
analysis extracts the dominant frequency components by computing the principal
eigenvalue of the cross-spectral density matrix.

References:
-----------
1. Bendat & Piersol (2010): "Random Data: Analysis and Measurement Procedures"
   - Cross-spectral density and modal analysis (Chapter 9)

2. Lacoss et al. (1969): "Estimation of Seismic Noise Structure using Arrays"
   - Eigenvalue-based spectral analysis for seismic arrays

3. Capon (1969): "High-resolution frequency-wavenumber spectrum analysis"
   - Maximum likelihood estimation of spectral peaks

4. Aki & Richards (2002): "Quantitative Seismology" (2nd ed.)
   - Crustal resonance and Q-factor estimation (Chapter 5)

5. Kanasewich (1981): "Time Sequence Analysis in Geophysics"
   - Cross-spectral methods for multi-station analysis (Chapter 4)

Key Concepts:
-------------
- Cross-Spectral Density Matrix P(f): Captures phase relationships between stations
- Principal Eigenvalue λmax(f): Measures coherent power at frequency f
- Modal Eigenvector v(f): Defines spatial pattern of coherent oscillation
- Resonant Frequency f_peak: argmax_f |λmax(f)|
- Quality Factor Q: Sharpness of resonance (f_peak / Δf_half_power)
- Geometry Estimate: f_geom ≈ Vs / (2L) for characteristic length L

Author: Earthquake Enhanced Team
License: MIT
"""

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from typing import Dict, List, Tuple, Optional, Union
import warnings


def spectral_modal_curve(
    waves_dict: Dict[str, np.ndarray],
    fs: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    detrend: str = 'constant',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute modal spectral curve from multi-station seismic waveforms.
    
    This function calculates the cross-spectral density (CSD) matrix P(f) for
    all station pairs and performs eigendecomposition to extract the principal
    eigenvalue λmax(f) and eigenvector v(f) at each frequency.
    
    The principal eigenvalue represents the maximum coherent power across the
    network at frequency f, while the eigenvector describes the spatial pattern
    of the coherent oscillation.
    
    Mathematical Formulation:
    -------------------------
    For N stations with signals x_i(t), i=1...N:
    
    1. Compute cross-spectral density:
       P_ij(f) = <X_i(f) X_j*(f)>  [Welch's method]
    
    2. Eigendecomposition:
       P(f) v_k(f) = λ_k(f) v_k(f),  k=1...N
    
    3. Principal eigenvalue:
       λmax(f) = max_k λ_k(f)
    
    4. Modal eigenvector:
       vmax(f) = v_k(f) where λ_k(f) = λmax(f)
    
    Physics Interpretation:
    -----------------------
    - λmax(f) >> other λ_k(f): Strong coherent signal at frequency f
    - Uniform phase in vmax(f): Plane wave or resonance mode
    - Peak in λmax(f): Resonant frequency of crustal structure
    
    Args:
        waves_dict: Dictionary mapping station_id -> waveform array
                   All waveforms must have the same length and sampling rate
        fs: Sampling frequency in Hz
        nperseg: Length of each segment for Welch's method (default: 256)
                Higher values → better frequency resolution, more smoothing
        noverlap: Number of points to overlap between segments
                 Default: nperseg // 2
        window: Window function for FFT ('hann', 'hamming', 'blackman', etc.)
        detrend: Detrending method ('constant', 'linear', False)
    
    Returns:
        freqs: Frequency array (Hz) of length M
        eigvals: Principal eigenvalue at each frequency, shape (M,)
                 Units: (signal amplitude)^2
        eigvecs: Modal eigenvector at each frequency, shape (M, N)
                 Rows correspond to frequencies, columns to stations
                 Complex-valued: magnitude = amplitude, phase = relative timing
    
    Raises:
        ValueError: If waves_dict is empty or waveforms have inconsistent lengths
        ValueError: If any waveform contains NaN or Inf values
    
    Example:
        >>> waves = {
        ...     'station1': np.sin(2*np.pi*5*t),  # 5 Hz sine
        ...     'station2': np.sin(2*np.pi*5*t + 0.1),  # Phase-shifted
        ...     'station3': np.sin(2*np.pi*5*t + 0.2),
        ... }
        >>> freqs, eigvals, eigvecs = spectral_modal_curve(waves, fs=100)
        >>> peak_idx = np.argmax(eigvals)
        >>> f_peak = freqs[peak_idx]
        >>> print(f"Resonant frequency: {f_peak:.2f} Hz")
    
    Notes:
        - All waveforms are normalized to unit variance before analysis
        - CSD matrix is Hermitian: P_ij* = P_ji
        - Eigenvalues are real and non-negative
        - Eigenvectors are complex-valued with unit norm
        - DC component (f=0) is typically removed by detrending
    
    References:
        Bendat & Piersol (2010), Chapter 9
        Lacoss et al. (1969), IEEE Trans. Geosci. Electron.
    """
    # Input validation
    if not waves_dict:
        raise ValueError("waves_dict cannot be empty")
    
    station_ids = list(waves_dict.keys())
    waveforms = [waves_dict[sid] for sid in station_ids]
    
    # Check waveform lengths
    lengths = [len(w) for w in waveforms]
    if len(set(lengths)) > 1:
        raise ValueError(f"All waveforms must have same length. Got: {lengths}")
    
    # Check for invalid values
    for sid, waveform in zip(station_ids, waveforms):
        if not np.all(np.isfinite(waveform)):
            raise ValueError(f"Waveform for station {sid} contains NaN or Inf")
    
    N = len(station_ids)
    
    # Normalize waveforms to unit variance (z-score normalization)
    normalized_waveforms = []
    for waveform in waveforms:
        std = np.std(waveform)
        if std > 0:
            norm_waveform = (waveform - np.mean(waveform)) / std
        else:
            warnings.warn("Waveform has zero variance, using raw values")
            norm_waveform = waveform - np.mean(waveform)
        normalized_waveforms.append(norm_waveform)
    
    # Set default overlap
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Compute cross-spectral density for first station to get frequency array
    freqs, _ = signal.csd(
        normalized_waveforms[0],
        normalized_waveforms[0],
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        detrend=detrend,
    )
    
    M = len(freqs)  # Number of frequency bins
    
    # Initialize CSD matrix: shape (M, N, N)
    # P[f_idx, i, j] = cross-spectral density between station i and j at freq f_idx
    csd_matrix = np.zeros((M, N, N), dtype=complex)
    
    # Compute all pairwise cross-spectral densities
    for i in range(N):
        for j in range(N):
            _, csd_ij = signal.csd(
                normalized_waveforms[i],
                normalized_waveforms[j],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window=window,
                detrend=detrend,
            )
            csd_matrix[:, i, j] = csd_ij
    
    # Perform eigendecomposition at each frequency
    eigvals = np.zeros(M, dtype=float)
    eigvecs = np.zeros((M, N), dtype=complex)
    
    for f_idx in range(M):
        # Extract CSD matrix at this frequency
        P_f = csd_matrix[f_idx, :, :]
        
        # Ensure Hermitian symmetry (numerical stability)
        P_f = 0.5 * (P_f + P_f.conj().T)
        
        # Eigendecomposition: returns eigenvalues in ascending order
        eig_vals, eig_vecs = eigh(P_f)
        
        # Principal (maximum) eigenvalue and eigenvector
        principal_idx = np.argmax(eig_vals)
        eigvals[f_idx] = eig_vals[principal_idx].real
        eigvecs[f_idx, :] = eig_vecs[:, principal_idx]
    
    return freqs, eigvals, eigvecs


def find_resonant_peak(
    freqs: np.ndarray,
    eigvals: np.ndarray,
    band: Optional[Tuple[float, float]] = None,
    threshold_factor: float = 0.5,
    min_peak_distance: int = 5,
) -> Tuple[float, float, int]:
    """
    Detect resonant frequency from modal spectral curve.
    
    Identifies the frequency with maximum principal eigenvalue (coherent power)
    within a specified frequency band. This corresponds to the resonant frequency
    where the crustal structure exhibits maximum coherent oscillation.
    
    The function uses scipy's peak detection with:
    - Minimum amplitude threshold (relative to global max)
    - Minimum distance between peaks (to avoid spurious detections)
    
    Physics Interpretation:
    -----------------------
    The resonant frequency f_peak satisfies:
    
        f_peak = argmax_f |λmax(f)|  for f ∈ [f_min, f_max]
    
    where λmax(f) is the principal eigenvalue of the cross-spectral density
    matrix. A sharp peak indicates high-Q resonance (low damping), while a
    broad peak suggests low-Q (high damping).
    
    Args:
        freqs: Frequency array (Hz), monotonically increasing
        eigvals: Principal eigenvalue at each frequency
                Must have same length as freqs
        band: Optional (f_min, f_max) to restrict search range
             If None, searches entire frequency range
        threshold_factor: Minimum relative amplitude for peak detection
                         Peak must satisfy: eigval > threshold_factor * max(eigvals)
                         Range: [0, 1], default: 0.5 (50% of maximum)
        min_peak_distance: Minimum number of frequency bins between peaks
                          Prevents detecting closely-spaced spurious peaks
    
    Returns:
        f_peak: Resonant frequency (Hz)
               Returns NaN if no peak detected
        amp_peak: Principal eigenvalue amplitude at f_peak
                 Returns NaN if no peak detected
        idx_peak: Index of peak in freqs array
                 Returns -1 if no peak detected
    
    Raises:
        ValueError: If freqs and eigvals have different lengths
        ValueError: If band is provided but f_max <= f_min
        ValueError: If threshold_factor not in [0, 1]
    
    Example:
        >>> freqs, eigvals, _ = spectral_modal_curve(waves, fs=100)
        >>> f_peak, amp_peak, idx = find_resonant_peak(
        ...     freqs, eigvals, band=(0.1, 10.0)
        ... )
        >>> if not np.isnan(f_peak):
        ...     print(f"Detected resonance at {f_peak:.3f} Hz")
        ...     print(f"Amplitude: {amp_peak:.2e}")
    
    Notes:
        - Returns NaN values if no peak exceeds threshold
        - For multiple peaks, returns the highest amplitude
        - Band edges are inclusive: f_min <= f <= f_max
        - Consider smoothing eigvals before peak detection if noisy
    
    References:
        Aki & Richards (2002), Section 5.4: Resonance in Layered Media
    """
    # Input validation
    if len(freqs) != len(eigvals):
        raise ValueError(
            f"freqs and eigvals must have same length. "
            f"Got {len(freqs)} and {len(eigvals)}"
        )
    
    if not (0 <= threshold_factor <= 1):
        raise ValueError(f"threshold_factor must be in [0, 1], got {threshold_factor}")
    
    # Apply frequency band filter
    if band is not None:
        f_min, f_max = band
        if f_max <= f_min:
            raise ValueError(f"Invalid band: f_max ({f_max}) <= f_min ({f_min})")
        
        # Create mask for frequency range
        mask = (freqs >= f_min) & (freqs <= f_max)
        freqs_filtered = freqs[mask]
        eigvals_filtered = eigvals[mask]
        
        # Map indices back to original array
        original_indices = np.where(mask)[0]
    else:
        freqs_filtered = freqs
        eigvals_filtered = eigvals
        original_indices = np.arange(len(freqs))
    
    # Check if filtered range is empty
    if len(freqs_filtered) == 0:
        return np.nan, np.nan, -1
    
    # Find peaks using scipy
    global_max = np.max(eigvals_filtered)
    threshold = threshold_factor * global_max
    
    peak_indices, properties = signal.find_peaks(
        eigvals_filtered,
        height=threshold,
        distance=min_peak_distance,
    )
    
    # No peaks detected
    if len(peak_indices) == 0:
        return np.nan, np.nan, -1
    
    # Select peak with maximum amplitude
    peak_amplitudes = eigvals_filtered[peak_indices]
    max_peak_idx_local = peak_indices[np.argmax(peak_amplitudes)]
    
    # Map back to original index
    max_peak_idx = original_indices[max_peak_idx_local]
    
    f_peak = freqs[max_peak_idx]
    amp_peak = eigvals[max_peak_idx]
    
    return f_peak, amp_peak, max_peak_idx


def geom_resonant_freq(
    area_km2: float,
    vs_km_s: float,
    geometry_factor: float = 0.5,
) -> float:
    """
    Calculate geometry-based resonance frequency estimate.
    
    For a crustal region of characteristic length L, the fundamental resonance
    frequency is approximately:
    
        f_geom ≈ Vs / (2 * L)
    
    where Vs is the shear wave velocity and L is derived from the region's area.
    This is analogous to the fundamental mode of a resonating cavity or plate.
    
    Physics Background:
    -------------------
    Crustal structures can resonate when their dimensions match integer multiples
    of the seismic wavelength. The fundamental mode (n=1) occurs when:
    
        L ≈ λ / 2 = Vs / (2*f)
    
    Solving for frequency: f ≈ Vs / (2*L)
    
    For a region with area A, we approximate the characteristic length as:
        L ≈ sqrt(A)  [assumes roughly square/circular region]
    
    The geometry_factor allows tuning for non-square geometries:
        L = geometry_factor * sqrt(A)
    
    Typical Values:
    ---------------
    - Vs (crustal): 3.0 - 4.0 km/s
    - Vs (upper mantle): 4.5 - 5.0 km/s
    - Geometry factor: 0.4 - 0.6 (calibrated to observations)
    
    Args:
        area_km2: Region area in square kilometers
                 Must be positive
        vs_km_s: Shear wave velocity in km/s
                Typical range: 2.5 - 5.0 km/s
        geometry_factor: Dimensionless factor for characteristic length
                        L = geometry_factor * sqrt(area_km2)
                        Default: 0.5 (appropriate for equidimensional regions)
    
    Returns:
        f_est_hz: Estimated fundamental resonance frequency in Hz
                 Returns NaN if inputs are invalid
    
    Raises:
        ValueError: If area_km2 or vs_km_s are non-positive
        ValueError: If geometry_factor is non-positive
    
    Example:
        >>> # Tokyo region: ~50 km × 50 km = 2500 km²
        >>> area = 2500.0  # km²
        >>> vs = 3.5  # km/s (typical crustal value)
        >>> f_est = geom_resonant_freq(area, vs)
        >>> print(f"Estimated resonance: {f_est:.3f} Hz")
        Estimated resonance: 0.035 Hz
        >>> 
        >>> # Period: T = 1/f ≈ 28.6 seconds
        >>> print(f"Period: {1/f_est:.1f} seconds")
    
    Notes:
        - This is an order-of-magnitude estimate, not a precise prediction
        - Actual resonance depends on 3D velocity structure, damping, boundary conditions
        - Compare with observed f_peak from spectral_modal_curve for validation
        - Large discrepancy may indicate complex velocity structure
    
    References:
        Aki & Richards (2002), Section 7.2: Surface Waves in Layered Media
        Stein & Wysession (2003), "An Introduction to Seismology, Earthquakes,
                                   and Earth Structure", Chapter 4
    """
    # Input validation
    if area_km2 <= 0:
        raise ValueError(f"area_km2 must be positive, got {area_km2}")
    
    if vs_km_s <= 0:
        raise ValueError(f"vs_km_s must be positive, got {vs_km_s}")
    
    if geometry_factor <= 0:
        raise ValueError(f"geometry_factor must be positive, got {geometry_factor}")
    
    # Calculate characteristic length (km)
    L_km = geometry_factor * np.sqrt(area_km2)
    
    # Fundamental frequency (Hz)
    # f = Vs / (2*L)
    # Convert: Vs [km/s] / L [km] = f [1/s = Hz]
    f_est_hz = vs_km_s / (2.0 * L_km)
    
    return f_est_hz


def compute_q_factor(
    freqs: np.ndarray,
    eigvals: np.ndarray,
    f_peak: float,
    idx_peak: int,
    half_power_ratio: float = 0.7071,  # 1/sqrt(2) ≈ -3 dB
) -> float:
    """
    Compute quality factor (Q) from half-power bandwidth.
    
    The Q-factor quantifies the sharpness of a resonance peak and is inversely
    related to damping. High Q indicates low damping (sharp peak), while low Q
    indicates high damping (broad peak).
    
    Definition:
    -----------
        Q = f_peak / Δf
    
    where Δf is the full width at half-maximum (FWHM) of the resonance peak.
    
    Half-Power Points:
    ------------------
    The half-power bandwidth is measured at the frequencies where the amplitude
    drops to half_power_ratio * peak_amplitude. For power spectra:
    
        half_power_ratio = 1/sqrt(2) ≈ 0.7071 (corresponds to -3 dB)
    
    Physics Interpretation:
    -----------------------
    - Q >> 100: Very low damping, sharp resonance (e.g., tuning fork)
    - Q ~ 10-100: Moderate damping (typical for crustal resonances)
    - Q < 10: High damping, broad resonance (e.g., soft sediments)
    
    Relationship to Attenuation:
    ----------------------------
        Q^-1 ≈ ΔE / (2π E)
    
    where ΔE/E is the fractional energy loss per cycle.
    
    Args:
        freqs: Frequency array (Hz)
        eigvals: Principal eigenvalue at each frequency
        f_peak: Resonant frequency (Hz)
        idx_peak: Index of peak in freqs array
        half_power_ratio: Ratio for half-power points
                         Default: 0.7071 (1/sqrt(2), standard definition)
    
    Returns:
        Q: Quality factor (dimensionless)
          Returns NaN if bandwidth cannot be determined
    
    Raises:
        ValueError: If idx_peak is out of bounds
        ValueError: If half_power_ratio not in (0, 1)
    
    Example:
        >>> f_peak, amp_peak, idx = find_resonant_peak(freqs, eigvals)
        >>> Q = compute_q_factor(freqs, eigvals, f_peak, idx)
        >>> print(f"Q-factor: {Q:.1f}")
        >>> print(f"Damping ratio: ζ = {1/(2*Q):.4f}")
    
    Notes:
        - Returns NaN if peak is at edge of frequency range
        - Returns NaN if half-power points cannot be found
        - For very sharp peaks, Q may be limited by frequency resolution
    
    References:
        Aki & Richards (2002), Section 5.6: Anelastic Attenuation
        Bath (1974), "Spectral Analysis in Geophysics"
    """
    # Input validation
    if not (0 < half_power_ratio < 1):
        raise ValueError(f"half_power_ratio must be in (0,1), got {half_power_ratio}")
    
    if idx_peak < 0 or idx_peak >= len(freqs):
        raise ValueError(f"idx_peak {idx_peak} out of range [0, {len(freqs)})")
    
    # Half-power threshold
    amp_peak = eigvals[idx_peak]
    threshold = half_power_ratio * amp_peak
    
    # Find left half-power point (searching backwards from peak)
    idx_left = idx_peak
    while idx_left > 0 and eigvals[idx_left] > threshold:
        idx_left -= 1
    
    # Find right half-power point (searching forwards from peak)
    idx_right = idx_peak
    while idx_right < len(eigvals) - 1 and eigvals[idx_right] > threshold:
        idx_right += 1
    
    # Check if we hit the boundaries
    if idx_left == 0 or idx_right == len(eigvals) - 1:
        return np.nan
    
    # Linear interpolation to estimate exact half-power frequencies
    # Left side
    if idx_left < len(eigvals) - 1:
        f_left = np.interp(
            threshold,
            [eigvals[idx_left], eigvals[idx_left + 1]],
            [freqs[idx_left], freqs[idx_left + 1]],
        )
    else:
        f_left = freqs[idx_left]
    
    # Right side
    if idx_right > 0:
        f_right = np.interp(
            threshold,
            [eigvals[idx_right - 1], eigvals[idx_right]],
            [freqs[idx_right - 1], freqs[idx_right]],
        )
    else:
        f_right = freqs[idx_right]
    
    # Full width at half maximum
    delta_f = f_right - f_left
    
    if delta_f <= 0:
        return np.nan
    
    # Q-factor
    Q = f_peak / delta_f
    
    return Q


def transfer_function(
    event_signal: np.ndarray,
    response_signal: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    regularization: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency response (transfer function) between two signals.
    
    The transfer function H(f) characterizes how a system transforms an input
    signal (event) into an output signal (response). For seismic applications,
    this can represent:
    - Site amplification: bedrock motion → surface motion
    - Path effects: source → receiver
    - Instrument response: ground motion → seismometer output
    
    Mathematical Definition:
    ------------------------
        H(f) = S_xy(f) / S_xx(f)
    
    where:
        S_xy(f) = cross-power spectral density (event × response*)
        S_xx(f) = auto-power spectral density of event (|event|²)
    
    The magnitude |H(f)| represents amplification/attenuation at frequency f,
    while the phase arg[H(f)] represents time delay.
    
    Coherence:
    ----------
    The coherence function measures the quality of the transfer function:
    
        γ²(f) = |S_xy(f)|² / (S_xx(f) S_yy(f))
    
    Values near 1 indicate good linear relationship; values near 0 indicate
    noise or nonlinear effects.
    
    Args:
        event_signal: Input signal (e.g., bedrock motion, source wavelet)
                     1D array
        response_signal: Output signal (e.g., surface motion, receiver recording)
                        Must have same length as event_signal
        fs: Sampling frequency (Hz)
        nperseg: Length of each segment for Welch's method
        noverlap: Number of overlapping points
                 Default: nperseg // 2
        window: Window function ('hann', 'hamming', 'blackman')
        regularization: Small constant added to denominator for numerical stability
                       Prevents division by zero for low-energy frequencies
    
    Returns:
        freqs: Frequency array (Hz)
        H: Complex transfer function at each frequency
          Magnitude: |H(f)| = amplification factor
          Phase: arg[H(f)] = phase shift (radians)
    
    Raises:
        ValueError: If signals have different lengths
        ValueError: If regularization is negative
    
    Example:
        >>> # Site amplification: bedrock → surface
        >>> freqs, H = transfer_function(bedrock_motion, surface_motion, fs=100)
        >>> amplification = np.abs(H)
        >>> phase_delay = np.angle(H) / (2*np.pi*freqs)  # Convert to time delay
        >>> 
        >>> # Find resonant frequencies (peaks in amplification)
        >>> resonances = freqs[signal.find_peaks(amplification)[0]]
        >>> print(f"Site resonances: {resonances} Hz")
    
    Notes:
        - Regularization improves stability but may bias low-amplitude frequencies
        - For high coherence (γ² > 0.9), H(f) is reliable
        - For low coherence (γ² < 0.5), H(f) may be unreliable (noise-dominated)
        - Phase wrapping: phase jumps by 2π are ambiguous
    
    References:
        Bendat & Piersol (2010), Chapter 6: Frequency Response Functions
        Kramer (1996), "Geotechnical Earthquake Engineering", Chapter 6
        Field et al. (1992): "Nonlinear site response", BSSA
    """
    # Input validation
    if len(event_signal) != len(response_signal):
        raise ValueError(
            f"Signals must have same length. "
            f"Got {len(event_signal)} and {len(response_signal)}"
        )
    
    if regularization < 0:
        raise ValueError(f"regularization must be non-negative, got {regularization}")
    
    # Set default overlap
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Compute cross-power spectral density S_xy(f)
    freqs, S_xy = signal.csd(
        event_signal,
        response_signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        detrend='constant',
    )
    
    # Compute auto-power spectral density S_xx(f)
    _, S_xx = signal.csd(
        event_signal,
        event_signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        detrend='constant',
    )
    
    # Transfer function with regularization
    # H(f) = S_xy(f) / (S_xx(f) + ε)
    H = S_xy / (S_xx + regularization)
    
    return freqs, H


if __name__ == "__main__":
    # Example: Synthetic test with 3 coherent sine waves
    print("Resonance Analysis Module - Synthetic Test")
    print("=" * 60)
    
    # Parameters
    fs = 100.0  # Hz
    duration = 10.0  # seconds
    f_true = 5.0  # Hz (true resonant frequency)
    
    # Time array
    t = np.arange(0, duration, 1/fs)
    
    # Generate 3 coherent sine waves with small phase shifts (simulating array)
    waves = {
        'station1': np.sin(2 * np.pi * f_true * t),
        'station2': np.sin(2 * np.pi * f_true * t + 0.1),  # +0.1 rad phase
        'station3': np.sin(2 * np.pi * f_true * t + 0.2),  # +0.2 rad phase
    }
    
    # Add small noise
    noise_level = 0.05
    for sid in waves:
        waves[sid] += noise_level * np.random.randn(len(t))
    
    # Compute modal spectral curve
    print("\n1. Computing modal spectral curve...")
    freqs, eigvals, eigvecs = spectral_modal_curve(waves, fs=fs, nperseg=512)
    print(f"   Frequency range: {freqs.min():.3f} - {freqs.max():.3f} Hz")
    print(f"   Number of frequency bins: {len(freqs)}")
    
    # Find resonant peak
    print("\n2. Detecting resonant peak...")
    f_peak, amp_peak, idx_peak = find_resonant_peak(freqs, eigvals, band=(0.1, 10.0))
    print(f"   Detected f_peak: {f_peak:.3f} Hz (true: {f_true:.3f} Hz)")
    print(f"   Error: {abs(f_peak - f_true):.3f} Hz ({100*abs(f_peak - f_true)/f_true:.1f}%)")
    print(f"   Amplitude: {amp_peak:.2e}")
    
    # Compute Q-factor
    print("\n3. Computing Q-factor...")
    Q = compute_q_factor(freqs, eigvals, f_peak, idx_peak)
    print(f"   Q-factor: {Q:.1f}")
    print(f"   Damping ratio: ζ = {1/(2*Q):.4f}")
    
    # Geometry-based estimate
    print("\n4. Geometry-based frequency estimate...")
    area_km2 = 2500.0  # 50 km × 50 km
    vs_km_s = 3.5  # km/s
    f_geom = geom_resonant_freq(area_km2, vs_km_s)
    print(f"   Area: {area_km2} km²")
    print(f"   Vs: {vs_km_s} km/s")
    print(f"   Estimated f_geom: {f_geom:.4f} Hz")
    print(f"   Period: {1/f_geom:.1f} seconds")
    
    # Transfer function example
    print("\n5. Transfer function example...")
    event = waves['station1']
    response = waves['station2']
    freqs_tf, H = transfer_function(event, response, fs=fs)
    magnitude = np.abs(H)
    phase = np.angle(H)
    peak_tf_idx = np.argmax(magnitude)
    print(f"   Peak amplification at f = {freqs_tf[peak_tf_idx]:.3f} Hz")
    print(f"   Amplification factor: {magnitude[peak_tf_idx]:.2f}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
