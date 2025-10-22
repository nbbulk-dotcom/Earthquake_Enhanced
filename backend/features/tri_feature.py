"""
Triangle Feature Extraction with Resonance Analysis

This module extracts physical features from triangulated earthquake data,
combining spatial strain rate calculations with frequency-domain resonance
analysis. Features are descriptive (for analysis) rather than predictive.

Integration Approach:
--------------------
For each triangle in the Delaunay mesh:
1. Extract strain rate using Kostrov formula (time-domain)
2. Analyze resonance using modal spectral curves (frequency-domain)
3. Compute geometry-based frequency estimates
4. Calculate Q-factors and coherence metrics
5. Track temporal evolution of resonance patterns

This provides a multi-scale view of seismic activity:
- Strain accumulation (slow, quasi-static)
- Resonant oscillations (dynamic, frequency-dependent)

Author: Earthquake Enhanced Team
License: MIT
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

from backend.features.resonance import (
    spectral_modal_curve,
    find_resonant_peak,
    geom_resonant_freq,
    compute_q_factor,
)
from backend.calculators.seismic_moment import SeismicMomentCalculator


def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml.
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_triangle_features(
    triangle_data: Dict[str, Any],
    waveform_data: Optional[Dict[str, np.ndarray]] = None,
    sampling_rate: float = 100.0,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract comprehensive features for a single triangle.
    
    Combines spatial geometry, seismic moment statistics, and
    frequency-domain resonance analysis.
    
    Args:
        triangle_data: Dictionary containing triangle information:
            - vertices: List of 3 vertex indices
            - area_m2: Triangle area (square meters)
            - area_km2: Triangle area (square kilometers)
            - centroid_lon: Centroid longitude
            - centroid_lat: Centroid latitude
            - vertex_magnitudes: List of magnitudes at vertices
            - event_times: Optional list of event timestamps
        waveform_data: Optional dictionary mapping vertex_id -> waveform array
                      Required for resonance analysis
                      Format: {0: wave0, 1: wave1, 2: wave2}
        sampling_rate: Sampling rate of waveforms (Hz)
        config: Configuration dictionary (if None, loads from config.yaml)
    
    Returns:
        Dictionary containing extracted features:
        
        Spatial features:
        - area_km2: Triangle area
        - centroid_lon, centroid_lat: Geographic center
        
        Seismic moment features:
        - total_moment: Sum of seismic moments (N⋅m)
        - avg_magnitude: Average magnitude
        - max_magnitude: Maximum magnitude
        
        Resonance features (per frequency band):
        - resonant_frequency_hz: Detected peak frequency
        - resonant_amplitude: Principal eigenvalue amplitude
        - geom_resonant_freq_hz: Geometry-based estimate
        - Q_factor: Quality factor (sharpness of resonance)
        - modal_eigenvector: Spatial pattern at resonance
        - freq_band: Frequency band analyzed
    
    Example:
        >>> triangle = {
        ...     'area_km2': 100.0,
        ...     'centroid_lon': 139.7,
        ...     'centroid_lat': 35.7,
        ...     'vertex_magnitudes': [4.5, 5.0, 4.8],
        ... }
        >>> # Without waveforms - only spatial/moment features
        >>> features = extract_triangle_features(triangle)
        >>> 
        >>> # With waveforms - includes resonance analysis
        >>> waves = {0: wave0, 1: wave1, 2: wave2}  # vertex_id -> waveform
        >>> features = extract_triangle_features(triangle, waves, fs=100)
    
    Notes:
        - If waveform_data is None, resonance features are skipped
        - Resonance analysis is performed for each frequency band in config
        - All features are deterministic (same input → same output)
    """
    if config is None:
        config = load_config()
    
    # Initialize feature dictionary
    features = {
        'area_km2': triangle_data.get('area_km2', 0.0),
        'centroid_lon': triangle_data.get('centroid_lon', 0.0),
        'centroid_lat': triangle_data.get('centroid_lat', 0.0),
    }
    
    # Extract seismic moment features
    magnitudes = triangle_data.get('vertex_magnitudes', [])
    if magnitudes:
        calc = SeismicMomentCalculator()
        moments = [calc.magnitude_to_moment(m) for m in magnitudes]
        features['total_moment_Nm'] = sum(moments)
        features['avg_magnitude'] = np.mean(magnitudes)
        features['max_magnitude'] = np.max(magnitudes)
        features['num_events'] = len(magnitudes)
    else:
        features['total_moment_Nm'] = 0.0
        features['avg_magnitude'] = 0.0
        features['max_magnitude'] = 0.0
        features['num_events'] = 0
    
    # Resonance analysis (if waveform data provided)
    if waveform_data is not None and len(waveform_data) >= 3:
        resonance_config = config.get('resonance', {})
        frequency_bands = resonance_config.get('frequency_bands', [[0.1, 10.0]])
        vs_by_band = resonance_config.get('vs_by_band_km_s', [3.5] * len(frequency_bands))
        
        spectral_config = resonance_config.get('spectral', {})
        nperseg = spectral_config.get('nperseg', 256)
        
        peak_config = resonance_config.get('peak_detection', {})
        threshold_factor = peak_config.get('threshold_factor', 0.5)
        min_peak_distance = peak_config.get('min_peak_distance', 5)
        
        geom_config = resonance_config.get('geometry', {})
        geom_factor = geom_config.get('factor', 0.5)
        
        # Analyze each frequency band
        resonance_features = []
        
        for band_idx, (f_min, f_max) in enumerate(frequency_bands):
            try:
                # Get Vs for this band
                vs_km_s = vs_by_band[band_idx] if band_idx < len(vs_by_band) else 3.5
                
                # Compute modal spectral curve
                freqs, eigvals, eigvecs = spectral_modal_curve(
                    waveform_data,
                    fs=sampling_rate,
                    nperseg=nperseg,
                )
                
                # Find resonant peak in this band
                f_peak, amp_peak, idx_peak = find_resonant_peak(
                    freqs,
                    eigvals,
                    band=(f_min, f_max),
                    threshold_factor=threshold_factor,
                    min_peak_distance=min_peak_distance,
                )
                
                # Geometry-based estimate
                area_km2 = features['area_km2']
                if area_km2 > 0:
                    f_geom = geom_resonant_freq(area_km2, vs_km_s, geom_factor)
                else:
                    f_geom = np.nan
                
                # Q-factor (if peak detected)
                if not np.isnan(f_peak) and idx_peak >= 0:
                    Q = compute_q_factor(freqs, eigvals, f_peak, idx_peak)
                    modal_vec = eigvecs[idx_peak, :].tolist()
                else:
                    Q = np.nan
                    modal_vec = None
                
                # Store features for this band
                band_features = {
                    'freq_band_hz': [f_min, f_max],
                    'resonant_frequency_hz': f_peak,
                    'resonant_amplitude': amp_peak,
                    'geom_resonant_freq_hz': f_geom,
                    'Q_factor': Q,
                    'modal_eigenvector': modal_vec,
                    'vs_km_s': vs_km_s,
                }
                
                # Validation: compare detected vs geometry estimate
                if not np.isnan(f_peak) and not np.isnan(f_geom) and f_geom > 0:
                    freq_ratio = f_peak / f_geom
                    band_features['freq_ratio_detected_to_geom'] = freq_ratio
                else:
                    band_features['freq_ratio_detected_to_geom'] = np.nan
                
                resonance_features.append(band_features)
                
            except Exception as e:
                warnings.warn(f"Resonance analysis failed for band {[f_min, f_max]}: {e}")
                # Add placeholder with NaN values
                resonance_features.append({
                    'freq_band_hz': [f_min, f_max],
                    'resonant_frequency_hz': np.nan,
                    'resonant_amplitude': np.nan,
                    'geom_resonant_freq_hz': np.nan,
                    'Q_factor': np.nan,
                    'modal_eigenvector': None,
                    'vs_km_s': vs_km_s,
                    'freq_ratio_detected_to_geom': np.nan,
                })
        
        features['resonance_by_band'] = resonance_features
    
    return features


def extract_features_from_triangulation(
    triangulation_result: Dict[str, Any],
    waveform_dict: Optional[Dict[int, np.ndarray]] = None,
    sampling_rate: float = 100.0,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract features for all triangles in triangulation result.
    
    Args:
        triangulation_result: Output from EarthquakeTriangulation.triangulate()
            Must contain 'triangles' key with list of triangle dictionaries
        waveform_dict: Optional mapping from point_index -> waveform
                      For resonance analysis
        sampling_rate: Sampling rate of waveforms (Hz)
        config: Configuration dictionary
    
    Returns:
        List of feature dictionaries, one per triangle
    
    Example:
        >>> from backend.triangulation.triangulate import EarthquakeTriangulation
        >>> 
        >>> # Triangulate earthquakes
        >>> tri = EarthquakeTriangulation()
        >>> result = tri.triangulate(lons, lats, depths, mags)
        >>> 
        >>> # Extract features (without resonance)
        >>> features = extract_features_from_triangulation(result)
        >>> 
        >>> # Extract features with resonance
        >>> waves = {i: generate_synthetic_wave(lons[i], lats[i]) for i in range(len(lons))}
        >>> features = extract_features_from_triangulation(result, waves, fs=100)
    """
    if config is None:
        config = load_config()
    
    triangles = triangulation_result.get('triangles', [])
    all_features = []
    
    for tri_idx, triangle in enumerate(triangles):
        # Prepare waveform data for this triangle's vertices
        if waveform_dict is not None:
            vertices = triangle.get('vertices', [])
            tri_waveforms = {}
            
            # Check if all vertices have waveforms
            if all(v in waveform_dict for v in vertices):
                tri_waveforms = {
                    v: waveform_dict[v] for v in vertices
                }
            else:
                tri_waveforms = None
        else:
            tri_waveforms = None
        
        # Convert area from m² to km² if needed
        if 'area_km2' not in triangle:
            triangle['area_km2'] = triangle.get('area_m2', 0.0) / 1e6
        
        # Extract features for this triangle
        features = extract_triangle_features(
            triangle,
            waveform_data=tri_waveforms,
            sampling_rate=sampling_rate,
            config=config,
        )
        
        # Add triangle identifier
        features['triangle_id'] = tri_idx
        features['vertices'] = triangle.get('vertices', [])
        
        all_features.append(features)
    
    return all_features


def summarize_resonance_features(
    feature_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute summary statistics across all triangles.
    
    Useful for regional-scale analysis and temporal tracking.
    
    Args:
        feature_list: List of feature dictionaries from extract_features_from_triangulation
    
    Returns:
        Dictionary with summary statistics:
        - total_moment_Nm: Sum of all seismic moments
        - num_triangles: Number of triangles
        - avg_area_km2: Average triangle area
        - resonance_summary: Per-band statistics
    
    Example:
        >>> features = extract_features_from_triangulation(tri_result, waves)
        >>> summary = summarize_resonance_features(features)
        >>> print(f"Total moment: {summary['total_moment_Nm']:.2e} N⋅m")
        >>> print(f"Average Q-factor (band 0): {summary['resonance_summary'][0]['avg_Q']:.1f}")
    """
    if not feature_list:
        return {}
    
    # Spatial statistics
    areas = [f['area_km2'] for f in feature_list if 'area_km2' in f]
    moments = [f['total_moment_Nm'] for f in feature_list if 'total_moment_Nm' in f]
    
    summary = {
        'num_triangles': len(feature_list),
        'total_moment_Nm': sum(moments) if moments else 0.0,
        'avg_area_km2': np.mean(areas) if areas else 0.0,
        'total_area_km2': sum(areas) if areas else 0.0,
    }
    
    # Resonance statistics (if available)
    # Check if any features have resonance data
    has_resonance = any('resonance_by_band' in f for f in feature_list)
    
    if has_resonance:
        # Determine number of bands (from first feature with resonance)
        first_with_res = next(f for f in feature_list if 'resonance_by_band' in f)
        num_bands = len(first_with_res['resonance_by_band'])
        
        resonance_summary = []
        
        for band_idx in range(num_bands):
            # Collect values for this band across all triangles
            f_peaks = []
            amplitudes = []
            Q_factors = []
            freq_ratios = []
            
            for feat in feature_list:
                if 'resonance_by_band' in feat and band_idx < len(feat['resonance_by_band']):
                    band_data = feat['resonance_by_band'][band_idx]
                    
                    f_peak = band_data.get('resonant_frequency_hz', np.nan)
                    if not np.isnan(f_peak):
                        f_peaks.append(f_peak)
                    
                    amp = band_data.get('resonant_amplitude', np.nan)
                    if not np.isnan(amp):
                        amplitudes.append(amp)
                    
                    Q = band_data.get('Q_factor', np.nan)
                    if not np.isnan(Q):
                        Q_factors.append(Q)
                    
                    ratio = band_data.get('freq_ratio_detected_to_geom', np.nan)
                    if not np.isnan(ratio):
                        freq_ratios.append(ratio)
            
            # Compute statistics
            band_summary = {
                'band_index': band_idx,
                'freq_band_hz': first_with_res['resonance_by_band'][band_idx]['freq_band_hz'],
                'num_detections': len(f_peaks),
                'detection_rate': len(f_peaks) / len(feature_list) if feature_list else 0.0,
            }
            
            if f_peaks:
                band_summary.update({
                    'avg_resonant_freq_hz': np.mean(f_peaks),
                    'std_resonant_freq_hz': np.std(f_peaks),
                    'median_resonant_freq_hz': np.median(f_peaks),
                })
            
            if amplitudes:
                band_summary.update({
                    'avg_amplitude': np.mean(amplitudes),
                    'max_amplitude': np.max(amplitudes),
                })
            
            if Q_factors:
                band_summary.update({
                    'avg_Q_factor': np.mean(Q_factors),
                    'median_Q_factor': np.median(Q_factors),
                })
            
            if freq_ratios:
                band_summary.update({
                    'avg_freq_ratio': np.mean(freq_ratios),
                    'std_freq_ratio': np.std(freq_ratios),
                })
            
            resonance_summary.append(band_summary)
        
        summary['resonance_summary'] = resonance_summary
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Triangle Feature Extraction - Example")
    print("=" * 60)
    
    # Load config
    config = load_config()
    print(f"\nLoaded configuration:")
    print(f"  Frequency bands: {config['resonance']['frequency_bands']}")
    print(f"  Vs by band: {config['resonance']['vs_by_band_km_s']} km/s")
    
    # Example triangle data
    triangle = {
        'vertices': [0, 1, 2],
        'area_m2': 1e8,  # 100 km²
        'area_km2': 100.0,
        'centroid_lon': 139.7,
        'centroid_lat': 35.7,
        'vertex_magnitudes': [4.5, 5.0, 4.8],
    }
    
    print("\n" + "=" * 60)
    print("Test 1: Extract features without waveforms")
    print("=" * 60)
    
    features = extract_triangle_features(triangle, config=config)
    print(f"\nExtracted features:")
    print(f"  Area: {features['area_km2']:.1f} km²")
    print(f"  Centroid: ({features['centroid_lon']:.2f}, {features['centroid_lat']:.2f})")
    print(f"  Total moment: {features['total_moment_Nm']:.2e} N⋅m")
    print(f"  Average magnitude: {features['avg_magnitude']:.2f}")
    print(f"  Max magnitude: {features['max_magnitude']:.2f}")
    
    print("\n" + "=" * 60)
    print("Test 2: Extract features with synthetic waveforms")
    print("=" * 60)
    
    # Generate synthetic waveforms (coherent 2 Hz signal)
    fs = 100.0
    duration = 20.0
    t = np.arange(0, duration, 1/fs)
    f_signal = 2.0  # Hz
    
    waveforms = {
        0: np.sin(2 * np.pi * f_signal * t) + 0.1 * np.random.randn(len(t)),
        1: np.sin(2 * np.pi * f_signal * t + 0.1) + 0.1 * np.random.randn(len(t)),
        2: np.sin(2 * np.pi * f_signal * t + 0.2) + 0.1 * np.random.randn(len(t)),
    }
    
    features = extract_triangle_features(
        triangle,
        waveform_data=waveforms,
        sampling_rate=fs,
        config=config,
    )
    
    print(f"\nExtracted features (with resonance):")
    print(f"  Area: {features['area_km2']:.1f} km²")
    print(f"  Total moment: {features['total_moment_Nm']:.2e} N⋅m")
    
    if 'resonance_by_band' in features:
        print(f"\n  Resonance analysis ({len(features['resonance_by_band'])} bands):")
        for band_idx, band_feat in enumerate(features['resonance_by_band']):
            print(f"\n  Band {band_idx}: {band_feat['freq_band_hz']} Hz")
            print(f"    Detected f_peak: {band_feat['resonant_frequency_hz']:.4f} Hz")
            print(f"    Geometry f_geom: {band_feat['geom_resonant_freq_hz']:.4f} Hz")
            print(f"    Amplitude: {band_feat['resonant_amplitude']:.2e}")
            print(f"    Q-factor: {band_feat['Q_factor']:.1f}")
            print(f"    Freq ratio (det/geom): {band_feat['freq_ratio_detected_to_geom']:.2f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
