
# backend/features/tri_feature.py
import numpy as np
from typing import Dict, List, Any, Optional
from .bandpass import apply_bandpass
from .coherence import compute_coherence, compute_cross_spectrum
from .resonance import spectral_modal_curve, find_resonant_peak, geom_resonant_freq
import yaml
from pathlib import Path

# Load resonance config once (best-effort; falls back to defaults)
_cfg = {}
_cfg_path = Path(__file__).parent / "config.yaml"
if _cfg_path.exists():
    try:
        _cfg = yaml.safe_load(_cfg_path.read_text(encoding="utf-8"))
    except Exception:
        _cfg = {}
_resonance_cfg = _cfg.get("resonance", {}) if isinstance(_cfg, dict) else {}
_DEFAULT_NPERSEG = int(_resonance_cfg.get("nperseg", 256))
_DEFAULT_MODE_FACTOR = float(_resonance_cfg.get("mode_factor", 2.0))

def triangle_feature_vector(waves: Dict[str, np.ndarray], fs: float, bands: List[List[float]], area_km2: Optional[float] = None) -> Dict[str, Any]:
    """
    Compute deterministic per-triangle features including band energies, coherence, cross-spectrum magnitude,
    and resonance diagnostics (observed resonant frequency and geometric estimate).

    Inputs and types:
    - waves: Dict[str, np.ndarray] - station_id to signal array
    - fs: float - sampling rate
    - bands: List[List[float]] - list of [low, high] bands
    - area_km2: Optional[float] - triangle area for geom freq

    Outputs and types:
    - Dict[str, Any] - feature dictionary

    Deterministic assumptions: Fixed nperseg; no randomness.
    Provenance expectations: Waves from snapshot.
    """
    station_ids = sorted(waves.keys())
    n = len(station_ids)
    features = {"stations": station_ids}

    for band_idx, band in enumerate(bands):
        low, high = band
        band_key = f"band_{low}_{high}"
        energies = {}
        for sid in station_ids:
            sig = waves[sid]
            filt = apply_bandpass(sig, low, high, fs)
            energies[sid] = float(np.sum(filt * filt))
        features[f"{band_key}_energies"] = energies

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = station_ids[i], station_ids[j]
            pair_key = f"{si}__{sj}"
            pair_coh = {}
            pair_cs_mag = {}
            for band in bands:
                low, high = band
                f, Cxy = compute_coherence(waves[si], waves[sj], fs)
                mean_coh = float(np.mean(Cxy)) if Cxy.size else 0.0
                pair_coh[f"{low}_{high}"] = mean_coh
                fcs, Pxy = compute_cross_spectrum(waves[si], waves[sj], fs)
                if Pxy.size:
                    mask = (fcs >= low) & (fcs <= high)
                    mag = float(np.mean(np.abs(Pxy[mask]))) if mask.any() else 0.0
                else:
                    mag = 0.0
                pair_cs_mag[f"{low}_{high}"] = mag
            features[f"{pair_key}_coherence"] = pair_coh
            features[f"{pair_key}_cs_mag"] = pair_cs_mag

    for band_idx, band in enumerate(bands):
        low, high = band
        band_key = f"band_{low}_{high}"
        band_waves = {}
        for sid in station_ids:
            sig = waves[sid]
            filt = apply_bandpass(sig, low, high, fs)
            band_waves[sid] = filt
        freqs, eigvals, eigvecs = spectral_modal_curve(band_waves, fs, nperseg=_DEFAULT_NPERSEG)
        f_peak, amp_peak, idx = find_resonant_peak(freqs, eigvals, band=(low, high))
        features[f"{band_key}_resonant_frequency_hz"] = float(f_peak) if f_peak is not None else None
        features[f"{band_key}_resonant_amplitude"] = float(amp_peak) if amp_peak is not None else 0.0
        if idx is not None and eigvecs.size:
            vec = eigvecs[:, idx]
            modal = []
            for v in vec:
                mag = float(np.abs(v))
                phase = float(np.angle(v))
                modal.append({"magnitude": mag, "phase": phase})
            features[f"{band_key}_modal_eigenvector"] = {sid: modal[i] for i, sid in enumerate(station_ids)}
        else:
            features[f"{band_key}_modal_eigenvector"] = {sid: {"magnitude": 0.0, "phase": 0.0} for sid in station_ids}

        # Q_factor from half-power bandwidth
        if idx is not None and f_peak is not None:
            half_amp = amp_peak / 2
            left_mask = eigvals[:idx] >= half_amp
            right_mask = eigvals[idx:] >= half_amp
            left_idx = np.where(left_mask)[0][-1] if left_mask.any() else idx
            right_idx = np.where(right_mask)[0][0] + idx if right_mask.any() else idx
            bw = freqs[right_idx] - freqs[left_idx] if left_idx < idx < right_idx else 0.0
            q = f_peak / bw if bw > 0 else 0.0
            features[f"{band_key}_Q_factor"] = float(q)
        else:
            features[f"{band_key}_Q_factor"] = 0.0

        # Geometric estimate
        vs = _cfg.get('vs_by_band_km_s', [3.5]*len(bands))[band_idx]
        geom_est = geom_resonant_freq(area_km2, vs_km_s=vs, mode_factor=_DEFAULT_MODE_FACTOR)
        features[f"{band_key}_geom_resonant_freq_hz"] = float(geom_est) if geom_est is not None else None

    return features
