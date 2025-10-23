
"""
backend/features/resonance.py

Deterministic resonance diagnostics for a triangle of stations.

Functions:
- spectral_modal_curve: compute frequencies, principal eigenvalues and eigenvectors from CSD matrices
- find_resonant_peak: detect peak frequency and amplitude
- geom_resonant_freq: estimate geometric resonance from triangle area and shear wave speed
- transfer_function: compute empirical transfer function between two signals (regularized)

Inputs and types:
- waves: Dict[str, np.ndarray] - station_id to signal array
- fs: float - sampling rate
- nperseg: int - segment length

Outputs and types:
- Tuple[np.ndarray, np.ndarray, np.ndarray] - freqs, eigvals_max, eigvecs_max

Deterministic assumptions: No randomness; uses fixed nperseg for CSD.
Provenance expectations: Requires aligned waveforms from snapshot.
"""
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.signal import csd, welch
from numpy.linalg import eig
from math import sqrt

def spectral_modal_curve(waves: Dict[str, np.ndarray], fs: float, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    station_ids = sorted(waves.keys())
    n = len(station_ids)
    if n < 2:
        return np.array([]), np.array([]), np.array([[]])

    f, Pxx = welch(waves[station_ids[0]], fs=fs, nperseg=nperseg)
    nf = f.size
    P = np.zeros((nf, n, n), dtype=complex)
    for i in range(n):
        xi = waves[station_ids[i]]
        for j in range(i, n):
            xj = waves[station_ids[j]]
            fj, Pij = csd(xi, xj, fs=fs, nperseg=nperseg)
            if not np.array_equal(fj, f):
                Pij = np.interp(f, fj, np.abs(Pij)) * np.exp(1j * 0.0)
            P[:, i, j] = Pij
            if i != j:
                P[:, j, i] = np.conjugate(Pij)

    eigvals_max = np.zeros(nf, dtype=float)
    eigvecs_max = np.zeros((n, nf), dtype=complex)

    for k in range(nf):
        mat = P[k]
        mat = 0.5 * (mat + mat.conj().T)
        try:
            vals, vecs = eig(mat)
        except Exception:
            eigvals_max[k] = 0.0
            eigvecs_max[:, k] = 0.0
            continue
        idx = int(np.argmax(np.abs(vals)))
        eigvals_max[k] = float(np.abs(vals[idx]))
        v = vecs[:, idx]
        norm = np.linalg.norm(v)
        eigvecs_max[:, k] = (v / norm) if norm > 0 else v

    return f, eigvals_max, eigvecs_max

def find_resonant_peak(freqs: np.ndarray, eigvals: np.ndarray, band: Optional[Tuple[float, float]] = None) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    if freqs.size == 0 or eigvals.size == 0:
        return None, None, None
    mask = np.ones_like(freqs, dtype=bool)
    if band is not None:
        low, high = band
        mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return None, None, None
    masked_vals = eigvals.copy()
    masked_vals[~mask] = -np.inf
    idx = int(np.argmax(masked_vals))
    if masked_vals[idx] <= 0 or not np.isfinite(masked_vals[idx]):
        return None, None, None
    return float(freqs[idx]), float(eigvals[idx]), idx

def geom_resonant_freq(area_km2: float, vs_km_s: float = 3.5, mode_factor: float = 2.0) -> Optional[float]:
    if area_km2 <= 0 or vs_km_s <= 0:
        return None
    L = sqrt(area_km2)
    if L <= 0:
        return None
    f = vs_km_s / (mode_factor * L)
    return float(f)

def transfer_function(event_signal: np.ndarray, response_signal: np.ndarray, fs: float, nperseg: int = 256, reg: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    if event_signal.size == 0 or response_signal.size == 0:
        return np.array([]), np.array([])
    f, Pee = csd(event_signal, event_signal, fs=fs, nperseg=nperseg)
    _, Pre = csd(response_signal, event_signal, fs=fs, nperseg=nperseg)
    Pee_reg = Pee + reg * np.mean(np.abs(Pee))
    H = Pre / Pee_reg
    return f, H
