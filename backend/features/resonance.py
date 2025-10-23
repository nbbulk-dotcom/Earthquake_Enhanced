"""
Resonance Module for Earthquake_Enhanced System
Implements strain-rate resonance analysis for seismic activity prediction

Features:
- Strain-rate tensor calculations
- Crustal stress resonance analysis
- Tectonic plate boundary resonance
- Harmonic frequency detection
- Seismic wave propagation modeling

Author: BRETT System
Version: 1.0.0
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResonanceEngine:
    """
    Resonance Engine for strain-rate and crustal stress analysis
    """
    
    def __init__(self):
        """Initialize Resonance Engine with seismic constants"""
        self.version = "RESONANCE-ENGINE-V1.0.0"
        self.engine_id = "RESONANCE_ENGINE_SEISMIC"
        
        # Seismic Constants
        self.SCHUMANN_BASE_FREQ = 7.83  # Hz
        self.SCHUMANN_HARMONICS = [7.83, 14.3, 20.8, 26.7, 33.8]  # Hz
        self.EARTH_RADIUS_KM = 6371.0
        self.CRUSTAL_THICKNESS_KM = 35.0  # Average continental crust
        
        # Elastic Properties
        self.ELASTIC_MODULUS_PA = 2.0e11  # Young's modulus (Pa)
        self.SHEAR_MODULUS_PA = 8.0e10  # Shear modulus (Pa)
        self.BULK_MODULUS_PA = 1.6e11  # Bulk modulus (Pa)
        self.POISSON_RATIO = 0.25
        
        # Seismic Wave Velocities (km/s)
        self.VP_CRUST = 6.1  # P-wave velocity in crust
        self.VS_CRUST = 3.5  # S-wave velocity in crust
        self.VP_MANTLE = 8.1  # P-wave velocity in mantle
        self.VS_MANTLE = 4.5  # S-wave velocity in mantle
        
        # Strain-Rate Thresholds (nanostrain/year)
        self.STRAIN_RATE_LOW = 1e-9
        self.STRAIN_RATE_MODERATE = 1e-8
        self.STRAIN_RATE_HIGH = 1e-7
        self.STRAIN_RATE_CRITICAL = 1e-6
        
        # Tectonic Plate Boundaries
        self.plate_boundaries = self._initialize_plate_boundaries()
        
        self.last_calculation = None
    
    def _initialize_plate_boundaries(self) -> Dict[str, Dict]:
        """Initialize major tectonic plate boundaries"""
        return {
            'pacific_ring_of_fire': {
                'type': 'convergent',
                'stress_accumulation_rate': 5.0,  # cm/year
                'resonance_frequency': 0.015,  # Hz
                'lat_range': (-60, 60),
                'lon_range': (90, -90)
            },
            'san_andreas_fault': {
                'type': 'transform',
                'stress_accumulation_rate': 4.5,  # cm/year
                'resonance_frequency': 0.012,  # Hz
                'lat_range': (32, 42),
                'lon_range': (-125, -115)
            },
            'himalayan_convergent': {
                'type': 'convergent',
                'stress_accumulation_rate': 6.0,  # cm/year
                'resonance_frequency': 0.018,  # Hz
                'lat_range': (25, 40),
                'lon_range': (70, 95)
            },
            'mid_atlantic_ridge': {
                'type': 'divergent',
                'stress_accumulation_rate': 2.5,  # cm/year
                'resonance_frequency': 0.008,  # Hz
                'lat_range': (-60, 70),
                'lon_range': (-40, -10)
            },
            'japan_trench': {
                'type': 'convergent',
                'stress_accumulation_rate': 8.0,  # cm/year
                'resonance_frequency': 0.022,  # Hz
                'lat_range': (30, 45),
                'lon_range': (138, 148)
            },
            'andean_subduction': {
                'type': 'convergent',
                'stress_accumulation_rate': 7.0,  # cm/year
                'resonance_frequency': 0.020,  # Hz
                'lat_range': (-55, 10),
                'lon_range': (-80, -65)
            }
        }
    
    # ========== Strain-Rate Tensor Calculations ==========
    
    def calculate_strain_rate_tensor(self, velocity_gradients: np.ndarray) -> Dict[str, Any]:
        """
        Calculate strain-rate tensor from velocity gradients
        
        Strain rate tensor: ε̇ᵢⱼ = 1/2 (∂vᵢ/∂xⱼ + ∂vⱼ/∂xᵢ)
        
        Args:
            velocity_gradients: 3x3 velocity gradient tensor (∂vᵢ/∂xⱼ)
            
        Returns:
            Strain-rate tensor and derived quantities
        """
        # Calculate symmetric strain-rate tensor
        strain_rate = 0.5 * (velocity_gradients + velocity_gradients.T)
        
        # Calculate principal strain rates (eigenvalues)
        eigenvalues, eigenvectors = np.linalg.eig(strain_rate)
        principal_strain_rates = np.sort(np.real(eigenvalues))[::-1]
        
        # Calculate strain-rate invariants
        first_invariant = np.trace(strain_rate)  # Volumetric strain rate
        second_invariant = 0.5 * (first_invariant**2 - np.trace(strain_rate @ strain_rate))
        third_invariant = np.linalg.det(strain_rate)
        
        # Calculate maximum shear strain rate
        max_shear_strain_rate = 0.5 * (principal_strain_rates[0] - principal_strain_rates[2])
        
        # Calculate deviatoric strain rate
        deviatoric_strain = strain_rate - (first_invariant / 3.0) * np.eye(3)
        
        # Von Mises equivalent strain rate
        von_mises_strain_rate = math.sqrt(1.5 * np.sum(deviatoric_strain**2))
        
        return {
            'strain_rate_tensor': strain_rate.tolist(),
            'principal_strain_rates': principal_strain_rates.tolist(),
            'principal_directions': eigenvectors.tolist(),
            'volumetric_strain_rate': first_invariant,
            'first_invariant': first_invariant,
            'second_invariant': second_invariant,
            'third_invariant': third_invariant,
            'max_shear_strain_rate': max_shear_strain_rate,
            'von_mises_strain_rate': von_mises_strain_rate,
            'deviatoric_strain_tensor': deviatoric_strain.tolist()
        }
    
    def calculate_stress_from_strain_rate(self, strain_rate_tensor: np.ndarray,
                                         viscosity: float = 1e19) -> Dict[str, Any]:
        """
        Calculate stress tensor from strain-rate using constitutive relation
        
        Stress: σᵢⱼ = 2μ ε̇ᵢⱼ + λ δᵢⱼ ε̇ₖₖ (for viscous material)
        
        Args:
            strain_rate_tensor: 3x3 strain-rate tensor
            viscosity: Dynamic viscosity (Pa·s), default for lower crust
            
        Returns:
            Stress tensor and derived quantities
        """
        # Calculate volumetric strain rate
        volumetric_strain_rate = np.trace(strain_rate_tensor)
        
        # Lamé parameters
        mu = viscosity  # Shear viscosity
        lambda_param = viscosity  # Bulk viscosity (simplified)
        
        # Calculate stress tensor
        stress_tensor = (2 * mu * strain_rate_tensor + 
                        lambda_param * volumetric_strain_rate * np.eye(3))
        
        # Principal stresses
        principal_stresses = np.linalg.eigvals(stress_tensor)
        principal_stresses = np.sort(np.real(principal_stresses))[::-1]
        
        # Maximum shear stress
        max_shear_stress = 0.5 * (principal_stresses[0] - principal_stresses[2])
        
        # Mean stress
        mean_stress = np.mean(principal_stresses)
        
        # Von Mises stress
        deviatoric_stress = stress_tensor - mean_stress * np.eye(3)
        von_mises_stress = math.sqrt(1.5 * np.sum(deviatoric_stress**2))
        
        return {
            'stress_tensor': stress_tensor.tolist(),
            'principal_stresses': principal_stresses.tolist(),
            'max_shear_stress': max_shear_stress,
            'mean_stress': mean_stress,
            'von_mises_stress': von_mises_stress,
            'deviatoric_stress': deviatoric_stress.tolist()
        }
    
    # ========== Crustal Stress Resonance ==========
    
    def calculate_crustal_resonance(self, latitude: float, longitude: float,
                                   depth_km: float = 15.0) -> Dict[str, Any]:
        """
        Calculate crustal stress resonance at location
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            depth_km: Depth below surface in km
            
        Returns:
            Crustal resonance parameters
        """
        # Identify nearby plate boundary
        plate_boundary = self._find_nearest_plate_boundary(latitude, longitude)
        
        # Calculate distance to boundary
        distance_to_boundary = self._calculate_boundary_distance(
            latitude, longitude, plate_boundary
        )
        
        # Base resonance frequency from depth
        base_frequency = self._calculate_depth_resonance_frequency(depth_km)
        
        # Modify by plate boundary influence
        if distance_to_boundary < 500:  # Within 500 km of boundary
            boundary_influence = 1.0 - (distance_to_boundary / 500.0)
            boundary_frequency = plate_boundary['resonance_frequency']
            
            # Combine frequencies
            combined_frequency = (base_frequency * (1 - boundary_influence) + 
                                 boundary_frequency * boundary_influence)
        else:
            combined_frequency = base_frequency
            boundary_influence = 0.0
        
        # Calculate resonance amplitude
        stress_rate = plate_boundary['stress_accumulation_rate']
        resonance_amplitude = self._calculate_resonance_amplitude(
            combined_frequency, stress_rate, depth_km
        )
        
        # Calculate harmonics
        harmonics = [combined_frequency * n for n in range(1, 6)]
        
        return {
            'location': {'latitude': latitude, 'longitude': longitude},
            'depth_km': depth_km,
            'base_resonance_frequency': base_frequency,
            'combined_resonance_frequency': combined_frequency,
            'resonance_amplitude': resonance_amplitude,
            'harmonics': harmonics,
            'nearest_plate_boundary': plate_boundary['type'],
            'distance_to_boundary_km': distance_to_boundary,
            'boundary_influence_factor': boundary_influence,
            'stress_accumulation_rate_cm_year': stress_rate
        }
    
    def _find_nearest_plate_boundary(self, lat: float, lon: float) -> Dict:
        """Find nearest tectonic plate boundary"""
        min_distance = float('inf')
        nearest_boundary = None
        
        for name, boundary in self.plate_boundaries.items():
            # Check if within boundary region
            lat_in_range = boundary['lat_range'][0] <= lat <= boundary['lat_range'][1]
            lon_in_range = (min(boundary['lon_range']) <= lon <= 
                           max(boundary['lon_range']))
            
            if lat_in_range and lon_in_range:
                return boundary
            
            # Calculate approximate distance to boundary region
            lat_dist = min(abs(lat - boundary['lat_range'][0]), 
                          abs(lat - boundary['lat_range'][1]))
            lon_dist = min(abs(lon - boundary['lon_range'][0]), 
                          abs(lon - boundary['lon_range'][1]))
            
            distance = math.sqrt(lat_dist**2 + lon_dist**2) * 111  # Approximate km
            
            if distance < min_distance:
                min_distance = distance
                nearest_boundary = boundary
        
        return nearest_boundary if nearest_boundary else self.plate_boundaries['pacific_ring_of_fire']
    
    def _calculate_boundary_distance(self, lat: float, lon: float,
                                    boundary: Dict) -> float:
        """Calculate distance to plate boundary in km"""
        # Simplified distance calculation to boundary center
        lat_center = sum(boundary['lat_range']) / 2.0
        lon_center = sum(boundary['lon_range']) / 2.0
        
        # Great circle distance approximation
        dlat = math.radians(lat - lat_center)
        dlon = math.radians(lon - lon_center)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat)) * math.cos(math.radians(lat_center)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        distance_km = self.EARTH_RADIUS_KM * c
        return distance_km
    
    def _calculate_depth_resonance_frequency(self, depth_km: float) -> float:
        """
        Calculate resonance frequency based on depth
        
        Frequency decreases with depth due to increased pressure/temperature
        """
        # Surface frequency is Schumann resonance
        surface_freq = self.SCHUMANN_BASE_FREQ
        
        # Frequency decreases exponentially with depth
        depth_factor = math.exp(-depth_km / 50.0)  # 50 km characteristic depth
        
        frequency = surface_freq * depth_factor
        return frequency
    
    def _calculate_resonance_amplitude(self, frequency: float, 
                                      stress_rate: float,
                                      depth_km: float) -> float:
        """Calculate resonance amplitude from stress accumulation"""
        # Amplitude proportional to stress rate and frequency
        # Decreases with depth
        depth_attenuation = math.exp(-depth_km / 30.0)
        
        amplitude = stress_rate * frequency * depth_attenuation * 0.1
        return amplitude
    
    # ========== Harmonic Frequency Detection ==========
    
    def detect_harmonic_resonances(self, frequency_spectrum: np.ndarray,
                                  frequencies: np.ndarray) -> Dict[str, Any]:
        """
        Detect harmonic resonances in frequency spectrum
        
        Args:
            frequency_spectrum: Power spectrum values
            frequencies: Corresponding frequency values (Hz)
            
        Returns:
            Detected harmonics and their properties
        """
        # Find peaks in spectrum
        peaks = self._find_spectral_peaks(frequency_spectrum, frequencies)
        
        # Identify harmonic relationships
        harmonics = self._identify_harmonics(peaks)
        
        # Calculate resonance quality factors
        q_factors = self._calculate_q_factors(frequency_spectrum, frequencies, peaks)
        
        # Find fundamental frequency
        if harmonics:
            fundamental = min(harmonics.keys())
        else:
            fundamental = peaks[0] if peaks else 0.0
        
        return {
            'fundamental_frequency': fundamental,
            'detected_peaks': peaks,
            'harmonic_series': harmonics,
            'quality_factors': q_factors,
            'total_harmonics_detected': len(harmonics),
            'spectral_centroid': self._calculate_spectral_centroid(
                frequency_spectrum, frequencies
            )
        }
    
    def _find_spectral_peaks(self, spectrum: np.ndarray, 
                            frequencies: np.ndarray,
                            threshold_ratio: float = 0.1) -> List[float]:
        """Find peaks in frequency spectrum"""
        peaks = []
        threshold = np.max(spectrum) * threshold_ratio
        
        for i in range(1, len(spectrum) - 1):
            if (spectrum[i] > spectrum[i-1] and 
                spectrum[i] > spectrum[i+1] and 
                spectrum[i] > threshold):
                peaks.append(frequencies[i])
        
        return peaks
    
    def _identify_harmonics(self, peaks: List[float], 
                           tolerance: float = 0.05) -> Dict[float, List[float]]:
        """Identify harmonic relationships between peaks"""
        if not peaks:
            return {}
        
        harmonics = {}
        peaks_sorted = sorted(peaks)
        
        for fundamental in peaks_sorted:
            harmonic_series = [fundamental]
            
            for peak in peaks_sorted:
                if peak == fundamental:
                    continue
                
                # Check if peak is harmonic of fundamental
                ratio = peak / fundamental
                nearest_integer = round(ratio)
                
                if abs(ratio - nearest_integer) / nearest_integer < tolerance:
                    harmonic_series.append(peak)
            
            if len(harmonic_series) > 1:
                harmonics[fundamental] = harmonic_series
        
        return harmonics
    
    def _calculate_q_factors(self, spectrum: np.ndarray, frequencies: np.ndarray,
                            peaks: List[float]) -> Dict[float, float]:
        """Calculate quality factors (Q) for resonance peaks"""
        q_factors = {}
        
        for peak_freq in peaks:
            # Find peak index
            peak_idx = np.argmin(np.abs(frequencies - peak_freq))
            peak_power = spectrum[peak_idx]
            
            # Find half-power bandwidth
            half_power = peak_power / math.sqrt(2)
            
            # Find bandwidth
            lower_idx = peak_idx
            while lower_idx > 0 and spectrum[lower_idx] > half_power:
                lower_idx -= 1
            
            upper_idx = peak_idx
            while upper_idx < len(spectrum) - 1 and spectrum[upper_idx] > half_power:
                upper_idx += 1
            
            bandwidth = frequencies[upper_idx] - frequencies[lower_idx]
            
            # Q factor = center_frequency / bandwidth
            if bandwidth > 0:
                q_factor = peak_freq / bandwidth
            else:
                q_factor = float('inf')
            
            q_factors[peak_freq] = q_factor
        
        return q_factors
    
    def _calculate_spectral_centroid(self, spectrum: np.ndarray,
                                    frequencies: np.ndarray) -> float:
        """Calculate spectral centroid (weighted mean frequency)"""
        total_power = np.sum(spectrum)
        if total_power == 0:
            return 0.0
        
        centroid = np.sum(frequencies * spectrum) / total_power
        return centroid
    
    # ========== Seismic Wave Propagation ==========
    
    def calculate_seismic_wave_propagation(self, source_lat: float, source_lon: float,
                                          target_lat: float, target_lon: float,
                                          depth_km: float = 15.0) -> Dict[str, Any]:
        """
        Calculate seismic wave propagation parameters
        
        Args:
            source_lat: Source latitude
            source_lon: Source longitude
            target_lat: Target latitude
            target_lon: Target longitude
            depth_km: Event depth in km
            
        Returns:
            Wave propagation parameters
        """
        # Calculate epicentral distance
        distance_km = self._calculate_epicentral_distance(
            source_lat, source_lon, target_lat, target_lon
        )
        
        # Select wave velocity based on depth
        if depth_km < self.CRUSTAL_THICKNESS_KM:
            vp = self.VP_CRUST
            vs = self.VS_CRUST
            layer = 'crust'
        else:
            vp = self.VP_MANTLE
            vs = self.VS_MANTLE
            layer = 'mantle'
        
        # Calculate travel times
        p_wave_time = distance_km / vp  # seconds
        s_wave_time = distance_km / vs  # seconds
        
        # Calculate ray parameter
        ray_parameter = self._calculate_ray_parameter(distance_km, depth_km, vp)
        
        # Calculate amplitude attenuation (geometric spreading + intrinsic)
        geometric_attenuation = 1.0 / distance_km if distance_km > 0 else 1.0
        intrinsic_attenuation = math.exp(-distance_km / 100.0)  # Q factor effect
        total_attenuation = geometric_attenuation * intrinsic_attenuation
        
        return {
            'epicentral_distance_km': distance_km,
            'depth_km': depth_km,
            'propagation_layer': layer,
            'p_wave_velocity_km_s': vp,
            's_wave_velocity_km_s': vs,
            'p_wave_travel_time_s': p_wave_time,
            's_wave_travel_time_s': s_wave_time,
            's_p_time_difference_s': s_wave_time - p_wave_time,
            'ray_parameter': ray_parameter,
            'geometric_attenuation': geometric_attenuation,
            'intrinsic_attenuation': intrinsic_attenuation,
            'total_attenuation_factor': total_attenuation
        }
    
    def _calculate_epicentral_distance(self, lat1: float, lon1: float,
                                      lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        a = (math.sin((lat2_rad - lat1_rad) / 2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon_rad / 2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        distance = self.EARTH_RADIUS_KM * c
        return distance
    
    def _calculate_ray_parameter(self, distance_km: float, depth_km: float,
                                velocity_km_s: float) -> float:
        """Calculate seismic ray parameter (p = r sin(i) / v)"""
        if distance_km == 0:
            return 0.0
        
        # Simplified ray parameter calculation
        r = self.EARTH_RADIUS_KM - depth_km
        
        # Incident angle approximation for shallow events
        incident_angle = math.atan2(distance_km, r)
        
        ray_param = (r * math.sin(incident_angle)) / (velocity_km_s * self.EARTH_RADIUS_KM)
        return ray_param
    
    # ========== Comprehensive Resonance Analysis ==========
    
    def calculate_comprehensive_resonance(self, latitude: float, longitude: float,
                                        depth_km: float = 15.0,
                                        timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive resonance analysis for location
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            depth_km: Depth below surface in km
            timestamp: Analysis timestamp (default: now)
            
        Returns:
            Comprehensive resonance analysis
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        try:
            # Crustal resonance
            crustal_resonance = self.calculate_crustal_resonance(
                latitude, longitude, depth_km
            )
            
            # Generate synthetic velocity gradients for demonstration
            # In production, these would come from GPS/geodetic measurements
            velocity_gradients = self._generate_velocity_gradients(
                latitude, longitude, crustal_resonance
            )
            
            # Strain-rate tensor
            strain_rate = self.calculate_strain_rate_tensor(velocity_gradients)
            
            # Stress tensor
            stress = self.calculate_stress_from_strain_rate(
                np.array(strain_rate['strain_rate_tensor'])
            )
            
            # Classify strain rate level
            strain_level = self._classify_strain_rate_level(
                strain_rate['von_mises_strain_rate']
            )
            
            # Calculate resonance risk score
            risk_score = self._calculate_resonance_risk_score(
                crustal_resonance, strain_rate, stress
            )
            
            result = {
                'success': True,
                'timestamp': timestamp.isoformat(),
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'depth_km': depth_km
                },
                'crustal_resonance': crustal_resonance,
                'strain_rate_analysis': strain_rate,
                'stress_analysis': stress,
                'strain_rate_level': strain_level,
                'resonance_risk_score': risk_score,
                'engine_info': {
                    'version': self.version,
                    'engine_id': self.engine_id
                }
            }
            
            self.last_calculation = result
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive resonance analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': timestamp.isoformat(),
                'location': {'latitude': latitude, 'longitude': longitude}
            }
    
    def _generate_velocity_gradients(self, lat: float, lon: float,
                                    crustal_resonance: Dict) -> np.ndarray:
        """
        Generate synthetic velocity gradients based on crustal resonance
        In production, would use real GPS/geodetic data
        """
        stress_rate = crustal_resonance['stress_accumulation_rate_cm_year']
        boundary_influence = crustal_resonance['boundary_influence_factor']
        
        # Convert stress rate to velocity gradient (simplified)
        base_gradient = stress_rate * 1e-9 * boundary_influence  # rad/year
        
        # Create velocity gradient tensor
        gradients = np.array([
            [base_gradient, base_gradient * 0.5, base_gradient * 0.3],
            [base_gradient * 0.5, base_gradient * 0.8, base_gradient * 0.2],
            [base_gradient * 0.3, base_gradient * 0.2, base_gradient * 0.6]
        ])
        
        return gradients
    
    def _classify_strain_rate_level(self, strain_rate: float) -> Dict[str, Any]:
        """Classify strain rate level"""
        if strain_rate >= self.STRAIN_RATE_CRITICAL:
            level = 'CRITICAL'
            risk = 'VERY_HIGH'
        elif strain_rate >= self.STRAIN_RATE_HIGH:
            level = 'HIGH'
            risk = 'HIGH'
        elif strain_rate >= self.STRAIN_RATE_MODERATE:
            level = 'MODERATE'
            risk = 'MODERATE'
        elif strain_rate >= self.STRAIN_RATE_LOW:
            level = 'LOW'
            risk = 'LOW'
        else:
            level = 'VERY_LOW'
            risk = 'MINIMAL'
        
        return {
            'level': level,
            'risk_category': risk,
            'strain_rate_value': strain_rate,
            'thresholds': {
                'low': self.STRAIN_RATE_LOW,
                'moderate': self.STRAIN_RATE_MODERATE,
                'high': self.STRAIN_RATE_HIGH,
                'critical': self.STRAIN_RATE_CRITICAL
            }
        }
    
    def _calculate_resonance_risk_score(self, crustal_resonance: Dict,
                                       strain_rate: Dict, stress: Dict) -> float:
        """Calculate overall resonance risk score (0-1)"""
        # Frequency factor (higher frequency = higher risk)
        freq_factor = min(crustal_resonance['combined_resonance_frequency'] / 0.1, 1.0)
        
        # Amplitude factor
        amp_factor = min(crustal_resonance['resonance_amplitude'] / 10.0, 1.0)
        
        # Strain rate factor
        strain_factor = min(strain_rate['von_mises_strain_rate'] / self.STRAIN_RATE_HIGH, 1.0)
        
        # Stress factor (von Mises stress)
        max_stress = 1e8  # 100 MPa reference
        stress_factor = min(stress['von_mises_stress'] / max_stress, 1.0)
        
        # Boundary proximity factor
        boundary_factor = crustal_resonance['boundary_influence_factor']
        
        # Combined risk score
        risk_score = (freq_factor * 0.2 + 
                     amp_factor * 0.2 + 
                     strain_factor * 0.3 + 
                     stress_factor * 0.2 + 
                     boundary_factor * 0.1)
        
        return min(1.0, max(0.0, risk_score))
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'version': self.version,
            'engine_id': self.engine_id,
            'status': 'operational',
            'last_calculation': (self.last_calculation['timestamp'] 
                               if self.last_calculation else None),
            'plate_boundaries_tracked': len(self.plate_boundaries),
            'features': [
                'strain_rate_tensor_calculations',
                'crustal_stress_resonance',
                'harmonic_frequency_detection',
                'seismic_wave_propagation',
                'comprehensive_resonance_analysis'
            ]
        }


# Convenience functions
def calculate_resonance(latitude: float, longitude: float, 
                       depth_km: float = 15.0) -> Dict:
    """Convenience function for resonance calculation"""
    engine = ResonanceEngine()
    return engine.calculate_comprehensive_resonance(latitude, longitude, depth_km)


def get_resonance_engine() -> ResonanceEngine:
    """Get a new ResonanceEngine instance"""
    return ResonanceEngine()
