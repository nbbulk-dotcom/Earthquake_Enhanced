"""
Correlation Engine Module for Earthquake_Enhanced System
Implements comprehensive multi-resonance overlay analysis system

Features:
1. Multi-Resonance Overlay Analysis (space + strain-rate + custom sources)
2. Resultant Frequency Calculation (wave superposition, interference)
3. Coherence and Amplification Detection (phase alignment, constructive/destructive)
4. Pattern Identification (recurring patterns, temporal evolution)
5. 21-Day Forward Prediction (sun path integration, confidence intervals)
6. Geolocated Point Analysis (single-point and multi-fault triangulation)
7. Resonance Set Tracking (registry, overlay counting, statistics)
8. Data Preparation for Visualization (3D wireframe, color coding, time-series)

Author: BRETT System
Version: 1.0.0
"""

import math
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import hashlib

# Import our engines
from backend.features.space_engine import SpaceEngine
from backend.features.resonance import ResonanceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResonanceSource:
    """Data class for resonance source tracking"""
    source_id: str
    source_name: str
    source_type: str  # 'space', 'strain-rate', 'custom'
    frequency: float  # Hz
    amplitude: float
    phase: float  # radians
    location: Tuple[float, float]  # (lat, lon)
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OverlayRegion:
    """Data class for resonance overlay regions"""
    region_id: str
    location: Tuple[float, float]
    resonance_sources: List[str]  # List of source_ids
    resultant_frequency: float
    resultant_amplitude: float
    resultant_phase: float
    interference_type: str  # 'constructive', 'destructive', 'mixed'
    coherence_coefficient: float
    overlay_count: int
    timestamp: datetime


@dataclass
class ResonancePattern:
    """Data class for identified resonance patterns"""
    pattern_id: str
    pattern_name: str
    frequency_signature: List[float]
    temporal_evolution: List[Dict[str, Any]]
    recurrence_period: Optional[float]  # days
    similarity_score: float
    first_observed: datetime
    last_observed: datetime
    occurrence_count: int


class CorrelationEngine:
    """
    Comprehensive Multi-Resonance Overlay Analysis Engine
    """
    
    def __init__(self):
        """Initialize Correlation Engine"""
        self.version = "CORRELATION-ENGINE-V1.0.0"
        self.engine_id = "CORRELATION_ENGINE"
        
        # Initialize sub-engines
        self.space_engine = SpaceEngine()
        self.resonance_engine = ResonanceEngine()
        
        # Feature 7: Resonance Set Tracking
        self.resonance_registry: Dict[str, ResonanceSource] = {}
        self.overlay_regions: Dict[str, OverlayRegion] = {}
        
        # Feature 4: Pattern Identification
        self.identified_patterns: Dict[str, ResonancePattern] = {}
        self.pattern_history: List[Dict[str, Any]] = []
        
        # Wave superposition constants
        self.CONSTRUCTIVE_THRESHOLD = 0.8  # Coherence for constructive interference
        self.DESTRUCTIVE_THRESHOLD = 0.2  # Coherence for destructive interference
        self.PHASE_ALIGNMENT_TOLERANCE = math.pi / 6  # ±30 degrees
        
        # Pattern matching thresholds
        self.PATTERN_SIMILARITY_THRESHOLD = 0.75
        self.PATTERN_RECURRENCE_MIN_COUNT = 3
        
        # Prediction parameters
        self.PREDICTION_DAYS = 21
        self.CONFIDENCE_DECAY_RATE = 0.05  # per day
        
        self.last_calculation = None
    
    # ========== Feature 1: Multi-Resonance Overlay Analysis ==========
    
    async def integrate_space_resonances(self, latitude: float, longitude: float,
                                        timestamp: Optional[datetime] = None) -> List[ResonanceSource]:
        """
        Integrate space engine resonances (RGB, solar, geomagnetic, ionospheric)
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            timestamp: Analysis timestamp
            
        Returns:
            List of space resonance sources
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Get space engine predictions
        space_prediction = await self.space_engine.calculate_space_prediction(
            latitude, longitude, timestamp
        )
        
        if not space_prediction['success']:
            logger.warning("Space engine prediction failed")
            return []
        
        sources = []
        
        # Extract RGB resonance components
        rgb_data = space_prediction['rgb_resonance']
        
        # Red component (Solar-related)
        sources.append(ResonanceSource(
            source_id=self._generate_source_id('space_rgb_red', timestamp),
            source_name='Space RGB Red Component',
            source_type='space',
            frequency=self._rgb_to_frequency(rgb_data['R_component'], 'R'),
            amplitude=rgb_data['R_component'],
            phase=0.0,
            location=(latitude, longitude),
            timestamp=timestamp,
            metadata={'rgb_component': 'R', 'space_variables': 'solar_related'}
        ))
        
        # Green component (Magnetic-related)
        sources.append(ResonanceSource(
            source_id=self._generate_source_id('space_rgb_green', timestamp),
            source_name='Space RGB Green Component',
            source_type='space',
            frequency=self._rgb_to_frequency(rgb_data['G_component'], 'G'),
            amplitude=rgb_data['G_component'],
            phase=math.pi / 2,  # 90 degree phase shift
            location=(latitude, longitude),
            timestamp=timestamp,
            metadata={'rgb_component': 'G', 'space_variables': 'magnetic_related'}
        ))
        
        # Blue component (Particle-related)
        sources.append(ResonanceSource(
            source_id=self._generate_source_id('space_rgb_blue', timestamp),
            source_name='Space RGB Blue Component',
            source_type='space',
            frequency=self._rgb_to_frequency(rgb_data['B_component'], 'B'),
            amplitude=rgb_data['B_component'],
            phase=math.pi,  # 180 degree phase shift
            location=(latitude, longitude),
            timestamp=timestamp,
            metadata={'rgb_component': 'B', 'space_variables': 'particle_related'}
        ))
        
        # Combined RGB resonance
        sources.append(ResonanceSource(
            source_id=self._generate_source_id('space_rgb_combined', timestamp),
            source_name='Space RGB Combined Resonance',
            source_type='space',
            frequency=space_prediction['resultant_resonance']['resultant_resonance'] * 10,  # Scale to Hz
            amplitude=space_prediction['resultant_resonance']['resultant_resonance'],
            phase=self._calculate_combined_phase(space_prediction),
            location=(latitude, longitude),
            timestamp=timestamp,
            metadata={'source': 'space_engine', 'resultant_type': 'rgb_combined'}
        ))
        
        # Register sources
        for source in sources:
            self.resonance_registry[source.source_id] = source
        
        logger.info(f"Integrated {len(sources)} space resonance sources")
        return sources
    
    def integrate_strain_rate_resonances(self, latitude: float, longitude: float,
                                        depth_km: float = 15.0,
                                        timestamp: Optional[datetime] = None) -> List[ResonanceSource]:
        """
        Integrate strain-rate resonances from resonance engine
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            depth_km: Analysis depth
            timestamp: Analysis timestamp
            
        Returns:
            List of strain-rate resonance sources
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Get resonance engine analysis
        resonance_analysis = self.resonance_engine.calculate_comprehensive_resonance(
            latitude, longitude, depth_km, timestamp
        )
        
        if not resonance_analysis['success']:
            logger.warning("Resonance engine analysis failed")
            return []
        
        sources = []
        crustal = resonance_analysis['crustal_resonance']
        
        # Base crustal resonance
        sources.append(ResonanceSource(
            source_id=self._generate_source_id('strain_crustal_base', timestamp),
            source_name='Crustal Base Resonance',
            source_type='strain-rate',
            frequency=crustal['base_resonance_frequency'],
            amplitude=crustal['resonance_amplitude'],
            phase=0.0,
            location=(latitude, longitude),
            timestamp=timestamp,
            metadata={'depth_km': depth_km, 'resonance_type': 'crustal_base'}
        ))
        
        # Combined crustal resonance (with boundary influence)
        sources.append(ResonanceSource(
            source_id=self._generate_source_id('strain_crustal_combined', timestamp),
            source_name='Crustal Combined Resonance',
            source_type='strain-rate',
            frequency=crustal['combined_resonance_frequency'],
            amplitude=crustal['resonance_amplitude'] * (1 + crustal['boundary_influence_factor']),
            phase=crustal['boundary_influence_factor'] * math.pi / 4,
            location=(latitude, longitude),
            timestamp=timestamp,
            metadata={'depth_km': depth_km, 'resonance_type': 'crustal_combined',
                     'boundary_influence': crustal['boundary_influence_factor']}
        ))
        
        # Harmonic resonances
        for i, harmonic_freq in enumerate(crustal['harmonics'][:3]):  # First 3 harmonics
            sources.append(ResonanceSource(
                source_id=self._generate_source_id(f'strain_harmonic_{i+1}', timestamp),
                source_name=f'Crustal Harmonic {i+1}',
                source_type='strain-rate',
                frequency=harmonic_freq,
                amplitude=crustal['resonance_amplitude'] / (i + 2),  # Decreasing amplitude
                phase=(i + 1) * math.pi / 4,
                location=(latitude, longitude),
                timestamp=timestamp,
                metadata={'depth_km': depth_km, 'harmonic_order': i+1}
            ))
        
        # Register sources
        for source in sources:
            self.resonance_registry[source.source_id] = source
        
        logger.info(f"Integrated {len(sources)} strain-rate resonance sources")
        return sources
    
    def add_custom_resonance_source(self, source_name: str, frequency: float,
                                   amplitude: float, phase: float,
                                   latitude: float, longitude: float,
                                   metadata: Optional[Dict] = None) -> ResonanceSource:
        """
        Add a custom resonance source
        
        Args:
            source_name: Name for the resonance source
            frequency: Frequency in Hz
            amplitude: Amplitude (0-1 normalized)
            phase: Phase in radians
            latitude: Geographic latitude
            longitude: Geographic longitude
            metadata: Optional metadata dictionary
            
        Returns:
            Created ResonanceSource
        """
        timestamp = datetime.utcnow()
        
        source = ResonanceSource(
            source_id=self._generate_source_id(f'custom_{source_name}', timestamp),
            source_name=source_name,
            source_type='custom',
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            location=(latitude, longitude),
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        self.resonance_registry[source.source_id] = source
        logger.info(f"Added custom resonance source: {source_name}")
        
        return source
    
    # ========== Feature 2: Resultant Frequency Calculation ==========
    
    def calculate_wave_superposition(self, sources: List[ResonanceSource],
                                    location: Tuple[float, float],
                                    time_point: float = 0.0) -> Dict[str, Any]:
        """
        Calculate wave superposition using empirical wave mechanics
        
        Wave superposition formula:
        ψ(t) = Σᵢ Aᵢ * cos(2π * fᵢ * t + φᵢ)
        
        Args:
            sources: List of resonance sources to superpose
            location: Location for calculation (lat, lon)
            time_point: Time point for calculation (seconds)
            
        Returns:
            Superposition results with resultant frequency and amplitude
        """
        if not sources:
            return {
                'resultant_frequency': 0.0,
                'resultant_amplitude': 0.0,
                'resultant_phase': 0.0,
                'interference_type': 'none',
                'source_count': 0
            }
        
        # Calculate instantaneous wave values
        wave_values = []
        for source in sources:
            # Distance-based amplitude attenuation
            distance_factor = self._calculate_distance_attenuation(
                source.location, location
            )
            
            # Wave value at time_point
            value = (source.amplitude * distance_factor * 
                    math.cos(2 * math.pi * source.frequency * time_point + source.phase))
            wave_values.append(value)
        
        # Resultant amplitude (sum of instantaneous values)
        resultant_instantaneous = sum(wave_values)
        
        # Calculate average frequency (weighted by amplitude)
        total_amplitude = sum(s.amplitude for s in sources)
        if total_amplitude > 0:
            resultant_frequency = sum(s.frequency * s.amplitude for s in sources) / total_amplitude
        else:
            resultant_frequency = sum(s.frequency for s in sources) / len(sources)
        
        # Calculate resultant amplitude (RMS over period)
        amplitudes = [s.amplitude for s in sources]
        phases = [s.phase for s in sources]
        
        # Vector addition of amplitudes
        real_sum = sum(a * math.cos(p) for a, p in zip(amplitudes, phases))
        imag_sum = sum(a * math.sin(p) for a, p in zip(amplitudes, phases))
        
        resultant_amplitude = math.sqrt(real_sum**2 + imag_sum**2)
        resultant_phase = math.atan2(imag_sum, real_sum)
        
        # Determine interference type
        max_possible_amplitude = sum(amplitudes)
        min_possible_amplitude = max(0, max(amplitudes) - sum(amplitudes[1:]))
        
        if resultant_amplitude > max_possible_amplitude * self.CONSTRUCTIVE_THRESHOLD:
            interference_type = 'constructive'
        elif resultant_amplitude < max_possible_amplitude * self.DESTRUCTIVE_THRESHOLD:
            interference_type = 'destructive'
        else:
            interference_type = 'mixed'
        
        return {
            'resultant_frequency': resultant_frequency,
            'resultant_amplitude': resultant_amplitude,
            'resultant_phase': resultant_phase,
            'resultant_instantaneous': resultant_instantaneous,
            'interference_type': interference_type,
            'source_count': len(sources),
            'max_possible_amplitude': max_possible_amplitude,
            'constructive_ratio': resultant_amplitude / max_possible_amplitude if max_possible_amplitude > 0 else 0
        }
    
    def detect_beat_frequencies(self, sources: List[ResonanceSource]) -> List[Dict[str, Any]]:
        """
        Detect beat frequencies from interfering resonances
        
        Beat frequency: f_beat = |f₁ - f₂|
        
        Args:
            sources: List of resonance sources
            
        Returns:
            List of detected beat frequencies
        """
        beat_frequencies = []
        
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                # Calculate beat frequency
                f_beat = abs(source1.frequency - source2.frequency)
                
                # Beat amplitude (product of individual amplitudes)
                a_beat = 2 * source1.amplitude * source2.amplitude
                
                # Only significant beats
                if f_beat > 0.001 and a_beat > 0.01:
                    beat_frequencies.append({
                        'source1': source1.source_name,
                        'source2': source2.source_name,
                        'beat_frequency': f_beat,
                        'beat_amplitude': a_beat,
                        'beat_period': 1.0 / f_beat if f_beat > 0 else float('inf')
                    })
        
        return sorted(beat_frequencies, key=lambda x: x['beat_amplitude'], reverse=True)
    
    # ========== Feature 3: Coherence and Amplification Detection ==========
    
    def calculate_coherence_coefficient(self, sources: List[ResonanceSource]) -> Dict[str, Any]:
        """
        Calculate coherence coefficient for phase alignment detection
        
        Coherence = |Σᵢ Aᵢ * e^(iφᵢ)| / Σᵢ Aᵢ
        
        Args:
            sources: List of resonance sources
            
        Returns:
            Coherence analysis results
        """
        if not sources:
            return {
                'coherence_coefficient': 0.0,
                'is_coherent': False,
                'phase_alignment_quality': 'none'
            }
        
        # Complex phasor representation
        total_amplitude = sum(s.amplitude for s in sources)
        
        if total_amplitude == 0:
            return {
                'coherence_coefficient': 0.0,
                'is_coherent': False,
                'phase_alignment_quality': 'none'
            }
        
        # Vector sum in complex plane
        complex_sum = sum(s.amplitude * np.exp(1j * s.phase) for s in sources)
        coherence = abs(complex_sum) / total_amplitude
        
        # Phase variance (circular variance)
        mean_phase = np.angle(complex_sum)
        phase_deviations = [abs(self._angular_difference(s.phase, mean_phase)) 
                           for s in sources]
        phase_variance = np.var(phase_deviations)
        
        # Determine alignment quality
        if coherence >= 0.8:
            alignment_quality = 'excellent'
            is_coherent = True
        elif coherence >= 0.6:
            alignment_quality = 'good'
            is_coherent = True
        elif coherence >= 0.4:
            alignment_quality = 'moderate'
            is_coherent = False
        else:
            alignment_quality = 'poor'
            is_coherent = False
        
        return {
            'coherence_coefficient': coherence,
            'is_coherent': is_coherent,
            'phase_alignment_quality': alignment_quality,
            'mean_phase': mean_phase,
            'phase_variance': phase_variance,
            'source_count': len(sources)
        }
    
    def identify_amplification_zones(self, sources: List[ResonanceSource],
                                    grid_resolution: int = 10) -> List[Dict[str, Any]]:
        """
        Identify zones of constructive interference (amplification)
        
        Args:
            sources: List of resonance sources
            grid_resolution: Grid points per dimension
            
        Returns:
            List of amplification zones
        """
        if not sources:
            return []
        
        # Determine spatial extent
        lats = [s.location[0] for s in sources]
        lons = [s.location[1] for s in sources]
        
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        
        # Add margin
        lat_margin = (lat_max - lat_min) * 0.2 or 1.0
        lon_margin = (lon_max - lon_min) * 0.2 or 1.0
        
        amplification_zones = []
        
        # Sample grid
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                lat = lat_min - lat_margin + (lat_max - lat_min + 2*lat_margin) * i / (grid_resolution - 1)
                lon = lon_min - lon_margin + (lon_max - lon_min + 2*lon_margin) * j / (grid_resolution - 1)
                
                # Calculate superposition at this point
                superposition = self.calculate_wave_superposition(
                    sources, (lat, lon), time_point=0.0
                )
                
                # Check if amplification zone
                if superposition['interference_type'] == 'constructive':
                    amplification_zones.append({
                        'location': (lat, lon),
                        'resultant_amplitude': superposition['resultant_amplitude'],
                        'constructive_ratio': superposition['constructive_ratio'],
                        'contributing_sources': len(sources)
                    })
        
        # Sort by amplitude
        amplification_zones.sort(key=lambda x: x['resultant_amplitude'], reverse=True)
        
        logger.info(f"Identified {len(amplification_zones)} amplification zones")
        return amplification_zones
    
    def identify_cancellation_zones(self, sources: List[ResonanceSource],
                                   grid_resolution: int = 10) -> List[Dict[str, Any]]:
        """
        Identify zones of destructive interference (cancellation)
        
        Args:
            sources: List of resonance sources
            grid_resolution: Grid points per dimension
            
        Returns:
            List of cancellation zones
        """
        if not sources:
            return []
        
        # Determine spatial extent
        lats = [s.location[0] for s in sources]
        lons = [s.location[1] for s in sources]
        
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        
        lat_margin = (lat_max - lat_min) * 0.2 or 1.0
        lon_margin = (lon_max - lon_min) * 0.2 or 1.0
        
        cancellation_zones = []
        
        # Sample grid
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                lat = lat_min - lat_margin + (lat_max - lat_min + 2*lat_margin) * i / (grid_resolution - 1)
                lon = lon_min - lon_margin + (lon_max - lon_min + 2*lon_margin) * j / (grid_resolution - 1)
                
                # Calculate superposition at this point
                superposition = self.calculate_wave_superposition(
                    sources, (lat, lon), time_point=0.0
                )
                
                # Check if cancellation zone
                if superposition['interference_type'] == 'destructive':
                    cancellation_zones.append({
                        'location': (lat, lon),
                        'resultant_amplitude': superposition['resultant_amplitude'],
                        'destructive_ratio': 1.0 - superposition['constructive_ratio'],
                        'contributing_sources': len(sources)
                    })
        
        logger.info(f"Identified {len(cancellation_zones)} cancellation zones")
        return cancellation_zones
    
    # ========== Feature 4: Pattern Identification ==========
    
    def identify_recurring_patterns(self, time_window_days: int = 30) -> List[ResonancePattern]:
        """
        Identify recurring resonance patterns across multiple sources
        
        Args:
            time_window_days: Time window for pattern analysis
            
        Returns:
            List of identified patterns
        """
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(days=time_window_days)
        
        # Get recent resonance sources
        recent_sources = [
            s for s in self.resonance_registry.values()
            if s.timestamp >= cutoff_time
        ]
        
        if len(recent_sources) < self.PATTERN_RECURRENCE_MIN_COUNT:
            logger.info("Insufficient data for pattern identification")
            return []
        
        # Extract frequency signatures
        frequency_signatures = self._extract_frequency_signatures(recent_sources)
        
        # Cluster similar signatures
        pattern_clusters = self._cluster_frequency_signatures(frequency_signatures)
        
        # Analyze temporal evolution
        patterns = []
        for cluster_id, cluster_sources in pattern_clusters.items():
            if len(cluster_sources) >= self.PATTERN_RECURRENCE_MIN_COUNT:
                pattern = self._analyze_pattern_cluster(cluster_sources, cluster_id)
                patterns.append(pattern)
                
                # Register pattern
                self.identified_patterns[pattern.pattern_id] = pattern
        
        logger.info(f"Identified {len(patterns)} recurring patterns")
        return patterns
    
    def calculate_pattern_similarity(self, pattern1: ResonancePattern,
                                    pattern2: ResonancePattern) -> float:
        """
        Calculate similarity between two resonance patterns
        
        Uses normalized cross-correlation of frequency signatures
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score (0-1)
        """
        sig1 = np.array(pattern1.frequency_signature)
        sig2 = np.array(pattern2.frequency_signature)
        
        # Normalize
        sig1_norm = sig1 / np.linalg.norm(sig1) if np.linalg.norm(sig1) > 0 else sig1
        sig2_norm = sig2 / np.linalg.norm(sig2) if np.linalg.norm(sig2) > 0 else sig2
        
        # Ensure same length
        min_len = min(len(sig1_norm), len(sig2_norm))
        sig1_norm = sig1_norm[:min_len]
        sig2_norm = sig2_norm[:min_len]
        
        # Calculate correlation
        if len(sig1_norm) == 0:
            return 0.0
        
        similarity = np.dot(sig1_norm, sig2_norm)
        
        return max(0.0, min(1.0, similarity))
    
    def track_pattern_evolution(self, pattern_id: str) -> Dict[str, Any]:
        """
        Track temporal evolution of a specific pattern
        
        Args:
            pattern_id: Pattern identifier
            
        Returns:
            Pattern evolution data
        """
        if pattern_id not in self.identified_patterns:
            return {'error': 'Pattern not found'}
        
        pattern = self.identified_patterns[pattern_id]
        
        # Analyze evolution
        evolution_data = {
            'pattern_id': pattern_id,
            'pattern_name': pattern.pattern_name,
            'first_observed': pattern.first_observed.isoformat(),
            'last_observed': pattern.last_observed.isoformat(),
            'time_span_days': (pattern.last_observed - pattern.first_observed).days,
            'occurrence_count': pattern.occurrence_count,
            'recurrence_period_days': pattern.recurrence_period,
            'temporal_evolution': pattern.temporal_evolution,
            'frequency_trend': self._analyze_frequency_trend(pattern.temporal_evolution)
        }
        
        return evolution_data
    
    # ========== Feature 5: 21-Day Forward Prediction ==========
    
    async def generate_21day_prediction(self, latitude: float, longitude: float,
                                       depth_km: float = 15.0) -> Dict[str, Any]:
        """
        Generate 21-day forward prediction with confidence intervals
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            depth_km: Analysis depth
            
        Returns:
            21-day prediction with confidence intervals
        """
        current_time = datetime.utcnow()
        predictions = []
        
        logger.info(f"Generating 21-day prediction for ({latitude}, {longitude})")
        
        for day in range(self.PREDICTION_DAYS + 1):  # 0 to 21 days
            prediction_date = current_time + timedelta(days=day)
            
            # Get space engine prediction (includes sun path)
            space_pred = await self.space_engine.calculate_space_prediction(
                latitude, longitude, prediction_date
            )
            
            # Get resonance prediction
            resonance_pred = self.resonance_engine.calculate_comprehensive_resonance(
                latitude, longitude, depth_km, prediction_date
            )
            
            # Integrate resonances
            space_sources = await self.integrate_space_resonances(
                latitude, longitude, prediction_date
            )
            
            strain_sources = self.integrate_strain_rate_resonances(
                latitude, longitude, depth_km, prediction_date
            )
            
            all_sources = space_sources + strain_sources
            
            # Calculate overlay
            overlay_analysis = self.calculate_wave_superposition(
                all_sources, (latitude, longitude)
            )
            
            # Calculate coherence
            coherence = self.calculate_coherence_coefficient(all_sources)
            
            # Calculate confidence (decreases with time)
            confidence = self._calculate_prediction_confidence(
                day, space_pred, resonance_pred
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                overlay_analysis, coherence, day
            )
            
            predictions.append({
                'day': day,
                'date': prediction_date.strftime('%Y-%m-%d'),
                'prediction_date': prediction_date.isoformat(),
                'resultant_frequency': overlay_analysis['resultant_frequency'],
                'resultant_amplitude': overlay_analysis['resultant_amplitude'],
                'interference_type': overlay_analysis['interference_type'],
                'coherence_coefficient': coherence['coherence_coefficient'],
                'is_coherent': coherence['is_coherent'],
                'source_count': len(all_sources),
                'confidence': confidence,
                'risk_score': risk_score,
                'risk_level': self._classify_risk_level(risk_score),
                'space_correlation': space_pred.get('earthquake_correlation_score', 0.0),
                'resonance_risk': resonance_pred.get('resonance_risk_score', 0.0)
            })
        
        # Calculate summary statistics
        summary = self._calculate_prediction_summary(predictions)
        
        logger.info(f"Generated {len(predictions)} day predictions")
        
        return {
            'success': True,
            'location': {'latitude': latitude, 'longitude': longitude, 'depth_km': depth_km},
            'prediction_start': current_time.isoformat(),
            'prediction_days': self.PREDICTION_DAYS,
            'daily_predictions': predictions,
            'summary': summary
        }
    
    # ========== Feature 6: Geolocated Point Analysis ==========
    
    async def analyze_single_point(self, latitude: float, longitude: float,
                                   depth_km: float = 15.0) -> Dict[str, Any]:
        """
        Analyze resonances at a single geographic point
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            depth_km: Analysis depth
            
        Returns:
            Single-point resonance analysis
        """
        timestamp = datetime.utcnow()
        
        # Integrate all resonance sources
        space_sources = await self.integrate_space_resonances(latitude, longitude, timestamp)
        strain_sources = self.integrate_strain_rate_resonances(latitude, longitude, depth_km, timestamp)
        
        all_sources = space_sources + strain_sources
        
        # Calculate superposition
        superposition = self.calculate_wave_superposition(all_sources, (latitude, longitude))
        
        # Calculate coherence
        coherence = self.calculate_coherence_coefficient(all_sources)
        
        # Detect beat frequencies
        beats = self.detect_beat_frequencies(all_sources)
        
        # Create overlay region
        region_id = self._generate_region_id(latitude, longitude, timestamp)
        overlay_region = OverlayRegion(
            region_id=region_id,
            location=(latitude, longitude),
            resonance_sources=[s.source_id for s in all_sources],
            resultant_frequency=superposition['resultant_frequency'],
            resultant_amplitude=superposition['resultant_amplitude'],
            resultant_phase=superposition['resultant_phase'],
            interference_type=superposition['interference_type'],
            coherence_coefficient=coherence['coherence_coefficient'],
            overlay_count=len(all_sources),
            timestamp=timestamp
        )
        
        self.overlay_regions[region_id] = overlay_region
        
        return {
            'success': True,
            'location': {'latitude': latitude, 'longitude': longitude, 'depth_km': depth_km},
            'timestamp': timestamp.isoformat(),
            'resonance_sources': len(all_sources),
            'superposition': superposition,
            'coherence': coherence,
            'beat_frequencies': beats[:5],  # Top 5 beats
            'overlay_region': {
                'region_id': region_id,
                'overlay_count': overlay_region.overlay_count,
                'resultant_frequency': overlay_region.resultant_frequency,
                'resultant_amplitude': overlay_region.resultant_amplitude
            }
        }
    
    async def analyze_multi_fault_region(self, center_lat: float, center_lon: float,
                                        triangulation_points: List[Tuple[float, float]],
                                        depth_km: float = 15.0) -> Dict[str, Any]:
        """
        Analyze multi-fault region using triangulation points (e.g., Tokyo region)
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            triangulation_points: List of (lat, lon) triangulation points
            depth_km: Analysis depth
            
        Returns:
            Multi-fault regional analysis
        """
        timestamp = datetime.utcnow()
        
        logger.info(f"Analyzing multi-fault region with {len(triangulation_points)} triangulation points")
        
        # Analyze each triangulation point
        point_analyses = []
        all_sources_combined = []
        
        for point_lat, point_lon in triangulation_points:
            # Integrate resonances at this point
            space_sources = await self.integrate_space_resonances(point_lat, point_lon, timestamp)
            strain_sources = self.integrate_strain_rate_resonances(point_lat, point_lon, depth_km, timestamp)
            
            point_sources = space_sources + strain_sources
            all_sources_combined.extend(point_sources)
            
            # Calculate point analysis
            superposition = self.calculate_wave_superposition(point_sources, (point_lat, point_lon))
            coherence = self.calculate_coherence_coefficient(point_sources)
            
            point_analyses.append({
                'location': (point_lat, point_lon),
                'source_count': len(point_sources),
                'resultant_frequency': superposition['resultant_frequency'],
                'resultant_amplitude': superposition['resultant_amplitude'],
                'coherence': coherence['coherence_coefficient']
            })
        
        # Aggregate regional analysis
        avg_frequency = np.mean([p['resultant_frequency'] for p in point_analyses])
        max_amplitude = max([p['resultant_amplitude'] for p in point_analyses])
        avg_coherence = np.mean([p['coherence'] for p in point_analyses])
        
        # Calculate regional superposition at center
        regional_superposition = self.calculate_wave_superposition(
            all_sources_combined, (center_lat, center_lon)
        )
        
        # Identify amplification zones across region
        amplification_zones = self.identify_amplification_zones(
            all_sources_combined, grid_resolution=15
        )
        
        # Calculate regional risk
        regional_risk = self._calculate_regional_risk(
            point_analyses, amplification_zones
        )
        
        return {
            'success': True,
            'region_center': {'latitude': center_lat, 'longitude': center_lon},
            'triangulation_points': len(triangulation_points),
            'timestamp': timestamp.isoformat(),
            'point_analyses': point_analyses,
            'regional_aggregates': {
                'average_frequency': avg_frequency,
                'maximum_amplitude': max_amplitude,
                'average_coherence': avg_coherence,
                'total_sources': len(all_sources_combined)
            },
            'regional_superposition': regional_superposition,
            'amplification_zones': amplification_zones[:10],  # Top 10 zones
            'regional_risk_score': regional_risk,
            'risk_level': self._classify_risk_level(regional_risk)
        }
    
    # ========== Feature 7: Resonance Set Tracking ==========
    
    def get_resonance_registry_summary(self) -> Dict[str, Any]:
        """Get summary of all active resonance sources"""
        if not self.resonance_registry:
            return {
                'total_sources': 0,
                'by_type': {},
                'frequency_range': (0, 0),
                'amplitude_range': (0, 0)
            }
        
        sources = list(self.resonance_registry.values())
        
        # Count by type
        by_type = defaultdict(int)
        for source in sources:
            by_type[source.source_type] += 1
        
        # Frequency and amplitude ranges
        frequencies = [s.frequency for s in sources]
        amplitudes = [s.amplitude for s in sources]
        
        return {
            'total_sources': len(sources),
            'by_type': dict(by_type),
            'frequency_range': (min(frequencies), max(frequencies)),
            'amplitude_range': (min(amplitudes), max(amplitudes)),
            'oldest_source': min(s.timestamp for s in sources).isoformat(),
            'newest_source': max(s.timestamp for s in sources).isoformat()
        }
    
    def get_overlay_statistics(self, location: Optional[Tuple[float, float]] = None,
                              radius_km: Optional[float] = None) -> Dict[str, Any]:
        """
        Get overlay statistics for location or global
        
        Args:
            location: Optional (lat, lon) to filter by proximity
            radius_km: Radius in km for proximity filter
            
        Returns:
            Overlay statistics
        """
        regions = list(self.overlay_regions.values())
        
        # Filter by location if specified
        if location and radius_km:
            regions = [
                r for r in regions
                if self._calculate_distance(r.location, location) <= radius_km
            ]
        
        if not regions:
            return {
                'total_overlays': 0,
                'max_overlay_count': 0,
                'dominant_frequencies': []
            }
        
        # Calculate statistics
        overlay_counts = [r.overlay_count for r in regions]
        frequencies = [r.resultant_frequency for r in regions]
        amplitudes = [r.resultant_amplitude for r in regions]
        
        # Find dominant frequencies (top 5)
        freq_amplitude_pairs = list(zip(frequencies, amplitudes))
        freq_amplitude_pairs.sort(key=lambda x: x[1], reverse=True)
        dominant_frequencies = [
            {'frequency': f, 'amplitude': a}
            for f, a in freq_amplitude_pairs[:5]
        ]
        
        # Interference type distribution
        interference_dist = defaultdict(int)
        for region in regions:
            interference_dist[region.interference_type] += 1
        
        return {
            'total_overlays': len(regions),
            'max_overlay_count': max(overlay_counts),
            'average_overlay_count': np.mean(overlay_counts),
            'frequency_range': (min(frequencies), max(frequencies)),
            'amplitude_range': (min(amplitudes), max(amplitudes)),
            'dominant_frequencies': dominant_frequencies,
            'interference_distribution': dict(interference_dist),
            'location_filter': location,
            'radius_km': radius_km
        }
    
    def query_overlays_by_criteria(self, min_overlay_count: Optional[int] = None,
                                   min_coherence: Optional[float] = None,
                                   interference_type: Optional[str] = None,
                                   time_window_hours: Optional[int] = None) -> List[OverlayRegion]:
        """
        Query overlay regions by multiple criteria
        
        Args:
            min_overlay_count: Minimum number of overlapping sources
            min_coherence: Minimum coherence coefficient
            interference_type: Filter by interference type
            time_window_hours: Only include regions within time window
            
        Returns:
            Filtered list of overlay regions
        """
        regions = list(self.overlay_regions.values())
        
        # Apply filters
        if min_overlay_count is not None:
            regions = [r for r in regions if r.overlay_count >= min_overlay_count]
        
        if min_coherence is not None:
            regions = [r for r in regions if r.coherence_coefficient >= min_coherence]
        
        if interference_type is not None:
            regions = [r for r in regions if r.interference_type == interference_type]
        
        if time_window_hours is not None:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            regions = [r for r in regions if r.timestamp >= cutoff]
        
        logger.info(f"Query returned {len(regions)} overlay regions")
        return regions
    
    # ========== Feature 8: Data Preparation for Visualization ==========
    
    def prepare_3d_wireframe_data(self, overlay_regions: List[OverlayRegion],
                                 time_steps: int = 24) -> Dict[str, Any]:
        """
        Prepare data for 3D wireframe visualization
        
        Args:
            overlay_regions: List of overlay regions to visualize
            time_steps: Number of time steps for animation
            
        Returns:
            3D wireframe data structure
        """
        if not overlay_regions:
            return {'vertices': [], 'edges': [], 'colors': [], 'values': []}
        
        vertices = []
        colors = []
        values = []
        
        for region in overlay_regions:
            lat, lon = region.location
            
            # Vertex position (lat, lon, amplitude as height)
            vertices.append({
                'x': lon,
                'y': lat,
                'z': region.resultant_amplitude * 100  # Scale for visibility
            })
            
            # Color coding by interference type
            color = self._get_interference_color(region.interference_type)
            colors.append(color)
            
            # Numerical value to display
            values.append({
                'frequency': region.resultant_frequency,
                'amplitude': region.resultant_amplitude,
                'overlay_count': region.overlay_count,
                'coherence': region.coherence_coefficient
            })
        
        # Generate edges (connect nearby points)
        edges = self._generate_wireframe_edges(vertices, max_distance=2.0)
        
        # Generate time series for animation
        time_series = self._generate_time_series_animation(
            overlay_regions, time_steps
        )
        
        return {
            'vertices': vertices,
            'edges': edges,
            'colors': colors,
            'values': values,
            'time_series': time_series,
            'metadata': {
                'region_count': len(overlay_regions),
                'time_steps': time_steps
            }
        }
    
    def prepare_time_series_data(self, location: Tuple[float, float],
                                days: int = 21) -> Dict[str, Any]:
        """
        Prepare time-series data for temporal visualization
        
        Args:
            location: (lat, lon) location
            days: Number of days to include
            
        Returns:
            Time-series data structure
        """
        # Get all sources for this location
        lat, lon = location
        relevant_sources = [
            s for s in self.resonance_registry.values()
            if self._calculate_distance(s.location, location) < 500  # Within 500km
        ]
        
        time_series = {
            'timestamps': [],
            'frequencies': [],
            'amplitudes': [],
            'coherence': [],
            'overlay_counts': []
        }
        
        current_time = datetime.utcnow()
        
        for day in range(days):
            timestamp = current_time - timedelta(days=days-day-1)
            
            # Filter sources by timestamp
            day_sources = [
                s for s in relevant_sources
                if abs((s.timestamp - timestamp).days) <= 1
            ]
            
            if day_sources:
                superposition = self.calculate_wave_superposition(day_sources, location)
                coherence = self.calculate_coherence_coefficient(day_sources)
                
                time_series['timestamps'].append(timestamp.isoformat())
                time_series['frequencies'].append(superposition['resultant_frequency'])
                time_series['amplitudes'].append(superposition['resultant_amplitude'])
                time_series['coherence'].append(coherence['coherence_coefficient'])
                time_series['overlay_counts'].append(len(day_sources))
        
        return time_series
    
    # ========== Helper Methods ==========
    
    def _generate_source_id(self, base_name: str, timestamp: datetime) -> str:
        """Generate unique source ID"""
        identifier = f"{base_name}_{timestamp.isoformat()}"
        return hashlib.md5(identifier.encode()).hexdigest()[:16]
    
    def _generate_region_id(self, lat: float, lon: float, timestamp: datetime) -> str:
        """Generate unique region ID"""
        identifier = f"region_{lat}_{lon}_{timestamp.isoformat()}"
        return hashlib.md5(identifier.encode()).hexdigest()[:16]
    
    def _rgb_to_frequency(self, rgb_value: float, component: str) -> float:
        """Convert RGB component value to frequency"""
        base_frequencies = {'R': 7.83, 'G': 14.3, 'B': 20.8}  # Schumann harmonics
        return base_frequencies.get(component, 7.83) * (1 + rgb_value)
    
    def _calculate_combined_phase(self, space_prediction: Dict) -> float:
        """Calculate combined phase from space prediction"""
        rgb = space_prediction['rgb_resonance']
        phase = math.atan2(rgb['G_component'] - rgb['B_component'],
                          rgb['R_component'])
        return phase
    
    def _calculate_distance_attenuation(self, source_loc: Tuple[float, float],
                                       target_loc: Tuple[float, float]) -> float:
        """Calculate amplitude attenuation based on distance"""
        distance = self._calculate_distance(source_loc, target_loc)
        # Exponential decay with 500km characteristic length
        attenuation = math.exp(-distance / 500.0)
        return attenuation
    
    def _calculate_distance(self, loc1: Tuple[float, float],
                           loc2: Tuple[float, float]) -> float:
        """Calculate great circle distance in km"""
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        a = (math.sin((lat2_rad - lat1_rad) / 2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon_rad / 2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return 6371.0 * c  # Earth radius in km
    
    def _angular_difference(self, angle1: float, angle2: float) -> float:
        """Calculate smallest angular difference between two angles"""
        diff = angle1 - angle2
        return math.atan2(math.sin(diff), math.cos(diff))
    
    def _extract_frequency_signatures(self, sources: List[ResonanceSource]) -> Dict[str, List[float]]:
        """Extract frequency signatures from sources"""
        signatures = {}
        
        # Group by time window (e.g., daily)
        time_groups = defaultdict(list)
        for source in sources:
            day_key = source.timestamp.strftime('%Y-%m-%d')
            time_groups[day_key].append(source.frequency)
        
        # Create signatures
        for day_key, frequencies in time_groups.items():
            signatures[day_key] = sorted(frequencies)
        
        return signatures
    
    def _cluster_frequency_signatures(self, signatures: Dict[str, List[float]]) -> Dict[int, List]:
        """Cluster similar frequency signatures"""
        # Simple clustering based on signature similarity
        clusters = defaultdict(list)
        cluster_id = 0
        processed = set()
        
        sig_items = list(signatures.items())
        
        for i, (key1, sig1) in enumerate(sig_items):
            if key1 in processed:
                continue
            
            clusters[cluster_id].append((key1, sig1))
            processed.add(key1)
            
            # Find similar signatures
            for key2, sig2 in sig_items[i+1:]:
                if key2 in processed:
                    continue
                
                similarity = self._signature_similarity(sig1, sig2)
                if similarity >= self.PATTERN_SIMILARITY_THRESHOLD:
                    clusters[cluster_id].append((key2, sig2))
                    processed.add(key2)
            
            cluster_id += 1
        
        return clusters
    
    def _signature_similarity(self, sig1: List[float], sig2: List[float]) -> float:
        """Calculate similarity between two frequency signatures"""
        if not sig1 or not sig2:
            return 0.0
        
        # Use normalized correlation
        s1 = np.array(sig1[:min(len(sig1), len(sig2))])
        s2 = np.array(sig2[:min(len(sig1), len(sig2))])
        
        if len(s1) == 0:
            return 0.0
        
        s1_norm = s1 / np.linalg.norm(s1) if np.linalg.norm(s1) > 0 else s1
        s2_norm = s2 / np.linalg.norm(s2) if np.linalg.norm(s2) > 0 else s2
        
        return float(np.dot(s1_norm, s2_norm))
    
    def _analyze_pattern_cluster(self, cluster_sources: List, cluster_id: int) -> ResonancePattern:
        """Analyze a cluster of similar signatures to create a pattern"""
        timestamps = [datetime.fromisoformat(key) if isinstance(key, str) 
                     else key for key, _ in cluster_sources]
        signatures = [sig for _, sig in cluster_sources]
        
        # Average signature
        avg_signature = np.mean(signatures, axis=0).tolist()
        
        # Calculate recurrence period
        if len(timestamps) >= 2:
            time_diffs = [(timestamps[i+1] - timestamps[i]).days 
                         for i in range(len(timestamps)-1)]
            recurrence_period = np.mean(time_diffs) if time_diffs else None
        else:
            recurrence_period = None
        
        # Temporal evolution
        evolution = [
            {
                'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                'signature': sig
            }
            for ts, (_, sig) in zip(timestamps, cluster_sources)
        ]
        
        pattern = ResonancePattern(
            pattern_id=f"pattern_{cluster_id}",
            pattern_name=f"Pattern {cluster_id}",
            frequency_signature=avg_signature,
            temporal_evolution=evolution,
            recurrence_period=recurrence_period,
            similarity_score=1.0,  # Average within-cluster similarity
            first_observed=min(timestamps),
            last_observed=max(timestamps),
            occurrence_count=len(cluster_sources)
        )
        
        return pattern
    
    def _analyze_frequency_trend(self, evolution: List[Dict]) -> Dict[str, Any]:
        """Analyze frequency trend over time"""
        if len(evolution) < 2:
            return {'trend': 'insufficient_data'}
        
        # Extract average frequencies over time
        avg_freqs = [np.mean(e['signature']) for e in evolution]
        
        # Linear regression for trend
        x = np.arange(len(avg_freqs))
        coeffs = np.polyfit(x, avg_freqs, 1)
        slope = coeffs[0]
        
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'frequency_range': (min(avg_freqs), max(avg_freqs))
        }
    
    def _calculate_prediction_confidence(self, day: int, space_pred: Dict,
                                        resonance_pred: Dict) -> float:
        """Calculate confidence for prediction at given day"""
        # Base confidence decreases with time
        base_confidence = math.exp(-day * self.CONFIDENCE_DECAY_RATE)
        
        # Boost from data availability
        data_boost = 1.0
        if space_pred.get('success'):
            data_boost *= 1.1
        if resonance_pred.get('success'):
            data_boost *= 1.1
        
        confidence = min(1.0, base_confidence * data_boost)
        return confidence
    
    def _calculate_risk_score(self, overlay: Dict, coherence: Dict, day: int) -> float:
        """Calculate earthquake risk score"""
        # Amplitude factor
        amp_factor = min(overlay['resultant_amplitude'], 1.0)
        
        # Coherence factor
        coh_factor = coherence['coherence_coefficient']
        
        # Interference type factor
        interference_factors = {
            'constructive': 1.0,
            'mixed': 0.6,
            'destructive': 0.2
        }
        int_factor = interference_factors.get(overlay['interference_type'], 0.5)
        
        # Time decay (near-term predictions more reliable)
        time_factor = math.exp(-day * 0.02)
        
        # Combined risk
        risk = (amp_factor * 0.4 + 
                coh_factor * 0.3 + 
                int_factor * 0.2 + 
                time_factor * 0.1)
        
        return min(1.0, max(0.0, risk))
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level from score"""
        if risk_score >= 0.8:
            return 'CRITICAL'
        elif risk_score >= 0.6:
            return 'HIGH'
        elif risk_score >= 0.4:
            return 'ELEVATED'
        elif risk_score >= 0.2:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _calculate_prediction_summary(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for predictions"""
        risk_scores = [p['risk_score'] for p in predictions]
        amplitudes = [p['resultant_amplitude'] for p in predictions]
        
        # Find peak risk day
        max_risk_pred = max(predictions, key=lambda x: x['risk_score'])
        
        # Count high-risk days
        high_risk_days = sum(1 for p in predictions if p['risk_level'] in ['HIGH', 'CRITICAL'])
        
        return {
            'max_risk_score': max(risk_scores),
            'average_risk_score': np.mean(risk_scores),
            'peak_risk_day': max_risk_pred['day'],
            'peak_risk_date': max_risk_pred['date'],
            'peak_risk_level': max_risk_pred['risk_level'],
            'high_risk_days': high_risk_days,
            'max_amplitude': max(amplitudes),
            'average_amplitude': np.mean(amplitudes)
        }
    
    def _calculate_regional_risk(self, point_analyses: List[Dict],
                                amplification_zones: List[Dict]) -> float:
        """Calculate regional risk score"""
        # Average amplitude across points
        avg_amp = np.mean([p['resultant_amplitude'] for p in point_analyses])
        
        # Maximum coherence
        max_coh = max([p['coherence'] for p in point_analyses])
        
        # Amplification zone factor
        zone_factor = min(len(amplification_zones) / 10.0, 1.0)
        
        # Regional risk
        risk = (avg_amp * 0.4 + max_coh * 0.3 + zone_factor * 0.3)
        
        return min(1.0, max(0.0, risk))
    
    def _get_interference_color(self, interference_type: str) -> str:
        """Get color code for interference type"""
        colors = {
            'constructive': '#FF0000',  # Red
            'destructive': '#0000FF',   # Blue
            'mixed': '#FFFF00'          # Yellow
        }
        return colors.get(interference_type, '#808080')  # Gray default
    
    def _generate_wireframe_edges(self, vertices: List[Dict],
                                 max_distance: float = 2.0) -> List[Tuple[int, int]]:
        """Generate edges for wireframe visualization"""
        edges = []
        
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                v1 = vertices[i]
                v2 = vertices[j]
                
                # Calculate 2D distance
                distance = math.sqrt((v1['x'] - v2['x'])**2 + (v1['y'] - v2['y'])**2)
                
                if distance <= max_distance:
                    edges.append((i, j))
        
        return edges
    
    def _generate_time_series_animation(self, regions: List[OverlayRegion],
                                       time_steps: int) -> List[Dict]:
        """Generate time-series data for animation"""
        time_series = []
        
        for step in range(time_steps):
            # Simulate time evolution (phase changes)
            time_point = step / time_steps
            
            frame_data = []
            for region in regions:
                # Calculate wave at this time point
                amplitude = (region.resultant_amplitude * 
                           math.cos(2 * math.pi * region.resultant_frequency * time_point + 
                                  region.resultant_phase))
                
                frame_data.append({
                    'location': region.location,
                    'amplitude': amplitude,
                    'phase': region.resultant_phase + 2 * math.pi * time_point
                })
            
            time_series.append({
                'step': step,
                'time_point': time_point,
                'data': frame_data
            })
        
        return time_series
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'version': self.version,
            'engine_id': self.engine_id,
            'status': 'operational',
            'resonance_sources_tracked': len(self.resonance_registry),
            'overlay_regions_tracked': len(self.overlay_regions),
            'patterns_identified': len(self.identified_patterns),
            'features': [
                'multi_resonance_overlay_analysis',
                'resultant_frequency_calculation',
                'coherence_amplification_detection',
                'pattern_identification',
                '21day_forward_prediction',
                'geolocated_point_analysis',
                'resonance_set_tracking',
                'visualization_data_preparation'
            ],
            'sub_engines': {
                'space_engine': self.space_engine.version,
                'resonance_engine': self.resonance_engine.version
            }
        }


# Convenience functions
async def analyze_earthquake_correlation(latitude: float, longitude: float,
                                       depth_km: float = 15.0) -> Dict:
    """Convenience function for comprehensive correlation analysis"""
    engine = CorrelationEngine()
    
    # Single point analysis
    point_analysis = await engine.analyze_single_point(latitude, longitude, depth_km)
    
    # 21-day prediction
    prediction = await engine.generate_21day_prediction(latitude, longitude, depth_km)
    
    return {
        'point_analysis': point_analysis,
        'prediction_21day': prediction
    }


def get_correlation_engine() -> CorrelationEngine:
    """Get a new CorrelationEngine instance"""
    return CorrelationEngine()
