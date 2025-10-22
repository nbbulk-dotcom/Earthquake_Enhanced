"""
Seismic Moment Calculator

Implements deterministic physics calculations for seismic moment:
M₀ = μ × A × D

Based on Hanks & Kanamori (1979) moment magnitude scale.
"""

import numpy as np
from typing import Union, List


class SeismicMomentCalculator:
    """Calculate seismic moment from magnitude using established physics."""
    
    # Physical constants
    RIGIDITY_MU = 3.0e10  # Pa (3.0×10¹⁰ Pa for crustal rocks)
    
    @staticmethod
    def magnitude_to_moment(magnitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert moment magnitude to seismic moment.
        
        Formula: M₀ = 10^(1.5 × Mw + 9.1) Newton-meters
        
        Reference: Hanks, T. C., & Kanamori, H. (1979). 
        A moment magnitude scale. Journal of Geophysical Research, 84(B5), 2348-2350.
        
        Args:
            magnitude: Moment magnitude (Mw) or array of magnitudes
        
        Returns:
            Seismic moment in Newton-meters (N⋅m)
        """
        return 10 ** (1.5 * magnitude + 9.1)
    
    @staticmethod
    def moment_to_magnitude(moment: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert seismic moment to moment magnitude.
        
        Formula: Mw = (2/3) × (log₁₀(M₀) - 9.1)
        
        Args:
            moment: Seismic moment in Newton-meters (N⋅m)
        
        Returns:
            Moment magnitude (Mw)
        """
        return (2.0 / 3.0) * (np.log10(moment) - 9.1)
    
    @staticmethod
    def rupture_area_from_magnitude(magnitude: float) -> float:
        """
        Estimate rupture area from magnitude.
        
        Empirical relationship: log₁₀(A) = Mw - 4.0
        where A is in km²
        
        Args:
            magnitude: Moment magnitude (Mw)
        
        Returns:
            Rupture area in square meters (m²)
        """
        area_km2 = 10 ** (magnitude - 4.0)
        return area_km2 * 1e6  # Convert km² to m²
    
    @staticmethod
    def average_slip_from_moment(moment: float, area: float, rigidity: float = RIGIDITY_MU) -> float:
        """
        Calculate average slip from seismic moment.
        
        Formula: D = M₀ / (μ × A)
        
        Args:
            moment: Seismic moment (N⋅m)
            area: Rupture area (m²)
            rigidity: Shear modulus (Pa), default 3.0×10¹⁰ Pa
        
        Returns:
            Average slip in meters
        """
        return moment / (rigidity * area)
    
    @classmethod
    def calculate_all_parameters(cls, magnitude: float) -> dict:
        """
        Calculate all seismic parameters from magnitude.
        
        Args:
            magnitude: Moment magnitude (Mw)
        
        Returns:
            Dictionary with all calculated parameters
        """
        moment = cls.magnitude_to_moment(magnitude)
        area = cls.rupture_area_from_magnitude(magnitude)
        slip = cls.average_slip_from_moment(moment, area)
        
        return {
            "magnitude": magnitude,
            "seismic_moment_Nm": moment,
            "rupture_area_m2": area,
            "rupture_area_km2": area / 1e6,
            "average_slip_m": slip,
            "rigidity_Pa": cls.RIGIDITY_MU,
        }


def calculate_cumulative_moment(magnitudes: List[float]) -> float:
    """
    Calculate cumulative seismic moment from multiple events.
    
    Args:
        magnitudes: List of moment magnitudes
    
    Returns:
        Total cumulative seismic moment (N⋅m)
    """
    calc = SeismicMomentCalculator()
    moments = [calc.magnitude_to_moment(m) for m in magnitudes]
    return sum(moments)


if __name__ == "__main__":
    # Example calculations
    calc = SeismicMomentCalculator()
    
    print("Seismic Moment Calculator - Example Calculations")
    print("=" * 60)
    
    # Example 1: Single magnitude
    mag = 6.5
    params = calc.calculate_all_parameters(mag)
    print(f"\nMagnitude {mag}:")
    print(f"  Seismic Moment: {params['seismic_moment_Nm']:.2e} N⋅m")
    print(f"  Rupture Area: {params['rupture_area_km2']:.2f} km²")
    print(f"  Average Slip: {params['average_slip_m']:.3f} m")
    
    # Example 2: Multiple events
    mags = [5.0, 5.5, 6.0, 6.5]
    total_moment = calculate_cumulative_moment(mags)
    equivalent_mag = calc.moment_to_magnitude(total_moment)
    print(f"\nCumulative moment from {len(mags)} events:")
    print(f"  Magnitudes: {mags}")
    print(f"  Total Moment: {total_moment:.2e} N⋅m")
    print(f"  Equivalent Magnitude: {equivalent_mag:.2f}")
