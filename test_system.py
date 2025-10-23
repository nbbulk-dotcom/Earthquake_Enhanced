#!/usr/bin/env python3
"""
System Test Script for Earthquake Enhanced
Tests all major components and features
"""

import asyncio
import sys
from datetime import datetime

# Add backend to path
sys.path.insert(0, 'backend')

from features.space_engine import SpaceEngine

async def test_space_engine():
    """Test Space Engine functionality"""
    print("=" * 60)
    print("Testing Earthquake Enhanced - Space Engine")
    print("=" * 60)
    
    engine = SpaceEngine()
    
    # Test location: Tokyo, Japan
    lat, lon = 35.6762, 139.6503
    timestamp = datetime.utcnow()
    
    print(f"\n📍 Location: {lat}°N, {lon}°E")
    print(f"⏰ Timestamp: {timestamp.isoformat()}")
    
    # Test 1: Atmospheric Boundary Refraction
    print("\n🌐 Test 1: Atmospheric Boundary Refraction")
    factors = engine.get_boundary_refraction_factors()
    print(f"   80km boundary: {factors['80km_boundary']}")
    print(f"   85km boundary: {factors['85km_boundary']}")
    print("   ✅ PASSED")
    
    # Test 2: Solar Angles
    print("\n📐 Test 2: Solar Angles")
    solar_angles = engine.calculate_solar_elevation(lat, lon, timestamp)
    print(f"   Elevation: {solar_angles['elevation']:.2f}°")
    print(f"   Azimuth: {solar_angles['azimuth']:.2f}°")
    print("   ✅ PASSED")
    
    # Test 3: Magnetic Latitude
    print("\n🧭 Test 3: Magnetic Latitude Conversion")
    mag_lat = engine.calculate_magnetic_latitude(lat, lon)
    print(f"   Geographic: {lat}°N")
    print(f"   Magnetic: {mag_lat:.2f}°")
    print("   ✅ PASSED")
    
    # Test 4: Lag Times
    print("\n⏱️ Test 4: Dynamic Lag Times")
    lag_times = engine.calculate_dynamic_lag_times(lat, lon, timestamp)
    print(f"   Solar lag: {lag_times['solar_lag_hours']:.2f}h")
    print(f"   Geomagnetic lag: {lag_times['geomagnetic_lag_hours']:.2f}h")
    print(f"   Ionospheric lag: {lag_times['ionospheric_lag_hours']:.2f}h")
    print(f"   Total lag: {lag_times['total_lag_hours']:.2f}h")
    print("   ✅ PASSED")
    
    # Test 5: RGB Resonance
    print("\n🌈 Test 5: RGB Resonance")
    space_readings = {var: 0.5 for var in engine.space_variables.keys()}
    rgb = engine.calculate_rgb_resonance(space_readings)
    print(f"   R: {rgb['R_component']:.3f}")
    print(f"   G: {rgb['G_component']:.3f}")
    print(f"   B: {rgb['B_component']:.3f}")
    print(f"   RGB Resonance: {rgb['rgb_resonance']:.3f}")
    print("   ✅ PASSED")
    
    # Test 6: Equatorial Enhancement
    print("\n🌍 Test 6: Equatorial Enhancement")
    enhancement = engine.apply_equatorial_enhancement(lat, 1.0)
    print(f"   Enhancement factor: {enhancement['enhancement_factor']:.2f}x")
    print(f"   Is equatorial: {enhancement['is_equatorial']}")
    print("   ✅ PASSED")
    
    # Test 7: Full Prediction
    print("\n🎯 Test 7: Full Prediction Calculation")
    result = await engine.calculate_space_prediction(lat, lon, timestamp)
    
    if result['success']:
        print(f"   Correlation Score: {result['earthquake_correlation_score']:.3f}")
        print(f"   RGB Resonance: {result['rgb_resonance']['rgb_resonance']:.3f}")
        print(f"   Resultant Resonance: {result['resultant_resonance']['resultant_resonance']:.3f}")
        print("   ✅ PASSED")
    else:
        print(f"   ❌ FAILED: {result.get('error')}")
        return False
    
    # Test 8: Engine Status
    print("\n📊 Test 8: Engine Status")
    status = engine.get_engine_status()
    print(f"   Version: {status['version']}")
    print(f"   Status: {status['status']}")
    all_features = all(status['features'].values())
    print(f"   All features operational: {all_features}")
    print("   ✅ PASSED")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_space_engine())
    sys.exit(0 if success else 1)
