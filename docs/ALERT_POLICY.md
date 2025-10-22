# Alert Policy

## Overview

This document defines the alert policy for the Earthquake Enhanced deterministic measurement system. Alerts are generated when measured physical quantities exceed pre-defined thresholds based on published seismological research.

## Alert Types

### 1. Elevated Strain Rate Alert

**Trigger Condition:**
- Calculated strain rate exceeds 1.0×10⁻⁷ per year

**Physical Basis:**
- Based on Kostrov (1974) strain rate formula: ε̇ = ΣM₀ / (2μVT)
- Threshold represents elevated tectonic stress accumulation
- Values above this indicate increased seismic activity

**Severity Levels:**
- **WARNING**: 1.0×10⁻⁷ to 5.0×10⁻⁷ per year
- **CRITICAL**: > 5.0×10⁻⁷ per year

**Response:**
- WARNING: Monitor region closely, review historical patterns
- CRITICAL: Immediate attention, cross-reference with fault proximity

### 2. Fault Proximity Alert

**Trigger Condition:**
- Earthquake within 10 km of known active fault
- Combined with elevated strain rate

**Physical Basis:**
- Proximity to faults increases likelihood of stress transfer
- Historical data shows clustering near fault zones

**Severity Levels:**
- **INFO**: Within 10-20 km of fault
- **WARNING**: Within 5-10 km with elevated strain
- **CRITICAL**: Within 5 km with critical strain rate

### 3. Seismic Moment Accumulation Alert

**Trigger Condition:**
- Cumulative seismic moment exceeds regional threshold
- Threshold varies by region based on historical data

**Physical Basis:**
- Large cumulative moment indicates significant energy release
- May indicate stress redistribution in region

**Regional Thresholds:**
- California: 1.0×10¹⁹ N⋅m per year
- Japan: 5.0×10¹⁹ N⋅m per year
- Other regions: 5.0×10¹⁸ N⋅m per year

## Alert Generation Process

1. **Data Collection**: Fetch earthquake data with full provenance
2. **Physics Calculation**: Calculate all physical quantities deterministically
3. **Threshold Comparison**: Compare against defined thresholds
4. **Alert Creation**: Generate alert with metadata
5. **Provenance Tracking**: Record all parameters and checksums

## Alert Metadata

Each alert includes:
- Alert type and severity
- Measured value and threshold
- Timestamp (UTC)
- Geographic region
- Calculation provenance (checksums, parameters)
- Reference to source data

## Important Notes

### What This System Does NOT Do

- **Does NOT predict earthquakes**: This system measures current stress, it does not forecast future events
- **Does NOT use probabilistic models**: All calculations are deterministic physics
- **Does NOT generate synthetic data**: Only real USGS data is used

### What Alerts Mean

- Alerts indicate **elevated measured stress**, not earthquake predictions
- Alerts are based on **empirical observations** of physical quantities
- Alerts should be interpreted in context with other seismological data

## Threshold Justification

### Strain Rate Threshold (1.0×10⁻⁷ /year)

**References:**
- Kostrov, V. V. (1974). Seismic moment and energy of earthquakes
- Molnar, P. (1979). Earthquake recurrence intervals and plate tectonics
- Ward, S. N. (1998). On the consistency of earthquake moment rates

**Rationale:**
- Represents upper bound of normal tectonic strain accumulation
- Values above indicate anomalous stress buildup
- Validated against historical seismic sequences

### Fault Proximity Threshold (10 km)

**References:**
- Stein, R. S. (1999). The role of stress transfer in earthquake occurrence
- King, G. C. P., & Cocco, M. (2001). Fault interaction by elastic stress changes

**Rationale:**
- Stress transfer effects significant within 10 km
- Historical clustering observed at this distance
- Conservative threshold for monitoring

## Review and Updates

This alert policy is reviewed:
- Annually for threshold appropriateness
- After significant seismic events
- When new research provides updated understanding

**Last Updated**: 2025-10-22
**Next Review**: 2026-10-22
