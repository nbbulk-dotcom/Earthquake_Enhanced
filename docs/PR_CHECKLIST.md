# Pull Request Checklist

## Overview

This checklist ensures all PRs maintain the deterministic, physics-based nature of the system.

## Required Checks

### 1. Code Quality

- [ ] All new code follows deterministic principles
- [ ] No probabilistic models or ML classifiers added
- [ ] No synthetic data generation
- [ ] Physical constants properly documented with references
- [ ] Code is well-commented with physics explanations

### 2. Testing

- [ ] Unit tests added for all new calculations
- [ ] Tests verify deterministic behavior
- [ ] Tests include known values from literature
- [ ] All existing tests pass
- [ ] Test coverage maintained or improved

### 3. Provenance

- [ ] All data operations include checksum generation
- [ ] Metadata tracking implemented
- [ ] Timestamps recorded in UTC
- [ ] Source attribution included
- [ ] Version control maintained

### 4. Documentation

- [ ] README updated if needed
- [ ] Docstrings added to all functions
- [ ] Physics formulas documented with references
- [ ] Alert policies updated if thresholds changed
- [ ] Reproducibility guide updated if needed

### 5. Physics Validation

- [ ] Formulas verified against published papers
- [ ] Physical constants checked (μ = 3.0×10¹⁰ Pa, etc.)
- [ ] Units clearly specified and consistent
- [ ] Calculations produce physically reasonable results
- [ ] Edge cases handled appropriately

### 6. Configuration

- [ ] Configuration changes documented
- [ ] Thresholds justified with references
- [ ] Regional parameters validated
- [ ] Backward compatibility maintained

## Prohibited Changes

The following changes will be **rejected**:

❌ Adding probabilistic prediction models
❌ Introducing ML/AI classifiers
❌ Generating synthetic earthquake data
❌ Removing provenance tracking
❌ Bypassing checksum verification
❌ Using non-deterministic algorithms

## Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least one reviewer
3. **Physics Review**: Verify formulas and constants
4. **Testing Review**: Verify test coverage
5. **Documentation Review**: Ensure completeness

## Example Good PR

**Title**: "Add fault proximity calculation with R-tree indexing"

**Description**:
- Implements spatial indexing for fault lines
- Uses R-tree for efficient nearest-neighbor queries
- Calculates geodesic distances using pyproj
- Fully deterministic, no approximations
- Includes unit tests with known distances
- Documents algorithm and complexity

**Checklist**: All items checked ✓

## Example Bad PR

**Title**: "Add ML model to predict earthquakes"

**Why Rejected**:
- Violates core principle: no probabilistic predictions
- Introduces non-deterministic behavior
- Not based on physics calculations
- Would require synthetic data

## Questions?

If unsure whether a change is appropriate:
1. Review system principles in README
2. Check ALERT_POLICY.md for threshold guidance
3. Consult REPRODUCIBILITY.md for provenance requirements
4. Open a discussion issue before implementing

## References

- System Architecture: README.md
- Alert Policy: docs/ALERT_POLICY.md
- Reproducibility: docs/REPRODUCIBILITY.md

**Last Updated**: 2025-10-22
