
// Earthquake Enhanced - Space Engine Frontend JavaScript

const API_BASE_URL = 'http://localhost:8000/api/v1';

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const latitude = parseFloat(document.getElementById('latitude').value);
    const longitude = parseFloat(document.getElementById('longitude').value);
    const timestampInput = document.getElementById('timestamp').value;
    
    // Show loading state
    showLoading();
    
    try {
        // Prepare request data
        const requestData = {
            latitude: latitude,
            longitude: longitude,
            timestamp: timestampInput || null,
            include_historical: false
        };
        
        // Call prediction API
        const response = await fetch(`${API_BASE_URL}/prediction`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display results
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to calculate prediction. Please check if the API server is running.');
    }
});

function showLoading() {
    const resultsPanel = document.getElementById('resultsPanel');
    resultsPanel.style.display = 'block';
    resultsPanel.innerHTML = '<div class="loading"></div><p>Calculating prediction...</p>';
}

function showError(message) {
    const resultsPanel = document.getElementById('resultsPanel');
    resultsPanel.style.display = 'block';
    resultsPanel.innerHTML = `
        <div class="result-box" style="border-color: var(--danger-color);">
            <h3>❌ Error</h3>
            <p>${message}</p>
        </div>
    `;
}

function displayResults(data) {
    // Show results panel
    document.getElementById('resultsPanel').style.display = 'block';
    document.getElementById('sunPathPanel').style.display = 'block';
    
    // Scroll to results
    document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth' });
    
    // Earthquake Correlation Score
    const score = data.earthquake_correlation_score || 0;
    document.getElementById('correlationScore').textContent = score.toFixed(3);
    document.getElementById('correlationBar').style.width = `${score * 100}%`;
    
    // Solar Angles
    if (data.solar_angles) {
        document.getElementById('solarElevation').textContent = 
            data.solar_angles.elevation.toFixed(2);
        document.getElementById('solarAzimuth').textContent = 
            data.solar_angles.azimuth.toFixed(2);
        document.getElementById('solarZenith').textContent = 
            data.solar_angles.zenith.toFixed(2);
        document.getElementById('solarDeclination').textContent = 
            data.solar_angles.declination.toFixed(2);
    }
    
    // RGB Resonance
    if (data.rgb_resonance) {
        const r = data.rgb_resonance.R_component;
        const g = data.rgb_resonance.G_component;
        const b = data.rgb_resonance.B_component;
        const rgb = data.rgb_resonance.rgb_resonance;
        
        document.getElementById('rValue').textContent = r.toFixed(3);
        document.getElementById('gValue').textContent = g.toFixed(3);
        document.getElementById('bValue').textContent = b.toFixed(3);
        document.getElementById('rgbResonance').textContent = rgb.toFixed(3);
        
        document.getElementById('rBar').style.height = `${r * 100}%`;
        document.getElementById('gBar').style.height = `${g * 100}%`;
        document.getElementById('bBar').style.height = `${b * 100}%`;
    }
    
    // Lag Times
    if (data.lag_times) {
        document.getElementById('lightTravel').textContent = 
            data.lag_times.light_travel_adjusted_hours.toFixed(2);
        document.getElementById('solarLag').textContent = 
            data.lag_times.solar_lag_hours.toFixed(2);
        document.getElementById('geomagLag').textContent = 
            data.lag_times.geomagnetic_lag_hours.toFixed(2);
        document.getElementById('ionoLag').textContent = 
            data.lag_times.ionospheric_lag_hours.toFixed(2);
        document.getElementById('totalLag').textContent = 
            data.lag_times.total_lag_hours.toFixed(2);
    }
    
    // Location Details
    if (data.location) {
        document.getElementById('magLatitude').textContent = 
            data.location.magnetic_latitude.toFixed(2);
    }
    
    if (data.tetrahedral_angle) {
        document.getElementById('tetraSeismic').textContent = 
            data.tetrahedral_angle.toFixed(2);
    }
    
    if (data.equatorial_enhancement) {
        document.getElementById('equatorialFactor').textContent = 
            data.equatorial_enhancement.enhancement_factor.toFixed(2);
    }
    
    if (data.corrected_rgb_resonance) {
        document.getElementById('atmCorrection').textContent = 
            data.corrected_rgb_resonance.toFixed(3);
    }
    
    // Resultant Resonance
    if (data.resultant_resonance) {
        document.getElementById('resultantResonance').textContent = 
            data.resultant_resonance.resultant_resonance.toFixed(3);
        document.getElementById('matrixContribution').textContent = 
            data.resultant_resonance.matrix_contribution.toFixed(3);
        document.getElementById('dominantEigen').textContent = 
            data.resultant_resonance.dominant_eigenvalue.toFixed(3);
    }
    
    // Space Data Status
    if (data.space_data_integration) {
        const statusBox = document.getElementById('spaceDataStatus');
        if (data.space_data_integration.data_available) {
            statusBox.innerHTML = `
                <p>✅ <strong>Space Data Available</strong></p>
                <p>Combined Reliability: ${(data.space_data_integration.combined_reliability * 100).toFixed(0)}%</p>
                <p>Sources: NASA OMNI2, NOAA SWPC</p>
            `;
            statusBox.style.borderColor = 'var(--success-color)';
        } else {
            statusBox.innerHTML = `
                <p>⚠️ <strong>Using Historical Baseline</strong></p>
                <p>${data.space_data_integration.error}</p>
                <p>Fallback: ${data.space_data_integration.fallback}</p>
            `;
            statusBox.style.borderColor = 'var(--warning-color)';
        }
    }
    
    // Sun Path Visualization
    if (data.sun_path_24h) {
        displaySunPath(data.sun_path_24h);
    }
}

function displaySunPath(sunPath) {
    const container = document.getElementById('sunPathChart');
    
    // Create simple text visualization (in production, use Chart.js or similar)
    let html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse;">';
    html += '<thead><tr><th>Hour</th><th>Elevation</th><th>Azimuth</th><th>Daytime</th></tr></thead>';
    html += '<tbody>';
    
    sunPath.forEach((point, index) => {
        if (index % 4 === 0) {  // Show every 4 hours
            const rowColor = point.is_daytime ? 'rgba(255, 215, 0, 0.1)' : 'rgba(0, 0, 139, 0.1)';
            html += `
                <tr style="background-color: ${rowColor};">
                    <td style="padding: 10px; border: 1px solid var(--border-color);">+${point.hour_offset}h</td>
                    <td style="padding: 10px; border: 1px solid var(--border-color);">${point.solar_elevation.toFixed(1)}°</td>
                    <td style="padding: 10px; border: 1px solid var(--border-color);">${point.solar_azimuth.toFixed(1)}°</td>
                    <td style="padding: 10px; border: 1px solid var(--border-color);">${point.is_daytime ? '☀️ Day' : '🌙 Night'}</td>
                </tr>
            `;
        }
    });
    
    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// Initialize with current timestamp
function setCurrentTimestamp() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    
    document.getElementById('timestamp').value = `${year}-${month}-${day}T${hours}:${minutes}`;
}

// Set current timestamp on load
window.addEventListener('load', setCurrentTimestamp);

// Check API status on load
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        if (response.ok) {
            const data = await response.json();
            console.log('API Status:', data);
        }
    } catch (error) {
        console.warn('API not available:', error);
    }
}

checkAPIStatus();
