/**
 * 21-Day Prediction Visualization
 */

function displayPrediction(predictionData) {
    if (!predictionData || !predictionData.daily_predictions) {
        console.error('Invalid prediction data');
        return;
    }
    
    // Create prediction chart
    createPredictionChart(predictionData);
    
    // Display summary
    displayPredictionSummary(predictionData.summary);
}

function createPredictionChart(data) {
    const daily = data.daily_predictions;
    
    // Prepare data for Plotly
    const dates = daily.map(d => d.date);
    const riskScores = daily.map(d => d.risk_score);
    const amplitudes = daily.map(d => d.resultant_amplitude);
    const confidence = daily.map(d => d.confidence);
    
    // Create traces
    const riskTrace = {
        x: dates,
        y: riskScores,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Risk Score',
        line: { color: '#ea4335', width: 3 },
        yaxis: 'y1'
    };
    
    const amplitudeTrace = {
        x: dates,
        y: amplitudes,
        type: 'scatter',
        mode: 'lines',
        name: 'Amplitude',
        line: { color: '#fbbc04', width: 2 },
        yaxis: 'y2'
    };
    
    const confidenceTrace = {
        x: dates,
        y: confidence,
        type: 'scatter',
        mode: 'lines',
        name: 'Confidence',
        line: { color: '#34a853', width: 2, dash: 'dash' },
        yaxis: 'y1'
    };
    
    // Layout
    const layout = {
        title: '21-Day Earthquake Risk Prediction',
        xaxis: {
            title: 'Date',
            tickangle: -45
        },
        yaxis: {
            title: 'Risk Score / Confidence',
            side: 'left',
            range: [0, 1]
        },
        yaxis2: {
            title: 'Amplitude',
            side: 'right',
            overlaying: 'y',
            range: [0, Math.max(...amplitudes) * 1.2]
        },
        hovermode: 'x unified',
        showlegend: true,
        legend: {
            x: 0.01,
            y: 0.99
        }
    };
    
    // Plot
    Plotly.newPlot('predictionChart', [riskTrace, amplitudeTrace, confidenceTrace], layout, {
        responsive: true
    });
}

function displayPredictionSummary(summary) {
    const container = document.getElementById('predictionSummary');
    if (!container) return;
    
    container.innerHTML = `
        <div class="summary-grid">
            <div class="summary-item">
                <span class="summary-label">Peak Risk Score</span>
                <span class="summary-value">${summary.max_risk_score.toFixed(2)}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Peak Risk Day</span>
                <span class="summary-value">Day ${summary.peak_risk_day}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Peak Risk Date</span>
                <span class="summary-value">${summary.peak_risk_date}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Peak Risk Level</span>
                <span class="summary-value risk-badge" data-level="${summary.peak_risk_level}">
                    ${summary.peak_risk_level}
                </span>
            </div>
            <div class="summary-item">
                <span class="summary-label">High Risk Days</span>
                <span class="summary-value">${summary.high_risk_days}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Max Amplitude</span>
                <span class="summary-value">${summary.max_amplitude.toFixed(3)}</span>
            </div>
        </div>
    `;
}

window.displayPrediction = displayPrediction;
