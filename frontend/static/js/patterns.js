/**
 * Pattern Identification and Visualization
 */

function displayPatterns(patternsData) {
    if (!patternsData || !patternsData.patterns) {
        console.error('Invalid patterns data');
        return;
    }
    
    const container = document.getElementById('patternsDisplay');
    if (!container) return;
    
    if (patternsData.patterns.length === 0) {
        container.innerHTML = '<p>No recurring patterns identified in the time window.</p>';
        return;
    }
    
    // Create pattern cards
    let html = '';
    patternsData.patterns.forEach(pattern => {
        html += createPatternCard(pattern);
    });
    
    container.innerHTML = html;
    
    // Display evolution chart for first pattern
    if (patternsData.patterns.length > 0) {
        displayPatternEvolution(patternsData.patterns[0]);
    }
}

function createPatternCard(pattern) {
    return `
        <div class="pattern-card" data-pattern-id="${pattern.pattern_id}">
            <div class="pattern-header">
                <span class="pattern-name">${pattern.pattern_name}</span>
                <span class="pattern-score">Similarity: ${(pattern.similarity_score * 100).toFixed(0)}%</span>
            </div>
            <div class="pattern-details">
                <p><strong>Frequency Signature:</strong> ${pattern.frequency_signature.map(f => f.toFixed(2)).join(', ')}</p>
                <p><strong>Recurrence Period:</strong> ${pattern.recurrence_period ? pattern.recurrence_period.toFixed(1) + ' days' : 'N/A'}</p>
                <p><strong>Occurrences:</strong> ${pattern.occurrence_count}</p>
                <p><strong>First Observed:</strong> ${new Date(pattern.first_observed).toLocaleDateString()}</p>
                <p><strong>Last Observed:</strong> ${new Date(pattern.last_observed).toLocaleDateString()}</p>
            </div>
        </div>
    `;
}

function displayPatternEvolution(pattern) {
    if (!pattern.temporal_evolution || pattern.temporal_evolution.length === 0) {
        return;
    }
    
    // Prepare data
    const timestamps = pattern.temporal_evolution.map(e => e.timestamp);
    const avgFreqs = pattern.temporal_evolution.map(e => {
        const sig = e.signature;
        return sig.reduce((a, b) => a + b, 0) / sig.length;
    });
    
    // Create trace
    const trace = {
        x: timestamps,
        y: avgFreqs,
        type: 'scatter',
        mode: 'lines+markers',
        name: pattern.pattern_name,
        line: { color: '#1a73e8', width: 2 }
    };
    
    // Layout
    const layout = {
        title: `Pattern Evolution: ${pattern.pattern_name}`,
        xaxis: {
            title: 'Time',
            tickangle: -45
        },
        yaxis: {
            title: 'Average Frequency (Hz)'
        }
    };
    
    // Plot
    Plotly.newPlot('patternEvolution', [trace], layout, {
        responsive: true
    });
}

window.displayPatterns = displayPatterns;
