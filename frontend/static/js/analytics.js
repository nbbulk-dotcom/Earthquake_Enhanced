/**
 * Analytics Dashboard
 */

let analyticsCharts = {};

function initializeAnalytics() {
    // Initialize all analytics charts with dummy data
    createFrequencyDistributionChart();
    createAmplitudeTimelineChart();
    createCoherenceTimelineChart();
    createRiskAssessmentChart();
}

function createFrequencyDistributionChart() {
    const ctx = document.getElementById('freqDistChart');
    if (!ctx) return;
    
    analyticsCharts.freqDist = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0-5 Hz', '5-10 Hz', '10-15 Hz', '15-20 Hz', '20+ Hz'],
            datasets: [{
                label: 'Frequency Distribution',
                data: [12, 19, 15, 8, 5],
                backgroundColor: 'rgba(26, 115, 232, 0.5)',
                borderColor: 'rgba(26, 115, 232, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createAmplitudeTimelineChart() {
    const ctx = document.getElementById('amplitudeChart');
    if (!ctx) return;
    
    const labels = Array.from({length: 24}, (_, i) => `${i}h`);
    const data = Array.from({length: 24}, () => Math.random());
    
    analyticsCharts.amplitude = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Amplitude',
                data: data,
                borderColor: 'rgba(251, 188, 4, 1)',
                backgroundColor: 'rgba(251, 188, 4, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

function createCoherenceTimelineChart() {
    const ctx = document.getElementById('coherenceChart');
    if (!ctx) return;
    
    const labels = Array.from({length: 24}, (_, i) => `${i}h`);
    const data = Array.from({length: 24}, () => Math.random());
    
    analyticsCharts.coherence = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Coherence Coefficient',
                data: data,
                borderColor: 'rgba(52, 168, 83, 1)',
                backgroundColor: 'rgba(52, 168, 83, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

function createRiskAssessmentChart() {
    const ctx = document.getElementById('riskChart');
    if (!ctx) return;
    
    analyticsCharts.risk = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low', 'Moderate', 'Elevated', 'High', 'Critical'],
            datasets: [{
                data: [45, 25, 15, 10, 5],
                backgroundColor: [
                    'rgba(52, 168, 83, 0.8)',
                    'rgba(251, 188, 4, 0.8)',
                    'rgba(255, 152, 0, 0.8)',
                    'rgba(234, 67, 53, 0.8)',
                    'rgba(156, 39, 176, 0.8)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Delay initialization to ensure canvas elements are ready
    setTimeout(initializeAnalytics, 500);
});

window.analyticsCharts = analyticsCharts;
