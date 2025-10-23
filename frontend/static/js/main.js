/**
 * Main Application Logic
 */

// Global state
const appState = {
    currentLocation: { lat: 35.6762, lon: 139.6503, depth: 15.0 },
    analysisData: null,
    predictionData: null,
    patternsData: null,
    isAnimating: false,
    currentTimeStep: 0
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Earthquake Enhanced System - Initializing...');
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize tabs
    initializeTabs();
    
    // Load initial data
    loadInitialData();
    
    console.log('System initialized successfully!');
});

function setupEventListeners() {
    // Analyze button
    document.getElementById('analyzeBtn').addEventListener('click', handleAnalyze);
    
    // Generate prediction button
    const predBtn = document.getElementById('generatePredictionBtn');
    if (predBtn) {
        predBtn.addEventListener('click', handleGeneratePrediction);
    }
    
    // Identify patterns button
    const patternBtn = document.getElementById('identifyPatternsBtn');
    if (patternBtn) {
        patternBtn.addEventListener('click', handleIdentifyPatterns);
    }
    
    // Animation controls
    document.getElementById('playBtn').addEventListener('click', handlePlay);
    document.getElementById('pauseBtn').addEventListener('click', handlePause);
    document.getElementById('resetBtn').addEventListener('click', handleReset);
    
    // Time slider
    document.getElementById('timeSlider').addEventListener('input', handleTimeSliderChange);
    
    // Location inputs
    document.getElementById('latitude').addEventListener('change', updateLocation);
    document.getElementById('longitude').addEventListener('change', updateLocation);
    document.getElementById('depth').addEventListener('change', updateLocation);
}

function initializeTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const tabName = this.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active from all nav tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    const contentId = tabName + 'Tab';
    document.getElementById(contentId).classList.add('active');
    
    // Activate nav tab
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
}

async function handleAnalyze() {
    showLoading(true);
    
    try {
        const analysisType = document.querySelector('input[name="analysisType"]:checked').value;
        
        if (analysisType === 'single') {
            appState.analysisData = await api.analyzeSinglePoint(
                appState.currentLocation.lat,
                appState.currentLocation.lon,
                appState.currentLocation.depth
            );
        } else {
            // Multi-fault analysis with Tokyo-style triangulation
            const triangulationPoints = generateTriangulationPoints(
                appState.currentLocation.lat,
                appState.currentLocation.lon,
                4  // 4 points around center
            );
            
            appState.analysisData = await api.analyzeMultiFault(
                appState.currentLocation.lat,
                appState.currentLocation.lon,
                triangulationPoints,
                appState.currentLocation.depth
            );
        }
        
        // Update displays
        updateStatisticsDisplay();
        updateVisualization();
        
        console.log('Analysis complete:', appState.analysisData);
    } catch (error) {
        console.error('Analysis failed:', error);
        alert('Analysis failed. Please try again.');
    } finally {
        showLoading(false);
    }
}

async function handleGeneratePrediction() {
    showLoading(true);
    
    try {
        appState.predictionData = await api.generate21DayPrediction(
            appState.currentLocation.lat,
            appState.currentLocation.lon,
            appState.currentLocation.depth
        );
        
        // Update prediction display
        displayPrediction(appState.predictionData);
        
        console.log('Prediction generated:', appState.predictionData);
    } catch (error) {
        console.error('Prediction generation failed:', error);
        alert('Prediction generation failed. Please try again.');
    } finally {
        showLoading(false);
    }
}

async function handleIdentifyPatterns() {
    showLoading(true);
    
    try {
        appState.patternsData = await api.identifyPatterns(30);
        
        // Update patterns display
        displayPatterns(appState.patternsData);
        
        console.log('Patterns identified:', appState.patternsData);
    } catch (error) {
        console.error('Pattern identification failed:', error);
        alert('Pattern identification failed. Please try again.');
    } finally {
        showLoading(false);
    }
}

function handlePlay() {
    appState.isAnimating = true;
    animateTimeSeries();
}

function handlePause() {
    appState.isAnimating = false;
}

function handleReset() {
    appState.isAnimating = false;
    appState.currentTimeStep = 0;
    document.getElementById('timeSlider').value = 0;
    updateTimeDisplay(0);
    updateVisualization();
}

function handleTimeSliderChange(event) {
    appState.currentTimeStep = parseInt(event.target.value);
    updateTimeDisplay(appState.currentTimeStep);
    updateVisualization();
}

function updateLocation() {
    appState.currentLocation.lat = parseFloat(document.getElementById('latitude').value);
    appState.currentLocation.lon = parseFloat(document.getElementById('longitude').value);
    appState.currentLocation.depth = parseFloat(document.getElementById('depth').value);
}

function updateStatisticsDisplay() {
    if (!appState.analysisData) return;
    
    const data = appState.analysisData;
    
    document.getElementById('totalSources').textContent = data.resonance_sources || 0;
    document.getElementById('overlayCount').textContent = data.overlay_region?.overlay_count || 0;
    document.getElementById('coherenceValue').textContent = 
        (data.coherence?.coherence_coefficient || 0).toFixed(2);
    
    const riskBadge = document.getElementById('riskLevel');
    const riskLevel = data.overlay_region?.risk_level || 'LOW';
    riskBadge.textContent = riskLevel;
    riskBadge.setAttribute('data-level', riskLevel);
}

function updateVisualization() {
    if (!appState.analysisData) return;
    
    // Update 3D visualization
    if (window.updateOverlay3D) {
        window.updateOverlay3D(appState.analysisData, appState.currentTimeStep);
    }
    
    // Update info panels
    updateInfoPanels();
}

function updateInfoPanels() {
    if (!appState.analysisData) return;
    
    const data = appState.analysisData;
    
    // Resultant frequency
    const freqDisplay = document.getElementById('resultantFreq');
    if (freqDisplay && data.superposition) {
        freqDisplay.textContent = data.superposition.resultant_frequency.toFixed(2);
    }
    
    // Interference type
    const interferenceDisplay = document.getElementById('interferenceType');
    if (interferenceDisplay && data.superposition) {
        const type = data.superposition.interference_type;
        interferenceDisplay.textContent = type.charAt(0).toUpperCase() + type.slice(1);
        interferenceDisplay.setAttribute('data-type', type);
    }
}

function updateTimeDisplay(step) {
    document.getElementById('timeDisplay').textContent = `${step}h`;
}

function animateTimeSeries() {
    if (!appState.isAnimating) return;
    
    appState.currentTimeStep = (appState.currentTimeStep + 1) % 25;
    document.getElementById('timeSlider').value = appState.currentTimeStep;
    updateTimeDisplay(appState.currentTimeStep);
    updateVisualization();
    
    setTimeout(() => animateTimeSeries(), 200);
}

function generateTriangulationPoints(centerLat, centerLon, count) {
    const points = [];
    const radius = 0.1; // Approximate 10km
    
    for (let i = 0; i < count; i++) {
        const angle = (2 * Math.PI * i) / count;
        const lat = centerLat + radius * Math.cos(angle);
        const lon = centerLon + radius * Math.sin(angle);
        points.push([lat, lon]);
    }
    
    return points;
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.add('active');
    } else {
        overlay.classList.remove('active');
    }
}

async function loadInitialData() {
    // Load initial statistics
    try {
        const stats = await api.getOverlayStatistics();
        console.log('Initial statistics:', stats);
    } catch (error) {
        console.error('Failed to load initial data:', error);
    }
}

// Export for use in other modules
window.appState = appState;
window.updateStatisticsDisplay = updateStatisticsDisplay;
window.updateVisualization = updateVisualization;
