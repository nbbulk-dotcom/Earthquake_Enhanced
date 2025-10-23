/**
 * API Client for Earthquake Enhanced System
 * Handles all backend communication
 */

const API_BASE_URL = 'http://localhost:8000/api';

class EarthquakeAPI {
    constructor() {
        this.baseURL = API_BASE_URL;
    }

    async analyzeSinglePoint(latitude, longitude, depth_km = 15.0) {
        try {
            const response = await fetch(`${this.baseURL}/analyze/single`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    latitude,
                    longitude,
                    depth_km
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error analyzing single point:', error);
            throw error;
        }
    }

    async analyzeMultiFault(centerLat, centerLon, triangulationPoints, depth_km = 15.0) {
        try {
            const response = await fetch(`${this.baseURL}/analyze/multi-fault`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    center_lat: centerLat,
                    center_lon: centerLon,
                    triangulation_points: triangulationPoints,
                    depth_km
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error analyzing multi-fault region:', error);
            throw error;
        }
    }

    async generate21DayPrediction(latitude, longitude, depth_km = 15.0) {
        try {
            const response = await fetch(`${this.baseURL}/predict/21-day`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    latitude,
                    longitude,
                    depth_km
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error generating prediction:', error);
            throw error;
        }
    }

    async identifyPatterns(timeWindowDays = 30) {
        try {
            const response = await fetch(`${this.baseURL}/patterns/identify?time_window_days=${timeWindowDays}`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error identifying patterns:', error);
            throw error;
        }
    }

    async getOverlayStatistics(location = null, radius_km = null) {
        try {
            let url = `${this.baseURL}/overlays/statistics`;
            const params = new URLSearchParams();
            
            if (location) {
                params.append('latitude', location.latitude);
                params.append('longitude', location.longitude);
            }
            if (radius_km) {
                params.append('radius_km', radius_km);
            }

            if (params.toString()) {
                url += `?${params.toString()}`;
            }

            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting overlay statistics:', error);
            throw error;
        }
    }

    async getResonanceRegistry() {
        try {
            const response = await fetch(`${this.baseURL}/registry/summary`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting resonance registry:', error);
            throw error;
        }
    }
}

// Export API instance
const api = new EarthquakeAPI();
