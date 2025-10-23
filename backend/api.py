"""
FastAPI Application for Earthquake_Enhanced Space Engine

Provides REST API endpoints for space weather correlation and earthquake prediction
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from features.space_engine import SpaceEngine, get_space_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Earthquake Enhanced - Space Engine API",
    description="Space weather correlation engine for earthquake prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Space Engine
space_engine = get_space_engine()


# Request/Response Models
class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Geographic latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Geographic longitude")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp (default: now)")
    include_historical: bool = Field(False, description="Include historical data analysis")


class SolarAngleRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: Optional[str] = None


class LagTimeRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: Optional[str] = None


class RGBResonanceRequest(BaseModel):
    space_readings: Dict[str, float]


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Earthquake Enhanced - Space Engine API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "prediction": "/api/v1/prediction",
            "solar_angles": "/api/v1/solar-angles",
            "lag_times": "/api/v1/lag-times",
            "rgb_resonance": "/api/v1/rgb-resonance",
            "sun_path": "/api/v1/sun-path",
            "engine_status": "/api/v1/status"
        },
        "documentation": "/docs"
    }


@app.get("/api/v1/status")
async def get_engine_status():
    """Get space engine status and configuration"""
    try:
        status = space_engine.get_engine_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/prediction")
async def calculate_prediction(request: PredictionRequest):
    """
    Calculate comprehensive space engine prediction
    
    Returns earthquake correlation analysis based on space weather data
    """
    try:
        # Parse timestamp
        if request.timestamp:
            timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
        else:
            timestamp = datetime.utcnow()
        
        # Calculate prediction
        result = await space_engine.calculate_space_prediction(
            latitude=request.latitude,
            longitude=request.longitude,
            timestamp=timestamp,
            include_historical=request.include_historical
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/solar-angles")
async def calculate_solar_angles(request: SolarAngleRequest):
    """
    Calculate solar elevation and angles
    
    Returns solar position information using spherical trigonometry
    """
    try:
        timestamp = (datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
                    if request.timestamp else datetime.utcnow())
        
        solar_angles = space_engine.calculate_solar_elevation(
            request.latitude,
            request.longitude,
            timestamp
        )
        
        magnetic_lat = space_engine.calculate_magnetic_latitude(
            request.latitude,
            request.longitude
        )
        
        tetrahedral_seismic = space_engine.calculate_tetrahedral_angle(
            request.latitude,
            request.longitude,
            'seismic'
        )
        
        tetrahedral_volcanic = space_engine.calculate_tetrahedral_angle(
            request.latitude,
            request.longitude,
            'volcanic'
        )
        
        return {
            "success": True,
            "timestamp": timestamp.isoformat(),
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude,
                "magnetic_latitude": magnetic_lat
            },
            "solar_angles": solar_angles,
            "tetrahedral_angles": {
                "seismic": tetrahedral_seismic,
                "volcanic": tetrahedral_volcanic
            }
        }
        
    except Exception as e:
        logger.error(f"Solar angle calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/lag-times")
async def calculate_lag_times(request: LagTimeRequest):
    """
    Calculate dynamic lag times for space-to-Earth effects
    
    Returns physics-based lag time calculations
    """
    try:
        timestamp = (datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
                    if request.timestamp else datetime.utcnow())
        
        lag_times = space_engine.calculate_dynamic_lag_times(
            request.latitude,
            request.longitude,
            timestamp
        )
        
        return {
            "success": True,
            "timestamp": timestamp.isoformat(),
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "lag_times": lag_times
        }
        
    except Exception as e:
        logger.error(f"Lag time calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rgb-resonance")
async def calculate_rgb_resonance(request: RGBResonanceRequest):
    """
    Calculate RGB resonance from space variable readings
    
    Formula: sqrt((R² + G² + B²) / 3.0)
    """
    try:
        rgb_result = space_engine.calculate_rgb_resonance(request.space_readings)
        
        return {
            "success": True,
            "rgb_resonance": rgb_result
        }
        
    except Exception as e:
        logger.error(f"RGB resonance calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/sun-path")
async def predict_sun_path(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    hours_ahead: int = Query(24, ge=1, le=168),
    timestamp: Optional[str] = Query(None)
):
    """
    Predict sun path over specified time period
    
    Returns sun position predictions at hourly intervals
    """
    try:
        ts = (datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
              if timestamp else datetime.utcnow())
        
        sun_path = space_engine.predict_sun_path(
            latitude,
            longitude,
            ts,
            hours_ahead
        )
        
        return {
            "success": True,
            "start_time": ts.isoformat(),
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "hours_predicted": hours_ahead,
            "sun_path": sun_path
        }
        
    except Exception as e:
        logger.error(f"Sun path prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/atmospheric-boundary")
async def get_atmospheric_boundary_factors():
    """
    Get atmospheric boundary refraction factors
    
    Returns 80km and 85km boundary calibration factors
    """
    try:
        factors = space_engine.get_boundary_refraction_factors()
        
        return {
            "success": True,
            "boundary_factors": factors,
            "description": {
                "80km_boundary": "Refraction factor for 80km atmospheric boundary (1.15)",
                "85km_boundary": "Refraction factor for 85km atmospheric boundary (1.12)",
                "average_boundary": "Average refraction factor"
            }
        }
        
    except Exception as e:
        logger.error(f"Boundary factors retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/equatorial-enhancement")
async def calculate_equatorial_enhancement(
    latitude: float = Query(..., ge=-90, le=90),
    base_value: float = Query(1.0, ge=0)
):
    """
    Calculate equatorial enhancement factor
    
    Applies 1.25 enhancement for equatorial regions (±23.5°)
    """
    try:
        result = space_engine.apply_equatorial_enhancement(latitude, base_value)
        
        return {
            "success": True,
            "enhancement": result
        }
        
    except Exception as e:
        logger.error(f"Equatorial enhancement calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "path": str(request.url)
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {str(exc)}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
