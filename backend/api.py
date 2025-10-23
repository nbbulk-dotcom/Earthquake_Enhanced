"""
FastAPI Backend for Earthquake Enhanced System
Provides REST API endpoints for correlation engine
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple
from datetime import datetime
import asyncio
import hashlib

# Import our engines
from backend.features.correlation_engine import CorrelationEngine, get_correlation_engine
from backend.features.space_engine import SpaceEngine, get_space_engine
from backend.features.resonance import ResonanceEngine, get_resonance_engine
from backend.models import get_database_manager

# Create FastAPI app
app = FastAPI(
    title="Earthquake Enhanced API",
    description="Multi-Resonance Overlay Analysis System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instances
correlation_engine = get_correlation_engine()
space_engine = get_space_engine()
resonance_engine = get_resonance_engine()
db_manager = get_database_manager()

# Request models
class SinglePointRequest(BaseModel):
    latitude: float
    longitude: float
    depth_km: float = 15.0

class MultiFaultRequest(BaseModel):
    center_lat: float
    center_lon: float
    triangulation_points: List[Tuple[float, float]]
    depth_km: float = 15.0

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    depth_km: float = 15.0

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Earthquake Enhanced System API",
        "version": "1.0.0",
        "endpoints": {
            "analyze_single": "/api/analyze/single",
            "analyze_multi_fault": "/api/analyze/multi-fault",
            "predict_21day": "/api/predict/21-day",
            "identify_patterns": "/api/patterns/identify",
            "overlay_statistics": "/api/overlays/statistics",
            "registry_summary": "/api/registry/summary",
            "engine_status": "/api/status"
        }
    }

# Status endpoint
@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "correlation_engine": correlation_engine.get_engine_status(),
        "space_engine": space_engine.get_engine_status(),
        "resonance_engine": resonance_engine.get_engine_status()
    }

# Single-point analysis
@app.post("/api/analyze/single")
async def analyze_single_point(request: SinglePointRequest):
    """
    Analyze resonances at a single geographic point
    """
    try:
        result = await correlation_engine.analyze_single_point(
            request.latitude,
            request.longitude,
            request.depth_km
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Multi-fault analysis
@app.post("/api/analyze/multi-fault")
async def analyze_multi_fault(request: MultiFaultRequest):
    """
    Analyze multi-fault region with triangulation points
    """
    try:
        result = await correlation_engine.analyze_multi_fault_region(
            request.center_lat,
            request.center_lon,
            request.triangulation_points,
            request.depth_km
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 21-day prediction
@app.post("/api/predict/21-day")
async def predict_21_days(request: PredictionRequest):
    """
    Generate 21-day forward prediction
    """
    try:
        result = await correlation_engine.generate_21day_prediction(
            request.latitude,
            request.longitude,
            request.depth_km
        )
        
        # Save to database
        prediction_id = hashlib.md5(
            f"{request.latitude}_{request.longitude}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        try:
            db_manager.save_prediction(result, prediction_id)
        except Exception as db_error:
            print(f"Warning: Failed to save prediction to database: {db_error}")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pattern identification
@app.get("/api/patterns/identify")
async def identify_patterns(time_window_days: int = 30):
    """
    Identify recurring resonance patterns
    """
    try:
        patterns = correlation_engine.identify_recurring_patterns(time_window_days)
        return {
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_name": p.pattern_name,
                    "frequency_signature": p.frequency_signature,
                    "recurrence_period": p.recurrence_period,
                    "similarity_score": p.similarity_score,
                    "first_observed": p.first_observed.isoformat(),
                    "last_observed": p.last_observed.isoformat(),
                    "occurrence_count": p.occurrence_count,
                    "temporal_evolution": p.temporal_evolution
                }
                for p in patterns
            ],
            "total_patterns": len(patterns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Overlay statistics
@app.get("/api/overlays/statistics")
async def get_overlay_statistics(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_km: Optional[float] = None
):
    """
    Get overlay region statistics
    """
    try:
        location = (latitude, longitude) if latitude and longitude else None
        stats = correlation_engine.get_overlay_statistics(location, radius_km)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Resonance registry
@app.get("/api/registry/summary")
async def get_registry_summary():
    """
    Get resonance source registry summary
    """
    try:
        summary = correlation_engine.get_resonance_registry_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Query overlays
@app.get("/api/overlays/query")
async def query_overlays(
    min_overlay_count: Optional[int] = None,
    min_coherence: Optional[float] = None,
    interference_type: Optional[str] = None,
    time_window_hours: Optional[int] = None
):
    """
    Query overlay regions by criteria
    """
    try:
        regions = correlation_engine.query_overlays_by_criteria(
            min_overlay_count,
            min_coherence,
            interference_type,
            time_window_hours
        )
        
        return {
            "regions": [
                {
                    "region_id": r.region_id,
                    "location": r.location,
                    "resultant_frequency": r.resultant_frequency,
                    "resultant_amplitude": r.resultant_amplitude,
                    "interference_type": r.interference_type,
                    "coherence_coefficient": r.coherence_coefficient,
                    "overlay_count": r.overlay_count,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in regions
            ],
            "total": len(regions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Space engine prediction (direct)
@app.post("/api/space/predict")
async def space_prediction(request: SinglePointRequest):
    """
    Get space engine prediction directly
    """
    try:
        result = await space_engine.calculate_space_prediction(
            request.latitude,
            request.longitude
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Resonance engine analysis (direct)
@app.post("/api/resonance/analyze")
async def resonance_analysis(request: SinglePointRequest):
    """
    Get resonance engine analysis directly
    """
    try:
        result = resonance_engine.calculate_comprehensive_resonance(
            request.latitude,
            request.longitude,
            request.depth_km
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    try:
        db_manager.create_all_tables()
        print("Database tables initialized successfully")
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
