
"""
Database Models for Earthquake_Enhanced System
Stores resonance sources, overlay regions, patterns, and predictions
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime, timedelta
import json
import math

Base = declarative_base()


class ResonanceSourceDB(Base):
    """Database model for resonance sources"""
    __tablename__ = 'resonance_sources'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(50), unique=True, nullable=False, index=True)
    source_name = Column(String(200), nullable=False)
    source_type = Column(String(50), nullable=False, index=True)  # 'space', 'strain-rate', 'custom'
    frequency = Column(Float, nullable=False)
    amplitude = Column(Float, nullable=False)
    phase = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    metadata = Column(JSON)
    
    # Relationships
    overlay_associations = relationship('OverlaySourceAssociation', back_populates='resonance_source')
    
    def to_dict(self):
        return {
            'id': self.id,
            'source_id': self.source_id,
            'source_name': self.source_name,
            'source_type': self.source_type,
            'frequency': self.frequency,
            'amplitude': self.amplitude,
            'phase': self.phase,
            'location': {'latitude': self.latitude, 'longitude': self.longitude},
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class OverlayRegionDB(Base):
    """Database model for overlay regions"""
    __tablename__ = 'overlay_regions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    region_id = Column(String(50), unique=True, nullable=False, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    resultant_frequency = Column(Float, nullable=False)
    resultant_amplitude = Column(Float, nullable=False)
    resultant_phase = Column(Float, nullable=False)
    interference_type = Column(String(50), nullable=False, index=True)  # 'constructive', 'destructive', 'mixed'
    coherence_coefficient = Column(Float, nullable=False)
    overlay_count = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Relationships
    source_associations = relationship('OverlaySourceAssociation', back_populates='overlay_region')
    
    def to_dict(self):
        return {
            'id': self.id,
            'region_id': self.region_id,
            'location': {'latitude': self.latitude, 'longitude': self.longitude},
            'resultant_frequency': self.resultant_frequency,
            'resultant_amplitude': self.resultant_amplitude,
            'resultant_phase': self.resultant_phase,
            'interference_type': self.interference_type,
            'coherence_coefficient': self.coherence_coefficient,
            'overlay_count': self.overlay_count,
            'timestamp': self.timestamp.isoformat()
        }


class OverlaySourceAssociation(Base):
    """Association table linking overlay regions to resonance sources"""
    __tablename__ = 'overlay_source_associations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    overlay_region_id = Column(Integer, ForeignKey('overlay_regions.id'), nullable=False, index=True)
    resonance_source_id = Column(Integer, ForeignKey('resonance_sources.id'), nullable=False, index=True)
    
    # Relationships
    overlay_region = relationship('OverlayRegionDB', back_populates='source_associations')
    resonance_source = relationship('ResonanceSourceDB', back_populates='overlay_associations')


class ResonancePatternDB(Base):
    """Database model for identified resonance patterns"""
    __tablename__ = 'resonance_patterns'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_id = Column(String(50), unique=True, nullable=False, index=True)
    pattern_name = Column(String(200), nullable=False)
    frequency_signature = Column(JSON, nullable=False)  # List of frequencies
    recurrence_period = Column(Float)  # Days
    similarity_score = Column(Float, nullable=False)
    first_observed = Column(DateTime, nullable=False, index=True)
    last_observed = Column(DateTime, nullable=False, index=True)
    occurrence_count = Column(Integer, nullable=False)
    temporal_evolution = Column(JSON)  # List of evolution data points
    
    def to_dict(self):
        return {
            'id': self.id,
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'frequency_signature': self.frequency_signature,
            'recurrence_period': self.recurrence_period,
            'similarity_score': self.similarity_score,
            'first_observed': self.first_observed.isoformat(),
            'last_observed': self.last_observed.isoformat(),
            'occurrence_count': self.occurrence_count
        }


class PredictionDB(Base):
    """Database model for 21-day predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(50), unique=True, nullable=False, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    depth_km = Column(Float, nullable=False)
    prediction_start = Column(DateTime, nullable=False, index=True)
    prediction_days = Column(Integer, nullable=False)
    daily_predictions = Column(JSON, nullable=False)  # List of daily prediction objects
    summary = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'prediction_id': self.prediction_id,
            'location': {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'depth_km': self.depth_km
            },
            'prediction_start': self.prediction_start.isoformat(),
            'prediction_days': self.prediction_days,
            'daily_predictions': self.daily_predictions,
            'summary': self.summary,
            'created_at': self.created_at.isoformat()
        }


class AnalysisResultDB(Base):
    """Database model for single-point and multi-fault analyses"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(50), unique=True, nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False, index=True)  # 'single_point', 'multi_fault'
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    depth_km = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    resonance_sources_count = Column(Integer, nullable=False)
    resultant_frequency = Column(Float, nullable=False)
    resultant_amplitude = Column(Float, nullable=False)
    coherence_coefficient = Column(Float, nullable=False)
    risk_score = Column(Float)
    risk_level = Column(String(50))
    analysis_data = Column(JSON, nullable=False)  # Full analysis results
    
    def to_dict(self):
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'analysis_type': self.analysis_type,
            'location': {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'depth_km': self.depth_km
            },
            'timestamp': self.timestamp.isoformat(),
            'resonance_sources_count': self.resonance_sources_count,
            'resultant_frequency': self.resultant_frequency,
            'resultant_amplitude': self.resultant_amplitude,
            'coherence_coefficient': self.coherence_coefficient,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level
        }


# Database initialization and utility functions
class DatabaseManager:
    """Manager for database operations"""
    
    def __init__(self, database_url='sqlite:///earthquake_enhanced.db'):
        """
        Initialize database manager
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_engine(database_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_all_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(self.engine)
    
    def drop_all_tables(self):
        """Drop all database tables (use with caution!)"""
        Base.metadata.drop_all(self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.Session()
    
    def save_resonance_source(self, source):
        """Save a resonance source to database"""
        session = self.get_session()
        try:
            db_source = ResonanceSourceDB(
                source_id=source.source_id,
                source_name=source.source_name,
                source_type=source.source_type,
                frequency=source.frequency,
                amplitude=source.amplitude,
                phase=source.phase,
                latitude=source.location[0],
                longitude=source.location[1],
                timestamp=source.timestamp,
                metadata=source.metadata
            )
            session.add(db_source)
            session.commit()
            return db_source.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def save_overlay_region(self, region, source_ids):
        """Save an overlay region with its associated sources"""
        session = self.get_session()
        try:
            db_region = OverlayRegionDB(
                region_id=region.region_id,
                latitude=region.location[0],
                longitude=region.location[1],
                resultant_frequency=region.resultant_frequency,
                resultant_amplitude=region.resultant_amplitude,
                resultant_phase=region.resultant_phase,
                interference_type=region.interference_type,
                coherence_coefficient=region.coherence_coefficient,
                overlay_count=region.overlay_count,
                timestamp=region.timestamp
            )
            session.add(db_region)
            session.flush()  # Get the ID
            
            # Add source associations
            for source_id in source_ids:
                # Get source DB ID
                source = session.query(ResonanceSourceDB).filter_by(source_id=source_id).first()
                if source:
                    assoc = OverlaySourceAssociation(
                        overlay_region_id=db_region.id,
                        resonance_source_id=source.id
                    )
                    session.add(assoc)
            
            session.commit()
            return db_region.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def save_prediction(self, prediction_data, prediction_id):
        """Save a 21-day prediction"""
        session = self.get_session()
        try:
            db_prediction = PredictionDB(
                prediction_id=prediction_id,
                latitude=prediction_data['location']['latitude'],
                longitude=prediction_data['location']['longitude'],
                depth_km=prediction_data['location']['depth_km'],
                prediction_start=datetime.fromisoformat(prediction_data['prediction_start']),
                prediction_days=prediction_data['prediction_days'],
                daily_predictions=prediction_data['daily_predictions'],
                summary=prediction_data['summary']
            )
            session.add(db_prediction)
            session.commit()
            return db_prediction.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def query_recent_overlays(self, hours=24, limit=100):
        """Query recent overlay regions"""
        session = self.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            overlays = session.query(OverlayRegionDB).filter(
                OverlayRegionDB.timestamp >= cutoff
            ).order_by(OverlayRegionDB.timestamp.desc()).limit(limit).all()
            
            return [overlay.to_dict() for overlay in overlays]
        finally:
            session.close()
    
    def query_overlays_by_location(self, latitude, longitude, radius_km=100, limit=100):
        """Query overlay regions near a location"""
        session = self.get_session()
        try:
            # Simple bounding box query (can be improved with PostGIS for real distance)
            lat_range = radius_km / 111.0  # Approximate km per degree latitude
            lon_range = radius_km / (111.0 * math.cos(math.radians(latitude)))
            
            overlays = session.query(OverlayRegionDB).filter(
                OverlayRegionDB.latitude >= latitude - lat_range,
                OverlayRegionDB.latitude <= latitude + lat_range,
                OverlayRegionDB.longitude >= longitude - lon_range,
                OverlayRegionDB.longitude <= longitude + lon_range
            ).limit(limit).all()
            
            return [overlay.to_dict() for overlay in overlays]
        finally:
            session.close()


# Convenience function
def get_database_manager(database_url='sqlite:///earthquake_enhanced.db'):
    """Get a database manager instance"""
    return DatabaseManager(database_url)
