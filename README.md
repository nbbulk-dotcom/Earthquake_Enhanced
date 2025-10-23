
# 🌍 Earthquake Enhanced - Space Engine

[![Python](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Blue_Python_3.8_Shield_Badge.svg/2560px-Blue_Python_3.8_Shield_Badge.svg.png)
[![FastAPI](https://i.ytimg.com/vi/BCXdLx6xHmc/maxresdefault.jpg)
[![License](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/MIT_Logo_New.svg/1200px-MIT_Logo_New.svg.png)

**Space Weather Correlation Engine for Earthquake Prediction**

Advanced earthquake prediction system integrating space weather data with seismic analysis using empirical physics-based calculations.

---

## 🎯 Features

### 8 Core Features Implemented

1. **🌐 85km/80km Atmospheric Boundary Refraction**
   - 1.15 calibration factor for 80km boundary
   - 1.12 calibration factor for 85km boundary
   - Physics-based refraction corrections

2. **📐 Angle of Incidence Tracking**
   - Solar elevation calculations using spherical trigonometry
   - Tetrahedral angles: 54.74° (volcanic), 26.52° (seismic)
   - Geographic to magnetic latitude conversion

3. **☀️ Sun Path Prediction**
   - Stationary Earth reference frame
   - Predictive ray path calculations
   - 24-hour ahead predictions

4. **⏱️ Dynamic Lag Time Calculation**
   - Light travel base delay: ~8.3 minutes
   - Solar lag: 4-12 hours (seasonal variation)
   - Geomagnetic lag: 4-8 hours (diurnal variation)
   - Ionospheric lag: 1-7 hours (semi-diurnal variation)

5. **🌈 RGB Resonance Calculations**
   - Formula: `sqrt((R² + G² + B²) / 3.0)`
   - R = Solar wind related variables
   - G = Magnetic field related variables
   - B = Particle flux related variables

6. **📡 Data Integration**
   - NASA OMNI2 API (88% reliability)
   - NOAA SWPC API (92% reliability)
   - Real-time space weather data
   - Graceful failure handling

7. **🎯 Resultant Resonance Calculations**
   - 12-dimensional correlation matrix
   - Eigenvalue analysis
   - Cross-variable correlations

8. **🌍 Equatorial Enhancement**
   - 1.25× enhancement factor for equatorial regions (±23.5°)
   - Latitude-based tapering

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/nbbulk-dotcom/Earthquake_Enhanced.git
cd Earthquake_Enhanced

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### 1. Start the API Server

```bash
cd backend
python api.py
```

The API will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

#### 2. Open the Frontend

Open `frontend/templates/index.html` in your web browser, or serve it with a local server:

```bash
cd frontend/templates
python -m http.server 8080
```

Then navigate to: `http://localhost:8080`

---

## 📖 API Documentation

### Main Prediction Endpoint

**POST** `/api/v1/prediction`

Calculate comprehensive space engine prediction.

**Request Body:**
```json
{
  "latitude": 35.0,
  "longitude": 140.0,
  "timestamp": "2024-06-21T12:00:00Z",
  "include_historical": false
}
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-06-21T12:00:00",
  "location": {
    "latitude": 35.0,
    "longitude": 140.0,
    "magnetic_latitude": 25.3
  },
  "earthquake_correlation_score": 0.752,
  "solar_angles": {...},
  "rgb_resonance": {...},
  "lag_times": {...},
  "resultant_resonance": {...},
  "equatorial_enhancement": {...}
}
```

### Other Endpoints

- **GET** `/api/v1/status` - Engine status
- **POST** `/api/v1/solar-angles` - Solar angle calculations
- **POST** `/api/v1/lag-times` - Lag time calculations
- **POST** `/api/v1/rgb-resonance` - RGB resonance calculation
- **GET** `/api/v1/sun-path` - Sun path prediction
- **GET** `/api/v1/atmospheric-boundary` - Boundary factors
- **GET** `/api/v1/equatorial-enhancement` - Enhancement calculation

Full API documentation available at: `http://localhost:8000/docs`

---

## 🧪 Testing

### Run Unit Tests

```bash
cd backend
python -m pytest features/tests/test_space_engine.py -v
```

Or run with coverage:

```bash
python -m pytest features/tests/test_space_engine.py --cov=features.space_engine --cov-report=html
```

### Test Coverage

The test suite includes:
- ✅ Atmospheric boundary refraction tests
- ✅ Angle of incidence tracking tests
- ✅ Sun path prediction tests
- ✅ Dynamic lag time calculation tests
- ✅ RGB resonance calculation tests
- ✅ Data integration tests (with mocking)
- ✅ Resultant resonance calculation tests
- ✅ Equatorial enhancement tests
- ✅ Edge case tests
- ✅ Integration tests

---

## 📊 Architecture

```
Earthquake_Enhanced/
├── backend/
│   ├── features/
│   │   ├── space_engine.py      # Core space engine module
│   │   └── tests/
│   │       └── test_space_engine.py
│   └── api.py                   # FastAPI application
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css       # Styling
│   │   └── js/
│   │       └── app.js           # Frontend logic
│   └── templates/
│       └── index.html           # Main UI
├── docs/
│   └── TECHNICAL.md             # Technical documentation
├── config/
│   └── config.yaml              # Configuration
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Details

### Physics Constants

- **Light Speed**: 299,792.458 km/s
- **Sun-Earth Distance**: 149,597,870.7 km (1 AU)
- **Earth Radius**: 6,371 km
- **Schumann Base Frequency**: 7.83 Hz

### Calibration Factors

- **80km Boundary Refraction**: 1.15
- **85km Boundary Refraction**: 1.12
- **Equatorial Enhancement**: 1.25
- **Tetrahedral Angle (Volcanic)**: 54.74°
- **Tetrahedral Angle (Seismic)**: 26.52°

### 12D Space Variables

1. Solar Activity
2. Geomagnetic Field
3. Planetary Alignment
4. Cosmic Ray Intensity
5. Solar Wind Pressure
6. Ionospheric Density
7. Magnetosphere Compression
8. Auroral Activity
9. Solar Flare Intensity
10. Coronal Mass Ejection
11. Interplanetary Magnetic Field
12. Galactic Cosmic Radiation

---

## 🌐 Data Sources

### NASA OMNI2 API
- **Reliability**: 88%
- **URL**: https://omniweb.gsfc.nasa.gov/
- **Data**: Solar wind, magnetic field, particle flux
- **Update Frequency**: Hourly

### NOAA SWPC API
- **Reliability**: 92%
- **URL**: https://www.swpc.noaa.gov/
- **Data**: Solar flares, geomagnetic indices, real-time space weather
- **Update Frequency**: 1-minute to hourly

---

## 📚 Usage Examples

### Python API Usage

```python
from features.space_engine import SpaceEngine
import asyncio
from datetime import datetime

async def main():
    engine = SpaceEngine()
    
    # Calculate prediction
    result = await engine.calculate_space_prediction(
        latitude=35.0,
        longitude=140.0,
        timestamp=datetime.utcnow()
    )
    
    print(f"Correlation Score: {result['earthquake_correlation_score']}")
    print(f"RGB Resonance: {result['rgb_resonance']['rgb_resonance']}")

asyncio.run(main())
```

### cURL Examples

```bash
# Get engine status
curl http://localhost:8000/api/v1/status

# Calculate prediction
curl -X POST http://localhost:8000/api/v1/prediction \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 35.0,
    "longitude": 140.0
  }'

# Get solar angles
curl -X POST http://localhost:8000/api/v1/solar-angles \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 35.0,
    "longitude": 140.0
  }'
```

---

## 🛠️ Development

### Adding New Features

1. Extend `SpaceEngine` class in `backend/features/space_engine.py`
2. Add corresponding tests in `backend/features/tests/test_space_engine.py`
3. Create API endpoint in `backend/api.py`
4. Update frontend if needed

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Document all public methods
- Write unit tests for new features

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **GitHub**: https://github.com/nbbulk-dotcom/Earthquake_Enhanced
- **Documentation**: `/docs`
- **API Docs**: http://localhost:8000/docs
- **Issues**: https://github.com/nbbulk-dotcom/Earthquake_Enhanced/issues

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- NASA OMNI2 for space weather data
- NOAA SWPC for real-time space weather
- Original BRETT system algorithms
- Extracted code from GEO_EARTH, QuakePredictionTestSystem, and EarthQuake_historical_Test repositories

---

## ⚠️ Disclaimer

This system is for research and educational purposes. Earthquake prediction is inherently uncertain. This tool should not be used as the sole basis for safety decisions. Always follow official guidance from geological and emergency management authorities.

---

**Version**: 1.0.0  
**Last Updated**: October 2024  
**Status**: ✅ Operational
