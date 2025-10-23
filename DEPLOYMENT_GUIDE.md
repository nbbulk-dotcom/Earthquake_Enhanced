# 🚀 Earthquake Enhanced - Deployment Guide

## System Overview

Complete multi-resonance overlay analysis system for earthquake prediction.

**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**Tests**: 20/20 Passing  

## Pre-Deployment Checklist

- [x] All core modules implemented
  - [x] space_engine.py (8 features)
  - [x] resonance.py (strain-rate analysis)
  - [x] correlation_engine.py (8 features)
- [x] Database models created
- [x] FastAPI backend implemented
- [x] Frontend visualization complete
- [x] Comprehensive unit tests (20/20 passing)
- [x] Documentation complete
- [x] Git repository initialized

## Quick Deployment

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/nbbulk-dotcom/Earthquake_Enhanced.git
cd Earthquake_Enhanced

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database
python -c "from backend.models import get_database_manager; get_database_manager().create_all_tables()"

# 5. Start backend
python backend/api.py
# Backend runs at http://localhost:8000

# 6. Open frontend
# Open frontend/templates/visualization.html in browser
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

EXPOSE 8000

CMD ["python", "backend/api.py"]
```

Build and run:

```bash
docker build -t earthquake-enhanced .
docker run -p 8000:8000 earthquake-enhanced
```

### Production Deployment (Linux Server)

```bash
# 1. System requirements
sudo apt-get update
sudo apt-get install python3.11 python3-pip nginx

# 2. Clone and setup
cd /opt
sudo git clone https://github.com/nbbulk-dotcom/Earthquake_Enhanced.git
cd Earthquake_Enhanced

# 3. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Setup systemd service
sudo nano /etc/systemd/system/earthquake-api.service
```

Service file content:

```ini
[Unit]
Description=Earthquake Enhanced API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/Earthquake_Enhanced
Environment="PATH=/opt/Earthquake_Enhanced/venv/bin"
ExecStart=/opt/Earthquake_Enhanced/venv/bin/python backend/api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 6. Enable and start service
sudo systemctl enable earthquake-api
sudo systemctl start earthquake-api
sudo systemctl status earthquake-api

# 7. Configure Nginx
sudo nano /etc/nginx/sites-available/earthquake-enhanced
```

Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        root /opt/Earthquake_Enhanced/frontend;
        try_files $uri $uri/ /templates/visualization.html;
    }
}
```

```bash
# 8. Enable site and restart Nginx
sudo ln -s /etc/nginx/sites-available/earthquake-enhanced /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Database Configuration

### SQLite (Default)

No additional configuration needed. Database file created automatically.

### PostgreSQL (Recommended for Production)

```bash
# 1. Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# 2. Create database
sudo -u postgres createdb earthquake_db
sudo -u postgres createuser earthquake_user -P

# 3. Grant permissions
sudo -u postgres psql
postgres=# GRANT ALL PRIVILEGES ON DATABASE earthquake_db TO earthquake_user;

# 4. Update connection in backend/api.py
# Change:
db_manager = get_database_manager()
# To:
db_manager = get_database_manager('postgresql://earthquake_user:password@localhost/earthquake_db')
```

## Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=sqlite:///earthquake_enhanced.db
# or
# DATABASE_URL=postgresql://user:pass@localhost/earthquake_db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=*

# NASA/NOAA API Keys (if required)
NASA_API_KEY=your_key_here
NOAA_API_KEY=your_key_here

# Logging
LOG_LEVEL=INFO
```

## Testing Deployment

### Backend API Test

```bash
# Health check
curl http://localhost:8000/

# Status check
curl http://localhost:8000/api/status

# Single-point analysis
curl -X POST http://localhost:8000/api/analyze/single \
  -H "Content-Type: application/json" \
  -d '{"latitude": 35.6762, "longitude": 139.6503, "depth_km": 15.0}'
```

### Frontend Test

1. Open browser to: `http://localhost:8000/` (or your domain)
2. Enter location: Tokyo (35.6762, 139.6503)
3. Click "Analyze"
4. Verify:
   - ✅ 3D visualization loads
   - ✅ Statistics update
   - ✅ No console errors

### Run Unit Tests

```bash
pytest backend/features/tests/test_correlation_engine.py -v

# Expected output:
# ======================== 20 passed in 40s ========================
```

## Performance Optimization

### Backend

```python
# Enable caching for repeated requests
from functools import lru_cache

@lru_cache(maxsize=100)
async def cached_analysis(lat, lon, depth):
    return await correlation_engine.analyze_single_point(lat, lon, depth)
```

### Database

```sql
-- Add indexes for better query performance
CREATE INDEX idx_resonance_location ON resonance_sources (latitude, longitude);
CREATE INDEX idx_overlay_timestamp ON overlay_regions (timestamp);
CREATE INDEX idx_predictions_location ON predictions (latitude, longitude);
```

### Frontend

- Enable gzip compression in Nginx
- Use CDN for Three.js libraries
- Implement lazy loading for charts

## Monitoring

### Logs

```bash
# View API logs
sudo journalctl -u earthquake-api -f

# View Nginx access logs
sudo tail -f /var/log/nginx/access.log

# View application logs
tail -f /opt/Earthquake_Enhanced/app.log
```

### Health Monitoring

Create `monitor.sh`:

```bash
#!/bin/bash

# Check API health
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)

if [ $response -eq 200 ]; then
    echo "✅ API is healthy"
else
    echo "❌ API is down (HTTP $response)"
    # Send alert
    # systemctl restart earthquake-api
fi
```

Add to cron:
```bash
*/5 * * * * /opt/Earthquake_Enhanced/monitor.sh >> /var/log/earthquake-monitor.log
```

## Backup Strategy

### Database Backup

```bash
# SQLite
cp earthquake_enhanced.db earthquake_enhanced.db.backup.$(date +%Y%m%d)

# PostgreSQL
pg_dump earthquake_db > earthquake_db_backup_$(date +%Y%m%d).sql
```

### Automated Backup Script

```bash
#!/bin/bash
BACKUP_DIR="/backups/earthquake"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
cp /opt/Earthquake_Enhanced/earthquake_enhanced.db $BACKUP_DIR/db_$DATE.db

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/Earthquake_Enhanced/.env

# Keep only last 7 days
find $BACKUP_DIR -name "*.db" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

## Security Considerations

1. **API Rate Limiting**:
   ```python
   from slowapi import Limiter
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.post("/api/analyze/single")
   @limiter.limit("10/minute")
   async def analyze_single_point(request):
       ...
   ```

2. **HTTPS**: Use Let's Encrypt
   ```bash
   sudo apt-get install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

3. **Firewall**:
   ```bash
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

4. **API Authentication** (if needed):
   ```python
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @app.post("/api/analyze/single")
   async def analyze(request: Request, credentials = Depends(security)):
       # Verify token
       ...
   ```

## Troubleshooting

### Common Issues

**Issue**: Import errors
```bash
# Solution: Ensure all dependencies installed
pip install -r requirements.txt --upgrade
```

**Issue**: Database connection failed
```bash
# Solution: Check database exists and permissions
python -c "from backend.models import get_database_manager; db = get_database_manager(); db.create_all_tables()"
```

**Issue**: Frontend not loading
```bash
# Solution: Check CORS settings in backend/api.py
# Ensure allow_origins includes your frontend URL
```

**Issue**: Tests failing
```bash
# Solution: Install test dependencies
pip install pytest pytest-asyncio
# Run with verbose output
pytest -v --tb=short
```

## Scaling

### Horizontal Scaling

Use load balancer (Nginx) with multiple backend instances:

```nginx
upstream earthquake_api {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    location /api {
        proxy_pass http://earthquake_api;
    }
}
```

Start multiple instances:
```bash
python backend/api.py --port 8000 &
python backend/api.py --port 8001 &
python backend/api.py --port 8002 &
```

### Database Scaling

- Use PostgreSQL with replication
- Implement read replicas for queries
- Use connection pooling

## Support

- **GitHub Issues**: https://github.com/nbbulk-dotcom/Earthquake_Enhanced/issues
- **Documentation**: See README.md and TECHNICAL.md
- **API Docs**: http://your-domain.com/docs

---

**Deployment Status**: ✅ Ready for Production  
**Last Updated**: October 23, 2025  
**System Version**: 1.0.0
