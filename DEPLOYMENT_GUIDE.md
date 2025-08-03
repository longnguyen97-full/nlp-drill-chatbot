# 🚀 DEPLOYMENT GUIDE - LawBot

## 📋 **Tổng quan**

Hướng dẫn triển khai LawBot trong môi trường production với các tối ưu hóa về hiệu suất và bảo mật.

## 🎯 **Môi trường Production**

### **Requirements:**

- **OS:** Ubuntu 20.04+ / CentOS 8+ / Windows Server 2019+
- **Python:** 3.8+
- **RAM:** Tối thiểu 8GB, khuyến nghị 16GB+
- **GPU:** NVIDIA GPU với CUDA 11.0+ (khuyến nghị)
- **Storage:** Tối thiểu 10GB cho models và data
- **Network:** Stable internet connection

### **System Dependencies:**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y build-essential git curl wget

# CUDA (nếu có GPU)
sudo apt install -y nvidia-cuda-toolkit

# CentOS/RHEL
sudo yum update
sudo yum install -y python3 python3-pip git curl wget
sudo yum groupinstall -y "Development Tools"
```

## 🔧 **Cài đặt Production**

### **1. Clone và Setup**

```bash
# Clone repository
git clone https://github.com/lawbot-team/lawbot.git
cd lawbot

# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate

# Cài đặt dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Cài đặt như package
pip install -e .
```

### **2. Cấu hình Environment**

```bash
# Tạo file .env
cat > .env << EOF
# Environment
LAWBOT_ENV=production
LAWBOT_DEBUG=false

# Directories
LAWBOT_DATA_DIR=/opt/lawbot/data
LAWBOT_MODELS_DIR=/opt/lawbot/models
LAWBOT_INDEXES_DIR=/opt/lawbot/indexes
LAWBOT_REPORTS_DIR=/opt/lawbot/reports
LAWBOT_LOGS_DIR=/opt/lawbot/logs

# Performance
LAWBOT_BI_ENCODER_BATCH_SIZE=8
LAWBOT_CROSS_ENCODER_BATCH_SIZE=4
LAWBOT_TOP_K_RETRIEVAL=100
LAWBOT_TOP_K_FINAL=5

# GPU Settings
LAWBOT_FP16_TRAINING=true
CUDA_VISIBLE_DEVICES=0
EOF

# Load environment variables
source .env
```

### **3. Tạo thư mục cần thiết**

```bash
# Tạo thư mục
sudo mkdir -p /opt/lawbot/{data,models,indexes,reports,logs}
sudo chown -R $USER:$USER /opt/lawbot

# Copy data files
cp -r data/* /opt/lawbot/data/
```

### **4. Train Models (nếu chưa có)**

```bash
# Chạy toàn bộ pipeline training
python run_pipeline.py --skip-filtering

# Hoặc chạy từng bước
python scripts/09_train_bi_encoder.py
python scripts/10_build_faiss_index.py
python scripts/11_train_cross_encoder.py
```

## 🐳 **Docker Deployment**

### **1. Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /opt/lawbot/{data,models,indexes,reports,logs}

# Set environment variables
ENV LAWBOT_ENV=production
ENV LAWBOT_DEBUG=false
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app/app.py"]
```

### **2. Docker Compose**

```yaml
# docker-compose.yml
version: '3.8'

services:
  lawbot:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/opt/lawbot/data
      - ./models:/opt/lawbot/models
      - ./indexes:/opt/lawbot/indexes
      - ./logs:/opt/lawbot/logs
    environment:
      - LAWBOT_ENV=production
      - LAWBOT_DEBUG=false
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - lawbot
    restart: unless-stopped
```

### **3. Build và Deploy**

```bash
# Build image
docker build -t lawbot:latest .

# Run với Docker Compose
docker-compose up -d

# Kiểm tra logs
docker-compose logs -f lawbot
```

## 🔒 **Security Configuration**

### **1. Firewall Setup**

```bash
# Ubuntu/Debian
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### **2. SSL Certificate**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### **3. Environment Security**

```bash
# Tạo user riêng cho application
sudo useradd -r -s /bin/false lawbot

# Set permissions
sudo chown -R lawbot:lawbot /opt/lawbot
sudo chmod -R 755 /opt/lawbot
```

## 📊 **Monitoring & Logging**

### **1. Logging Configuration**

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_production_logging():
    # File handler với rotation
    file_handler = RotatingFileHandler(
        '/opt/lawbot/logs/lawbot.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Root logger
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)
```

### **2. Health Check**

```python
# health_check.py
import requests
import time

def health_check():
    try:
        response = requests.get('http://localhost:8000/health')
        return response.status_code == 200
    except:
        return False

def monitor_health():
    while True:
        if not health_check():
            # Send alert
            send_alert("LawBot is down!")
        time.sleep(60)
```

### **3. Performance Monitoring**

```python
# monitoring.py
import psutil
import time

def monitor_resources():
    while True:
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # GPU usage (nếu có)
        gpu_usage = get_gpu_usage()
        
        # Log metrics
        log_metrics({
            'cpu': cpu_percent,
            'memory': memory.percent,
            'gpu': gpu_usage
        })
        
        time.sleep(30)
```

## ⚡ **Performance Optimization**

### **1. Model Caching**

```python
# cache_manager.py
import pickle
import os
from functools import lru_cache

class ModelCache:
    def __init__(self, cache_dir="/opt/lawbot/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    @lru_cache(maxsize=1000)
    def get_cached_result(self, query_hash):
        cache_file = os.path.join(self.cache_dir, f"{query_hash}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def cache_result(self, query_hash, result):
        cache_file = os.path.join(self.cache_dir, f"{query_hash}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
```

### **2. Batch Processing**

```python
# batch_processor.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, queries):
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self.process_single, query)
            for query in queries
        ]
        return await asyncio.gather(*tasks)
    
    def process_single(self, query):
        # Process single query
        return pipeline.predict(query)
```

### **3. Memory Optimization**

```python
# memory_manager.py
import gc
import torch

class MemoryManager:
    @staticmethod
    def clear_cache():
        """Clear PyTorch cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def monitor_memory():
        """Monitor memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
        return psutil.virtual_memory().percent
```

## 🔄 **CI/CD Pipeline**

### **1. GitHub Actions**

```yaml
# .github/workflows/deploy.yml
name: Deploy LawBot

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to server
      uses: appleboy/ssh-action@v0.1.4
      with:
        host: ${{ secrets.HOST }}
        username: ${{ secrets.USERNAME }}
        key: ${{ secrets.KEY }}
        script: |
          cd /opt/lawbot
          git pull origin main
          source venv/bin/activate
          pip install -r requirements.txt
          sudo systemctl restart lawbot
```

### **2. Systemd Service**

```ini
# /etc/systemd/system/lawbot.service
[Unit]
Description=LawBot Legal QA Service
After=network.target

[Service]
Type=simple
User=lawbot
WorkingDirectory=/opt/lawbot
Environment=PATH=/opt/lawbot/venv/bin
ExecStart=/opt/lawbot/venv/bin/python app/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 📈 **Scaling**

### **1. Load Balancer**

```nginx
# nginx.conf
upstream lawbot_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://lawbot_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### **2. Horizontal Scaling**

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  lawbot:
    build: .
    ports:
      - "8000-8002:8000"
    deploy:
      replicas: 3
    volumes:
      - shared_models:/opt/lawbot/models
      - shared_indexes:/opt/lawbot/indexes

volumes:
  shared_models:
  shared_indexes:
```

## 🚨 **Troubleshooting**

### **Common Issues:**

#### **1. Out of Memory**
```bash
# Giảm batch size
export LAWBOT_BI_ENCODER_BATCH_SIZE=2
export LAWBOT_CROSS_ENCODER_BATCH_SIZE=1

# Hoặc tăng swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### **2. Slow Response Time**
```bash
# Kiểm tra GPU usage
nvidia-smi

# Tối ưu FAISS index
python scripts/optimize_faiss_index.py

# Cache models
python scripts/cache_models.py
```

#### **3. Service Not Starting**
```bash
# Kiểm tra logs
sudo journalctl -u lawbot -f

# Kiểm tra permissions
sudo chown -R lawbot:lawbot /opt/lawbot

# Restart service
sudo systemctl restart lawbot
```

## 📞 **Support**

- **Documentation:** [https://lawbot.readthedocs.io/](https://lawbot.readthedocs.io/)
- **Issues:** [https://github.com/lawbot-team/lawbot/issues](https://github.com/lawbot-team/lawbot/issues)
- **Email:** lawbot@example.com

---

**Happy Deploying! 🚀** 