# Railway Deployment Guide

## ✅ Current Production Setup

### **Active Files:**
- **`Dockerfile`** (root) - Production Railway deployment
- **`backend/requirements.txt`** - Python dependencies with ONNX Runtime
- **`backend/main.py`** - FastAPI application

### **Deprecated Files:**
- **`backend/requirements.final.txt`** - ❌ DEPRECATED (had ONNX Runtime commented out)

## 🚀 Railway Configuration

### **Dockerfile Order (Cache-Friendly):**
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1
WORKDIR /app
COPY backend/requirements.txt .          # ← Requirements first
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .                                # ← App code second
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 15"]
```

### **Key Features:**
- ✅ **ONNX Runtime 1.18.1** - Full YOLO detection
- ✅ **Railway Port Binding** - Uses `$PORT` environment variable
- ✅ **Production Settings** - Single worker, timeout-keep-alive
- ✅ **Image Enhancement** - Full `convert.py` pipeline
- ✅ **Structured Logging** - Request IDs and latency tracking

## 🔍 Verification Endpoints

### **Health Check:**
```bash
curl -s https://your-app.railway.app/health
```

### **Readiness Check:**
```bash
curl -s https://your-app.railway.app/ready
# Expected: {"status": "ready", "yolo_status": "available", "onnx": true}
```

### **ONNX Runtime Debug:**
```bash
curl -s https://your-app.railway.app/debug/onnx
# Expected: {"onnxruntime": "1.18.1", "available_providers": ["CPUExecutionProvider"]}
```

### **Python + ORT Debug:**
```bash
curl -s https://your-app.railway.app/debug/py
# Expected: {"python": "3.11.x", "onnxruntime": "1.18.1", "installed": true}
```

## 🎯 Expected Logs

### **Build Logs:**
```
ONNX Runtime version: 1.18.1
Available providers: ['CPUExecutionProvider']
```

### **Runtime Logs:**
```
level=info event=startup port=8080
INFO: ONNX Runtime version: 1.18.1
INFO: Available providers: ['CPUExecutionProvider']
INFO: YOLO model loaded successfully
INFO: Uvicorn running on http://0.0.0.0:8080
```

## 🚫 Troubleshooting

### **If ONNX Runtime Missing:**
1. Clear Railway build cache
2. Verify `backend/requirements.txt` contains `onnxruntime==1.18.1`
3. Check Dockerfile uses `backend/requirements.txt` (not `requirements.final.txt`)

### **If Port Mismatch:**
1. Verify Dockerfile uses shell CMD: `CMD ["sh", "-c", "..."]`
2. Check `--port ${PORT:-8000}` in CMD
3. Ensure Railway sets `$PORT` environment variable

## 📊 Complete Pipeline

1. **YOLO Detection** → Crop tire region
2. **Image Enhancement** → `convert.py` warpPolar function
3. **OCR Processing** → Google Cloud Vision API
4. **ML Processing** → Extract tire data (make, model, size, DOT)
5. **Plant Lookup** → Factory and country information
6. **JSON Response** → Structured tire data

## 🎯 Production Ready Features

- ✅ **Full YOLO Detection** - Tire region detection
- ✅ **Image Enhancement** - Perspective transform for better OCR
- ✅ **Google Cloud Vision** - High-quality OCR
- ✅ **ML Text Processing** - Tire data extraction
- ✅ **Plant Code Lookup** - Factory information
- ✅ **Structured Logging** - Request tracing and latency
- ✅ **Security Checks** - File type and size validation
- ✅ **Graceful Shutdown** - ONNX session cleanup
- ✅ **Health Monitoring** - `/health` and `/ready` endpoints
