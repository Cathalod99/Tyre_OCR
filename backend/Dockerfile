FROM python:3.11-slim

# System deps for ONNX Runtime (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Workdir first
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install deps
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your app (includes backend/, models/, etc.)
COPY . .

# Test ONNX Runtime installation
RUN python -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)"

# Start: bind to Railway's $PORT
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]