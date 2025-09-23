# Railway-optimized Dockerfile with proper ONNX Runtime installation
FROM python:3.11-slim

# Install system dependencies for ONNX Runtime and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy Google Cloud credentials
COPY ocrtyre-9369d891cdc1.json /app/ocrtyre-9369d891cdc1.json

# Create models directory and copy YOLO model
RUN mkdir -p models
COPY models/Tyre_Detect.onnx /app/models/Tyre_Detect.onnx

# Test ONNX Runtime installation
RUN python -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)"

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ocrtyre-9369d891cdc1.json

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]