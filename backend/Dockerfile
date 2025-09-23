# Railway-optimized Dockerfile with forced ONNX Runtime installation
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and ONNX Runtime
RUN apt-get update && apt-get install -y \
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
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install ONNX Runtime with multiple fallback methods
RUN pip install --no-cache-dir --force-reinstall --no-deps onnxruntime==1.19.2 || \
    pip install --no-cache-dir --force-reinstall onnxruntime==1.19.2 || \
    pip install --no-cache-dir onnxruntime==1.19.2 || \
    pip install --no-cache-dir onnxruntime-cpu==1.15.1

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Force reinstall ONNX Runtime again to ensure it's available
RUN pip install --no-cache-dir --force-reinstall onnxruntime==1.19.2

# Verify ONNX Runtime installation with detailed output
RUN python -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__); print('Available providers:', onnxruntime.get_available_providers()); print('Installation path:', onnxruntime.__file__)"

# Copy application code
COPY . .

# Copy Google Cloud credentials
COPY ocrtyre-9369d891cdc1.json /app/ocrtyre-9369d891cdc1.json

# Create models directory and copy YOLO model
RUN mkdir -p models
COPY models/Tyre_Detect.onnx /app/models/Tyre_Detect.onnx

# Run comprehensive ONNX Runtime test
RUN python test_onnx.py

# Final verification
RUN python -c "import sys; print('Python path:', sys.path); import onnxruntime; print('ONNX Runtime successfully imported')"

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ocrtyre-9369d891cdc1.json

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]