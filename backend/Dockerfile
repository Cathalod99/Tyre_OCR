FROM python:3.11-slim

# Install libgomp for ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

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

# Bind to Railway's PORT environment variable
ENV PORT=8080

# Run the application - Railway will set the PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "0"]