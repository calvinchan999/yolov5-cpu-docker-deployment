# Build stage
FROM python:3.8-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Clone YOLOv5
RUN git clone https://github.com/ultralytics/yolov5.git

# Download YOLOv5n model
RUN wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
RUN wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt


# Install Python dependencies with CPU optimizations
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r yolov5/requirements.txt \
    && pip install --no-cache-dir paho-mqtt opencv-python-headless \
    && pip install --no-cache-dir intel-openmp mkl

# Final stage
FROM python:3.8-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy YOLOv5 from builder
COPY --from=builder /app/yolov5 ./yolov5
COPY --from=builder /app/yolov5n.pt .
COPY --from=builder /app/yolov5m.pt .
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages

# Copy your application
COPY human_yolo.py .

# Set environment variables for CPU optimization
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# Set Python path to include YOLOv5
ENV PYTHONPATH="/app:/app/yolov5:${PYTHONPATH}"

# Run the application
CMD ["python3", "human_yolo.py"] 