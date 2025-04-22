FROM python:3.10-slim

# Sustavske biblioteke potrebne za OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Radni direktorij
WORKDIR /app

# Kopiraj requirements i kod
COPY requirements.txt .
COPY server.py .

# Instaliraj pakete
RUN pip install --no-cache-dir -r requirements.txt

# Pokreni server
CMD ["python", "server.py"]
