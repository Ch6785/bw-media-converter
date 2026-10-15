# Use an official lightweight Python image
FROM python:3.12-slim

# Install system dependencies including FFmpeg and required OpenCV C-libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirement list and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose port 5000 (standard for Flask / Gunicorn)
EXPOSE 5000

# Start Gunicorn server binding to PORT environment variable provided by Render
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app:app"]