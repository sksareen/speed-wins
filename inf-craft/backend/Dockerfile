FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY web_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p saves

# Expose the port
EXPOSE 5001

# Set environment variables
ENV FLASK_APP=web_app/game_api.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "web_app/game_api.py"]
