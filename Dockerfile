FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY production_requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r production_requirements.txt

# Application code
COPY . .

# Optional: Create data folder if used
RUN mkdir -p /app/data

# Expose Streamlit port
EXPOSE 8501

# Final CMD (most stable form)
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
