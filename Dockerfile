# Adam API Dockerfile
# Production-ready container for the Advanced RAG API

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including ODBC drivers for MS SQL Server
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gnupg2 \
    ca-certificates \
    unixodbc \
    unixodbc-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY airgapped_rag_advanced.py .
COPY parent_child_store.py .
COPY query_classifier.py .
COPY feedback_store.py .
COPY semantic_chunker.py .
COPY document_cleaner.py .
COPY metadata_extractor.py .
COPY ollama_client_lb.py .
COPY sql_routes.py .
COPY sql_query_handler.py .
COPY conversation_manager.py .
COPY run_advanced.py .

# Copy config directory
COPY config/ ./config/

# Create data directory
RUN mkdir -p /data/airgapped_rag/documents /data/airgapped_rag/chromadb_advanced

# Environment variables (can be overridden)
ENV DATA_DIR=/data/airgapped_rag
ENV OLLAMA_HOSTS=https://adam.amentumspacemissions.com:11433,https://adam.amentumspacemissions.com:11436
ENV LLM_MODEL=llama3:8b
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "run_advanced.py"]
