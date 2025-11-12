# ADAM RAG Application - Architecture & AWS Deployment Analysis

## Executive Summary

**Adam** is a production-grade, air-gapped Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and Ollama. It's designed for enterprise document intelligence, supporting hybrid search (BM25 + semantic embeddings), parent-child chunking, and local LLM inference.

### Current Architecture: On-Premises with Remote Services
- **API Server**: FastAPI container (8000)
- **Embedding Model**: Sentence Transformers (e5-large-v2, 1024 dims)
- **LLM**: Ollama (running on remote server at `adam.amentumspacemissions.com:11434`)
- **Vector Database**: ChromaDB (local, persistent)
- **SQL Backend**: MS SQL Server (Corporate database)

---

## 1. MODEL LOADING & LLM USAGE

### Embedding Models
**File**: `/home/user/Adam-api/parent_child_store.py` (Line 182)

```python
embedding_model: str = "intfloat/e5-large-v2"  # 1024 dimensions
```

**Key Details**:
- **Model**: e5-large-v2 (Hugging Face sentence-transformers)
- **Size**: ~1.3GB (downloaded on first run)
- **Dimensions**: 1024 (compared to smaller alternatives like all-MiniLM-L6-v2 with 384 dims)
- **GPU Auto-Detection**: Uses CUDA if available, falls back to CPU
  ```python
  device = "cuda" if torch.cuda.is_available() else "cpu"
  self.embedding_model = SentenceTransformer(embedding_model, device=device)
  ```
- **Query Prefix**: e5 models require "query: " prefix for optimal retrieval
  ```python
  if "e5" in self.embedding_model_name.lower():
      query_text = f"query: {expanded_query}"
  ```
- **Batch Encoding**: Generates embeddings in batches (size=32) for efficiency

### Large Language Models (Ollama)

**Configuration Files**:
- `airgapped_rag_advanced.py` (Lines 64-72)
- `metadata_extractor.py` (Lines 66-72)
- `sql_routes.py` (Lines 23-25)

**Current Configuration**:
```python
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://adam.amentumspacemissions.com:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small:22b")
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "16384"))  # 16K tokens
```

**Model Information**:
- **Primary Model**: Mistral Small 22B (excellent for RAG tasks)
- **VRAM Required**: ~22GB (system currently uses 2 GPUs with 32GB total VRAM)
- **Context Window**: 16,384 tokens (out of 128K max supported)
- **Temperature**: 0.3 (default, for consistency)
- **Max Predictions**: 2000 tokens per response

**Usage Points**:
1. **Metadata Extraction**: Ollama extracts document metadata (type, topics, keywords, departments)
2. **Answer Generation**: Generates final answers with citations
3. **SQL Query Generation**: Converts natural language to SQL
4. **Query Classification**: Detects system vs. document queries
5. **Document Selection**: Uses LLM for ranking retrieved documents
6. **Follow-up Questions**: Generates suggested follow-up queries

**Ollama Integration**:
```python
import ollama
self.ollama_client = ollama.Client(host=OLLAMA_HOST)
response = self.ollama_client.generate(
    model=LLM_MODEL,
    prompt=prompt,
    options={"temperature": 0.3, "num_predict": 2000, "num_ctx": 16384},
    stream=True
)
```

---

## 2. CONFIGURATION MANAGEMENT

### Environment Variables

**Core Configuration**:
| Variable | Default | Purpose | File(s) |
|----------|---------|---------|---------|
| `DATA_DIR` | `/data/airgapped_rag` | Root data directory | All modules |
| `OLLAMA_HOST` | `http://adam.amentumspacemissions.com:11434` | Ollama service endpoint | airgapped_rag_advanced.py, sql_routes.py, metadata_extractor.py |
| `LLM_MODEL` | `mistral-small:22b` | Primary LLM to use | airgapped_rag_advanced.py, sql_routes.py |
| `LLM_CONTEXT_WINDOW` | `16384` | Context window in tokens | airgapped_rag_advanced.py |
| `PYTHONUNBUFFERED` | `1` | Ensures streaming responses | run_advanced.py |

**Database Configuration** (SQL):
| Variable | Location | Purpose |
|----------|----------|---------|
| `EMPLOYEE_DB_PASSWORD` | Environment | MS SQL password (not in config) |
| Database host/user | `config/databases.yaml` | Hardcoded connection details |

**Hardcoded Server Addresses**:
```python
# Line 66 in airgapped_rag_advanced.py
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://adam.amentumspacemissions.com:11434")

# Line 69 in metadata_extractor.py
ollama_host: str = "http://adam.amentumspacemissions.com:11434"
```

**CRITICAL ISSUE**: The metadata_extractor.py has the production server hardcoded as a default parameter.

### Configuration Files

**1. `/config/databases.yaml`** - SQL Database Configuration
```yaml
employee_directory:
  enabled: true
  database_type: "mssql"
  connection:
    server: "CLGDBS02"          # HARDCODED SERVER
    port: 1433
    database: "Corporate"
    user: "svcasm-adamAi"        # HARDCODED USER
    driver: "ODBC Driver 17 for SQL Server"
    use_windows_auth: true       # Windows Auth enabled
  schema:
    vwPersonnelAll:
      columns: [...20 columns including salary...]
```

**2. `/config/acronyms.json`** - Domain-specific acronym mappings
- Maps acronyms (PTO, HR, IT, CUI, PII) to full forms
- Helps embedding models understand domain terminology
- Auto-loaded and expandable via API

**3. `/cleaning_config.yaml`** - Document cleaning patterns
- 30+ regex patterns for removing noise
- Amentum-specific patterns for security banners
- CUI/FOUO/confidential markings
- Page numbers, signatures, boilerplate

---

## 3. DATABASE CONNECTIONS & VECTOR STORE

### Vector Database: ChromaDB

**Location**: `/data/airgapped_rag/chromadb_advanced/`

**Dual Collection Strategy**:
```python
# From parent_child_store.py
self.child_collection = self.client.get_or_create_collection(
    name="child_chunks",
    metadata={"description": "Small chunks for precise retrieval"}
)
self.parent_collection = self.client.get_or_create_collection(
    name="parent_chunks",
    metadata={"description": "Large chunks for LLM context"}
)
```

**Metadata Stored per Chunk**:
```python
{
    "document_id": str,
    "source_url": str,
    "document_title": str,
    "document_type": str,
    "summary": str,
    "primary_topics": str,
    "keywords": str,
    "departments": str,
    "answerable_questions": str,
    "section_title": str,
    "section_number": str,
    "ingestion_date": str,
    "extraction_confidence": float,
    "parent_id": str,           # Child->Parent reference
    "chunk_index": int,
    "good_count": int,          # Feedback-based weighting
    "bad_count": int
}
```

**Embedding Features**:
- Only child chunks have embeddings (parents are context)
- Embeddings generated with Sentence Transformers (batch size 32)
- e5-large-v2 model with "query: " prefix for searches
- BM25 hybrid search with semantic reranking

### SQL Database Connections

**MS SQL Server Backend** (`config/databases.yaml`):
```python
# Connection handled via pyodbc in sql_query_handler.py
import pyodbc

conn_str = (
    f"DRIVER={driver};"
    f"SERVER={server};"
    f"PORT={port};"
    f"DATABASE={database};"
    f"UID={user};"
    f"PWD={password};"
)
conn = pyodbc.connect(conn_str, timeout=timeout)
```

**Configured Databases**:
1. **employee_directory** (ENABLED)
   - Server: `CLGDBS02`
   - Database: `Corporate`
   - User: `svcasm-adamAi`
   - Table: `vwPersonnelAll` (personnel view)
   - Max rows: 1000
   - Query timeout: 30 seconds

2. **purchasing_system** (DISABLED)
   - Configured but not implemented

### Local SQLite Databases

**Feedback Storage** (`/data/airgapped_rag/feedback.db`):
```python
# Tables:
# - feedback: Query -> Answer -> Feedback mapping
# - chunk_quality: Chunk scoring based on feedback
```

**Conversation History** (`/data/airgapped_rag/conversations.db`):
```python
# Tables:
# - conversations: Conversation sessions
# - messages: Individual messages with context
```

---

## 4. API STRUCTURE & ENDPOINTS

### Main Application
**Entry Point**: `airgapped_rag_advanced.py` (67KB)

### Core Endpoints

#### Document Management
```
POST   /upload-document          - Upload & process PDF
GET    /documents                - List all indexed documents
PUT    /documents/{id}/questions - Update answerable questions
DELETE /documents/{id}           - Delete specific document
DELETE /documents                - Clear all documents
GET    /debug/document/{id}      - Debug document contents
```

#### Query & Retrieval
```
POST   /query                    - Synchronous question answering
POST   /query-stream             - Streaming response (SSE format)
GET    /documents                - List documents with stats
```

#### System Management
```
GET    /health                   - Health check endpoint
GET    /statistics               - System statistics
GET    /acronyms                 - List acronym mappings
PUT    /acronyms                 - Update acronym mappings
```

#### SQL Query Endpoints
```
POST   /query-employee           - Query employee directory (natural language)
```

#### Debug & Testing
```
POST   /debug/extract-markdown   - Test PDF extraction with Docling
```

### Request/Response Models

**Query Request**:
```python
class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 30                    # Initial retrieval candidate count
    parent_limit: int = 5               # Max parent chunks for LLM
    temperature: float = 0.3            # LLM temperature
    metadata_filter: Optional[Dict] = None
    use_hybrid: bool = True             # BM25 + semantic search
    bm25_weight: float = 0.2            # Weight distribution (0.2 = 80% semantic)
    use_llm_selection: bool = False     # Deprecated
```

**Query Response**:
```python
class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict]               # Document citations with URLs
    retrieval_stats: Dict               # Retrieval metadata
    suggested_followups: Optional[List[str]]
```

**Citations Format**:
```python
{
    "source_url": str,
    "document_title": str,
    "section_title": str,
    "section_number": str,
    "excerpt": str                      # First 500 chars
}
```

---

## 5. HARDCODED PATHS & SERVER-SPECIFIC CONFIGURATIONS

### Critical Hardcoded Values

**1. Ollama Server Hardcoding**
```python
# metadata_extractor.py, Line 69
def __init__(
    self,
    model_name: str = "mistral-small:22b",
    ollama_host: str = "http://adam.amentumspacemissions.com:11434",  # HARDCODED!
    ...
):
```

**Impact**: If Ollama host changes, this default must be overridden via parameter.

**2. SQL Server Hardcoding**
```yaml
# config/databases.yaml
connection:
    server: "CLGDBS02"              # HARDCODED
    database: "Corporate"
    user: "svcasm-adamAi"            # HARDCODED
```

**Impact**: Cannot connect to different SQL servers without editing YAML.

**3. Data Directory Paths**
```python
# All modules use /data/airgapped_rag
DATA_DIR = Path(os.getenv("DATA_DIR", "/data/airgapped_rag"))
DOCS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chromadb_advanced"
```

**4. Docling/PDF Processing**
- Requires `poppler-utils` and `tesseract-ocr` (system packages)
- Downloaded on first use via `DocumentConverter()`
- No explicit model path configuration

---

## 6. GPU & COMPUTE RESOURCE DEPENDENCIES

### GPU Requirements

**Embedding Model (e5-large-v2)**:
- **GPU Memory**: 2-3GB (with batch processing)
- **CPU Fallback**: Supported (much slower - 10-50x)
- **Auto-Detection**: 
  ```python
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ```

**PyTorch Configuration**:
```txt
torch==2.5.1+cu118           # CUDA 11.8 support
torchvision==0.16.0+cu118
torchaudio==2.1.0+cu118
```

### CPU Requirements
- **Embedding Generation**: Multi-threaded batch processing
- **Document Processing**: Docling uses CPU-intensive PDF parsing
- **BM25 Search**: Full CPU task
- **Recommendation**: 8+ cores for document ingestion

### Memory Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| e5-large-v2 (GPU) | 2-3GB VRAM | Batch size 32 |
| e5-large-v2 (CPU) | 4-5GB RAM | Single threaded |
| ChromaDB | 1-2GB RAM | In-memory + disk |
| Docling | 2GB RAM | Large documents |
| Base Python | 1-2GB | All libraries |
| **Total** | **8-15GB minimum** | **32GB recommended** |

### LLM (Ollama) Requirements
- **Mistral Small 22B**: ~22GB VRAM
- **Llama 3 8B**: ~8GB VRAM
- **Location**: Remote server (not co-located)
- **Current Setup**: Remote Ollama on `adam.amentumspacemissions.com` with 32GB VRAM (2 GPUs)

---

## 7. DOCKER & DEPLOYMENT CONFIGURATIONS

### Dockerfiles Available

**1. `Dockerfile` (production)**
- Python 3.11-slim base
- Requires curl for health checks
- No system packages for PDF processing

**2. `Dockerfile.advanced` (full-featured)**
- Python 3.10-slim base
- Includes: poppler-utils, tesseract-ocr, tesseract-ocr-eng
- NLTK data download for punkt_tab
- Better for document processing

**3. `Dockerfile.airgapped` (air-gapped production)**
- Same as Dockerfile.advanced
- Designed for RHEL9 deployment
- All dependencies pre-cached

### Docker Compose Files

**1. `docker-compose.yml` (All-in-one)**
```yaml
- adam-api (port 8000)
- ollama (port 11434)
- Networks: adam-network
- Volumes: adam-data, ollama-data
```

**2. `docker-compose.advanced.yml` (API-only)**
```yaml
- rag-api-advanced (port 8000)
- Connects to external Ollama on host.docker.internal:11434
- Volume: rag_data
```

**3. `docker-compose.airgapped.yml` (Production RHEL9)**
```yaml
Services:
  - ollama
  - airgapped-rag-api
Volumes:
  - ollama_models
  - airgapped_data
Networks:
  - airgapped-rag-network
```

**4. `docker-compose.ollama.yml` (Ollama standalone)**
- Just Ollama service
- For when Ollama runs separately

### Environment Variables in Containers

```dockerfile
# From all Dockerfiles
ENV DATA_DIR=/data/airgapped_rag
ENV OLLAMA_HOST=http://ollama:11434    # or localhost
ENV LLM_MODEL=llama3:8b                 # or mistral-small:22b
ENV PYTHONUNBUFFERED=1
```

### Health Checks
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

---

## 8. DEPENDENCIES & REQUIREMENTS

### Python Dependencies
**File**: `requirements.txt` (35 packages)

**Key packages**:
```txt
# FastAPI & Web
fastapi==0.110.1
uvicorn[standard]==0.27.1
python-multipart

# AI/ML Core
torch==2.5.1+cu118              # GPU-accelerated
sentence-transformers==3.2.0    # Embeddings
transformers==4.48.1            # BERT/models
ollama>=0.1.0                   # LLM client

# Document Processing
docling>=2.8,<3                 # PDF extraction
nltk>=3.8.1                     # NLP

# Vector Database
chromadb>=0.4.0                 # Vector DB
rank_bm25==0.2.2                # BM25 search
rapidfuzz==3.9.2                # Fuzzy matching

# SQL & Databases
pyodbc>=4.0.39                  # ODBC driver
pyyaml>=6.0                     # Config parsing

# Utilities
numpy==1.26.4
huggingface_hub==0.25.2
safetensors==0.4.5
accelerate==0.34.2
```

### System Dependencies
```bash
# From Dockerfile.advanced
poppler-utils           # PDF rendering
tesseract-ocr          # OCR engine
tesseract-ocr-eng      # English language data
curl                   # Health checks
build-essential        # C compiler
```

### External Services
1. **Ollama**: Remote LLM inference service
2. **HuggingFace Hub**: Model downloads (e5-large-v2, ~1.3GB)
3. **MS SQL Server**: Employee directory database

---

## 9. KEY COMPONENTS & FILE OVERVIEW

### Main Application Modules

| File | Lines | Purpose |
|------|-------|---------|
| `airgapped_rag_advanced.py` | 1700+ | Main FastAPI app, pipeline orchestration |
| `parent_child_store.py` | 700+ | ChromaDB management, hybrid search, BM25 |
| `sql_query_handler.py` | 1200+ | Text-to-SQL, query execution |
| `semantic_chunker.py` | 500+ | Document chunking with parent-child relationships |
| `metadata_extractor.py` | 600+ | LLM-based metadata extraction |
| `document_cleaner.py` | 350+ | Noise removal (regex patterns) |
| `sql_routes.py` | 600+ | SQL query API endpoints |
| `conversation_manager.py` | 400+ | Conversation history tracking |
| `feedback_store.py` | 400+ | User feedback storage & analytics |
| `query_classifier.py` | 400+ | System vs. document query detection |

### Supporting Utilities
- `run_advanced.py`: Entry point with configuration printing
- `verify_gpu_setup.py`: GPU capability detection
- `diagnose_gpu.py`: Detailed GPU diagnostics
- `reembed_clean_chunks.py`: Re-embedding utility
- `restore_parent_metadata.py`: ChromaDB recovery tool

---

## AWS DEPLOYMENT RECOMMENDATIONS

### 1. CRITICAL MODIFICATIONS FOR AWS

#### A. Model Loading & Service Architecture

**Current State**: Remote Ollama (hardcoded to `adam.amentumspacemissions.com`)

**AWS Options**:

**Option 1: SageMaker Endpoint** (Recommended for large models)
```python
# Instead of ollama.Client()
import boto3
sagemaker_client = boto3.client('sagemaker-runtime', region_name='us-east-1')

response = sagemaker_client.invoke_endpoint(
    EndpointName='mistral-small-endpoint',
    ContentType='application/json',
    Body=json.dumps({"inputs": prompt, "parameters": {...}})
)
```

**Benefits**:
- Auto-scaling based on load
- Managed infrastructure
- Integration with IAM
- Cost optimization with spot instances

**Option 2: EC2 with Ollama** (Lower cost, simpler)
```python
OLLAMA_HOST = os.getenv("OLLAMA_HOST", 
    "http://ollama.internal.example.com:11434"
)
# Keep existing code, just change hostname
```

**Option 3: ECS Fargate Containers**
```yaml
# Sidecar pattern
containers:
  - name: rag-api
    image: adam-rag:latest
    environment:
      - OLLAMA_HOST=http://localhost:11434
      
  - name: ollama
    image: ollama/ollama:latest
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
```

#### B. Configuration Management

**Current Issue**: Hardcoded server names in code

**AWS Solution: AWS Systems Manager Parameter Store**

```python
import boto3

ssm = boto3.client('ssm')

def get_config(param_name):
    response = ssm.get_parameter(
        Name=f'/adam-rag/{param_name}',
        WithDecryption=True
    )
    return response['Parameter']['Value']

OLLAMA_HOST = get_config('ollama_host')
LLM_MODEL = get_config('llm_model')
DB_SERVER = get_config('db_server')
DB_PASSWORD = get_config('db_password')
```

**Setup**:
```bash
aws ssm put-parameter \
  --name /adam-rag/ollama_host \
  --value "http://ollama-internal:11434" \
  --type String

aws ssm put-parameter \
  --name /adam-rag/db_password \
  --value "xxxxx" \
  --type SecureString
```

**Or: AWS Secrets Manager for sensitive data**

```python
import json
from botocore.exceptions import ClientError

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])
    except ClientError as e:
        raise e

db_config = get_secret('adam-rag/database')
```

#### C. Vector Database (ChromaDB) Migration

**Current**: Local ChromaDB at `/data/airgapped_rag/chromadb_advanced/`

**AWS Options**:

**Option 1: RDS PostgreSQL + pgvector** (Best for teams)
```python
from pgvector.psycopg2 import register_vector
import psycopg2

# Migrate ChromaDB to PostgreSQL
conn = psycopg2.connect(
    host="adam-rag-db.xxxxx.rds.amazonaws.com",
    database="chromadb",
    user="postgres",
    password=get_secret('db_password')
)
```

**Option 2: DynamoDB** (Serverless, fully managed)
```python
import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('ChromaDB-Embeddings')

response = table.query(
    KeyConditionExpression=Key('document_id').eq(doc_id)
)
```

**Option 3: OpenSearch** (Search-optimized)
```python
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{'host': 'adam-rag-opensearch.amazonaws.com', 'port': 443}],
    http_auth=('username', 'password'),
    use_ssl=True,
    verify_certs=True
)
```

**Option 4: EFS for Persistent Storage** (Simplest migration)
```yaml
# Use existing ChromaDB, mount EFS
volumes:
  chromadb:
    driver: nfs
    driver_opts:
      o: "addr=adam-rag-efs.xxxxx.efs.amazonaws.com,vers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2"
      device: ":/cromaddb_advanced"
```

#### D. SQL Server Connection

**Current**: Hardcoded ODBC to `CLGDBS02`

**AWS Options**:

**Option 1: AWS RDS SQL Server**
```yaml
# Update config/databases.yaml
connection:
  server: "adam-rag-sqlserver.xxxxx.rds.amazonaws.com"
  port: 1433
  database: "Corporate"
  user: "svcasm-adamAi"
  password_env: "EMPLOYEE_DB_PASSWORD"  # Store in Secrets Manager
```

**Option 2: AWS DMS** (Database Migration Service)
- Migrate on-prem SQL Server to RDS
- Full managed replication

**Option 3: VPN + Private Link**
```python
# Keep on-premises server, access via VPN
# Update DNS in config to use VPN hostname
```

#### E. Document Storage

**Current**: `/data/airgapped_rag/documents/`

**AWS Solution: S3 + EFS Hybrid**

```python
import boto3

s3 = boto3.client('s3')

async def upload_document(file: UploadFile):
    # Save to local EFS
    file_path = DOCS_DIR / file.filename
    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    # Also upload to S3 for backup/compliance
    s3.upload_file(
        file_path,
        'adam-rag-documents',
        f'documents/{file.filename}'
    )
```

### 2. DEPLOYMENT ARCHITECTURE FOR AWS

#### Option A: ECS Fargate (Recommended for scalability)

```yaml
# ECS Task Definition
containers:
  - name: rag-api
    image: 123456789.dkr.ecr.us-east-1.amazonaws.com/adam-rag:latest
    cpu: 2048
    memory: 4096
    portMappings:
      - containerPort: 8000
    environment:
      - DATA_DIR=/data
      - OLLAMA_HOST=http://ollama-svc:11434
      - LLM_MODEL=mistral-small:22b
    volumeMounts:
      - sourceVolume: chromadb_data
        containerPath: /data
    logConfiguration:
      logDriver: awslogs
      options:
        awslogs-group: /ecs/adam-rag
        awslogs-stream-prefix: ecs
        
  - name: ollama
    image: ollama/ollama:latest
    cpu: 4096
    memory: 24576
    gpuCount: 2
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    volumeMounts:
      - sourceVolume: ollama_models
        containerPath: /root/.ollama

volumes:
  chromadb_data:
    efs:
      fileSystemId: fs-xxxxx
      rootDirectory: /chromadb
  ollama_models:
    efs:
      fileSystemId: fs-xxxxx
      rootDirectory: /ollama_models
```

**Cost**: ~$500-1000/month for dual-GPU instance

#### Option B: Lambda + API Gateway (For light workloads)

**Issue**: Max 10GB memory, 15min timeout - won't work for large models

**Viable only for**: Query endpoint on pre-embedded documents

#### Option C: Self-Managed EC2 (Cost-effective)

```bash
# Launch EC2 with GPU
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.2xlarge \
  --key-name adam-rag-key \
  --security-groups adam-rag-sg
```

**Cost**: ~$1.5/hour (~$1100/month reserved)

### 3. REQUIRED CODE MODIFICATIONS

#### Modification 1: Environment Configuration

**File**: `airgapped_rag_advanced.py`

```python
# BEFORE
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://adam.amentumspacemissions.com:11434")

# AFTER
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
if not OLLAMA_HOST:
    if os.getenv("AWS_REGION"):
        # AWS deployment
        ssm = boto3.client('ssm')
        OLLAMA_HOST = ssm.get_parameter(
            Name='/adam-rag/ollama_host'
        )['Parameter']['Value']
    else:
        # Local/on-prem fallback
        OLLAMA_HOST = "http://localhost:11434"
```

#### Modification 2: Database Configuration

**File**: `config/databases.yaml`

```yaml
# Use environment variables instead of hardcoded values
employee_directory:
  connection:
    server: ${SQLSERVER_HOST:-CLGDBS02}
    port: ${SQLSERVER_PORT:-1433}
    database: ${SQLSERVER_DB:-Corporate}
    user: ${SQLSERVER_USER:-svcasm-adamAi}
```

**Code to support this**:

```python
# sql_query_handler.py
import os
import yaml

def load_config_with_env(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Recursively replace ${VAR:-default} with env values
    def replace_env_vars(obj):
        if isinstance(obj, dict):
            return {k: replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, str):
            import re
            pattern = r'\$\{(\w+):-([^}]+)\}'
            return re.sub(
                pattern,
                lambda m: os.getenv(m.group(1), m.group(2)),
                obj
            )
        return obj
    
    return replace_env_vars(config)
```

#### Modification 3: Model Handling for Larger Models

**File**: `parent_child_store.py`

```python
# For handling model quantization in memory-constrained environments
def get_embedding_model(model_name: str, device: str):
    """
    Load embedding model with optional quantization for AWS.
    """
    # Use 8-bit quantization if in AWS and low memory
    load_in_8bit = (
        os.getenv("AWS_REGION") and 
        os.getenv("QUANTIZE_EMBEDDINGS", "false") == "true"
    )
    
    if load_in_8bit:
        model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"load_in_8bit": True}
        )
    else:
        model = SentenceTransformer(model_name, device=device)
    
    return model
```

#### Modification 4: Add CloudWatch Logging

**File**: `airgapped_rag_advanced.py`

```python
import logging
import json
from pythonjsonlogger import jsonlogger

# For CloudWatch JSON parsing
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

# Structured logging
logger.info("query_executed", extra={
    "query": question,
    "doc_count": len(citations),
    "latency_ms": elapsed_time * 1000
})
```

### 4. COST ANALYSIS

#### Current On-Premises Setup
- Remote Ollama server: $0/month (existing hardware)
- Data storage: $0/month (existing infrastructure)
- **Total**: ~$0/month (sunk costs)

#### AWS Setup Comparison

| Component | ECS Fargate | EC2 Spot | SageMaker |
|-----------|-----------|----------|-----------|
| **Ollama (GPU)** | $600/mo | $800/mo | $1200/mo |
| **RAG API** | $150/mo | $50/mo | Incl. |
| **EFS Storage** | $150/mo | $150/mo | $150/mo |
| **RDS (Optional)** | $300/mo | $300/mo | $0/mo |
| **Data Transfer** | $50/mo | $50/mo | $100/mo |
| **Monitoring** | $30/mo | $30/mo | $30/mo |
| **Total** | **$1,280/mo** | **$1,380/mo** | **$1,480/mo** |

**Cost-saving recommendations**:
1. Use **reserved instances** for 50-60% savings
2. Use **spot instances** for non-critical LLM inference
3. Implement **request batching** to reduce API calls
4. Cache embeddings in **CloudFront** for repeated queries

### 5. MIGRATION CHECKLIST

- [ ] Extract hardcoded values to environment variables
- [ ] Update Ollama connection to use Parameter Store
- [ ] Migrate SQL Server credentials to Secrets Manager
- [ ] Plan Vector DB migration (PostgreSQL + pgvector recommended)
- [ ] Set up EFS for persistent document storage
- [ ] Create ECR repository and build Docker images
- [ ] Define ECS task definitions with GPU support
- [ ] Set up CloudWatch logging and monitoring
- [ ] Create RDS instance if using managed SQL Server
- [ ] Configure S3 for document backups
- [ ] Set up IAM roles for ECS tasks
- [ ] Implement health check endpoints for ELB
- [ ] Load test with actual document volumes
- [ ] Plan data migration strategy
- [ ] Document deployment procedures

---

## Summary & Recommendations

### Current State Assessment

**Strengths**:
✅ Well-architected hybrid search (semantic + BM25)
✅ Production-ready error handling and logging
✅ Modular, maintainable code structure
✅ Comprehensive RAG pipeline (ingest → chunk → embed → search)
✅ Support for feedback-weighted retrieval
✅ SQL query integration for structured data

**Weaknesses for AWS**:
❌ Hardcoded server hostnames
❌ No cloud-native configuration management
❌ Local file-based data storage (not scalable)
❌ Remote Ollama dependency (network latency)
❌ No built-in auto-scaling
❌ Missing cloud observability (CloudWatch integration)

### AWS Deployment Path

**Recommended Approach**: **ECS Fargate + RDS PostgreSQL (with pgvector) + SageMaker Endpoints**

**Timeline**:
1. **Week 1**: Configuration externalization + AWS SDK integration
2. **Week 2**: Database migration testing
3. **Week 3**: ECR/ECS setup and container testing
4. **Week 4**: Load testing and cost optimization
5. **Week 5**: Production migration and validation

**Expected Outcome**:
- Scalable RAG system supporting 100x current document volume
- Auto-healing infrastructure with self-service deployment
- Per-request cost tracking and optimization
- 99.9% SLA availability

---

## References & Related Files

### Documentation
- `/DEPLOYMENT.md` - Current air-gapped deployment guide
- `/DOCKER_BUILD.md` - Docker containerization details
- `/README_ADVANCED_RAG.md` - Complete system documentation
- `/GPU_EMBEDDING_SETUP.md` - GPU configuration

### Configuration
- `/config/databases.yaml` - SQL server configs
- `/config/acronyms.json` - Domain terminology mappings
- `/cleaning_config.yaml` - Document cleaning patterns

### Key Source Files
- `airgapped_rag_advanced.py` - Main FastAPI application
- `parent_child_store.py` - Vector DB management
- `sql_query_handler.py` - SQL integration
- `metadata_extractor.py` - LLM metadata pipeline

---

**Document Generated**: 2025-11-12
**System**: ADAM RAG v2.0.0
**Branch**: claude/rag-production-server-011CV4Kkeetr9Vt6JMy5qWqD
