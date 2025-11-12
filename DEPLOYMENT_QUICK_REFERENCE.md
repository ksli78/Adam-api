# ADAM RAG - AWS Deployment Quick Reference

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          CURRENT SETUP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐                ┌─────────────────────┐   │
│  │   FastAPI RAG    │                │  Remote Ollama      │   │
│  │   (Container)    │◄──HTTP─────────►(GPU Server)         │   │
│  │  Port: 8000      │ :11434          Mistral Small 22B    │   │
│  │  e5-large-v2     │                 22GB VRAM            │   │
│  │  (Embeddings)    │                 Hardcoded Host       │   │
│  └────────┬─────────┘                └─────────────────────┘   │
│           │                                                     │
│           │ ChromaDB                  ┌──────────────────┐     │
│           ├──────────────────────────►│  MS SQL Server   │     │
│           │ (Vector Store)            │  Corporate DB    │     │
│           │ Persistent Storage        │  Hardcoded Conn  │     │
│           │                           └──────────────────┘     │
│  ┌────────▼──────────┐                                          │
│  │ Local Databases   │                                          │
│  │ - feedback.db     │                                          │
│  │ - conversations.db│                                          │
│  └───────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Critical Configuration Points

### 1. Environment Variables

| Variable | Current Default | Location | Impact |
|----------|-----------------|----------|--------|
| `OLLAMA_HOST` | `adam.amentumspacemissions.com:11434` | airgapped_rag_advanced.py:66 | **HARDCODED** - Breaking issue for AWS |
| `LLM_MODEL` | `mistral-small:22b` | metadata_extractor.py:69 | Ollama model selection |
| `DATA_DIR` | `/data/airgapped_rag` | All modules | Data persistence path |
| `LLM_CONTEXT_WINDOW` | `16384` | airgapped_rag_advanced.py:72 | Token limit for LLM |

### 2. Hardcoded Server Addresses

**CRITICAL ISSUES** ⚠️:

```python
# metadata_extractor.py, Line 69 - HARDCODED DEFAULT
def __init__(
    self,
    ollama_host: str = "http://adam.amentumspacemissions.com:11434",  # 🚨
    ...
):
```

**Solution**: Replace with environment variable + Parameter Store

```yaml
# config/databases.yaml - HARDCODED SERVER
connection:
    server: "CLGDBS02"          # 🚨 Cannot change without editing YAML
    user: "svcasm-adamAi"
```

**Solution**: Move to Secrets Manager + environment variables

## Component Summary

### Models & Services

| Component | Type | Config | Size | Resources |
|-----------|------|--------|------|-----------|
| **e5-large-v2** | Embedding | `parent_child_store.py:182` | 1.3GB | 2-3GB VRAM (GPU) |
| **Mistral Small 22B** | LLM | Remote Ollama | ~45GB | 22GB VRAM + 32GB RAM |
| **ChromaDB** | Vector DB | Persistent on disk | Varies | 1-2GB RAM |
| **MS SQL Server** | Relational DB | `config/databases.yaml` | N/A | Remote connection |

### API Endpoints

| Endpoint | Purpose | Rate |
|----------|---------|------|
| `POST /upload-document` | Ingest PDFs | ~5 min per doc |
| `POST /query` | Q&A (sync) | ~2-5 sec per query |
| `POST /query-stream` | Q&A (streaming) | ~1-3 sec per query |
| `POST /query-employee` | SQL natural language | ~2-4 sec |
| `GET /health` | Health check | <100ms |

### Data Directories

```
/data/airgapped_rag/
├── documents/              # Uploaded PDFs
├── chromadb_advanced/      # Vector database (child + parent chunks)
├── feedback.db             # SQLite - user feedback
└── conversations.db        # SQLite - conversation history
```

## AWS Deployment Decision Matrix

### Vector Database Options

| Option | Pros | Cons | Cost | Migration |
|--------|------|------|------|-----------|
| **EFS + Local ChromaDB** | Simplest, quick | Limited scaling | $150/mo | Easy |
| **RDS PostgreSQL + pgvector** | Scalable, managed, SQL | Complexity | $300/mo | Moderate |
| **OpenSearch** | Full-text search | Overkill cost | $500+/mo | Hard |
| **DynamoDB** | Serverless, auto-scale | Limited search, cost | $200-400/mo | Hard |

**RECOMMENDATION**: RDS PostgreSQL + pgvector (best balance)

### LLM Deployment Options

| Option | Pros | Cons | Cost/Month | Setup Time |
|--------|------|------|-----------|-----------|
| **ECS Fargate Sidecar** | Full control, local | Complex | $650 | 1 week |
| **EC2 with Ollama** | Simple, familiar | Manual scaling | $850 | 3 days |
| **SageMaker Endpoint** | Auto-scaling | AWS-specific | $1200+ | 4 days |
| **Keep Remote** | No change | Network latency | $0 | 0 days |

**RECOMMENDATION**: EC2 with Ollama (simplest migration with auto-scaling group)

## Code Modifications Required

### Priority 1 - CRITICAL (Week 1)

```python
# 1. Fix hardcoded Ollama host
# File: metadata_extractor.py, Line 69
# BEFORE: def __init__(self, ollama_host: str = "http://adam...")
# AFTER: def __init__(self, ollama_host: str = None)
#        if ollama_host is None:
#            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 2. Add AWS Parameter Store support
import boto3
ssm = boto3.client('ssm')
OLLAMA_HOST = os.getenv("OLLAMA_HOST") or ssm.get_parameter(
    Name='/adam-rag/ollama_host'
)['Parameter']['Value']

# 3. Move SQL credentials to Secrets Manager
secrets = boto3.client('secretsmanager')
db_config = json.loads(
    secrets.get_secret_value(SecretId='adam-rag/database')['SecretString']
)
```

### Priority 2 - IMPORTANT (Week 2)

```python
# 4. Add CloudWatch logging
from pythonjsonlogger import jsonlogger
logHandler = logging.StreamHandler()
logHandler.setFormatter(jsonlogger.JsonFormatter())
logger.addHandler(logHandler)

# 5. Externalize config files
import os
import yaml
with open(os.getenv("CONFIG_PATH", "config/databases.yaml")) as f:
    config = yaml.safe_load(f)

# 6. Add environment variable substitution
# For ${VARIABLE:-default} syntax in YAML
```

### Priority 3 - NICE-TO-HAVE (Week 3-4)

```python
# 7. Add SageMaker fallback
# 8. Implement caching layer (CloudFront)
# 9. Add batch processing for embeddings
# 10. Implement request rate limiting
```

## Testing Checklist

- [ ] Embedding model loads on GPU/CPU
- [ ] Ollama connection works over network
- [ ] SQL Server queries execute
- [ ] Document upload/ingestion completes
- [ ] Hybrid search (BM25 + semantic) works
- [ ] Streaming responses work
- [ ] Feedback storage/retrieval works
- [ ] Error handling for missing models
- [ ] Health checks pass

## Deployment Architecture (Recommended)

```yaml
# AWS ECS Fargate Cluster
┌─────────────────────────────────────────────────────────────┐
│                    Application Load Balancer                │
│                        (Port 8000)                          │
└─────────────┬───────────────────────────────────────────────┘
              │
    ┌─────────┴──────────┐
    │                    │
┌───▼────────┐    ┌─────▼────────┐
│ ECS Task 1 │    │ ECS Task 2   │ ← Auto-scaling
│ RAG API    │    │ RAG API      │
│ + Ollama   │    │ + Ollama     │
└───┬────────┘    └─────┬────────┘
    │                    │
    └────────┬───────────┘
             │
      ┌──────▼───────────┐
      │  EFS Storage     │
      │ - Documents      │
      │ - ChromaDB       │
      │ - Models         │
      └──────┬───────────┘
             │
      ┌──────▼───────────┐
      │ RDS PostgreSQL   │
      │ + pgvector       │
      └──────────────────┘

    ┌──────────────────────┐
    │ Secrets Manager      │
    │ - DB credentials     │
    │ - API keys           │
    └──────────────────────┘
```

## Estimated Timeline & Effort

| Phase | Task | Effort | Time | Owner |
|-------|------|--------|------|-------|
| 1 | Code modifications (hardcoded values) | 16h | 2d | Dev |
| 2 | AWS infrastructure setup | 20h | 3d | DevOps |
| 3 | Database migration & testing | 24h | 4d | Data |
| 4 | Load testing (1000+ docs) | 16h | 2d | QA |
| 5 | Production migration | 8h | 1d | DevOps |
| **Total** | | **84h** | **2 weeks** | |

## Cost Breakdown (Monthly)

```
Current Setup (On-Premises):
  Ollama Server:        $0 (existing)
  Storage:              $0 (existing)
  ─────────────────────────────
  Total:                $0

AWS Recommended Setup (ECS Fargate):
  Ollama EC2 (g4dn.2xlarge): $600
  RAG API (ECS):             $150
  EFS Storage:               $150
  RDS PostgreSQL:            $300
  Data Transfer:              $50
  CloudWatch Monitoring:      $30
  ─────────────────────────────
  Total:                    $1,280/month

  → 50-60% savings with 1-year reserved instances
  → Additional savings with spot instances for non-critical
```

## Success Metrics

- [ ] Sub-2 second query latency (P99)
- [ ] 99.9% availability SLA
- [ ] Auto-scaling to handle 10x load
- [ ] No manual intervention for 30 days
- [ ] < $1 per query cost
- [ ] All requests logged to CloudWatch

## Key Files to Review

1. **Configuration**
   - `/config/databases.yaml` - SQL configs (hardcoded server)
   - `/config/acronyms.json` - Domain terms
   - `/cleaning_config.yaml` - Document cleaning

2. **Main Application**
   - `/airgapped_rag_advanced.py` - Entry point (hardcoded OLLAMA_HOST)
   - `/metadata_extractor.py` - Hardcoded Ollama default
   - `/parent_child_store.py` - Vector DB management
   - `/run_advanced.py` - Startup configuration

3. **Deployment**
   - `/docker-compose.yml` - All-in-one
   - `/docker-compose.advanced.yml` - API only
   - `/Dockerfile.advanced` - Production Dockerfile
   - `/requirements.txt` - Dependencies

4. **Documentation**
   - `/DEPLOYMENT.md` - Current air-gapped setup
   - `/AWS_DEPLOYMENT_ANALYSIS.md` - Detailed analysis (29KB)
   - `/README_ADVANCED_RAG.md` - System documentation

---

**Last Updated**: 2025-11-12
**Version**: 1.0
**Status**: READY FOR EXECUTION
