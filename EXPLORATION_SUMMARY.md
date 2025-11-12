# ADAM RAG Application - Codebase Exploration Summary

**Exploration Date**: November 12, 2025  
**Repository**: Adam-api (claude/rag-production-server branch)  
**Total Files Analyzed**: 60+ files (27,000+ lines of code & documentation)

---

## What Was Explored

This comprehensive analysis examined the ADAM RAG system to understand:
1. ✅ How models are currently loaded and used (embeddings, LLMs)
2. ✅ Configuration management (environment variables, config files)
3. ✅ Database connections and vector store usage
4. ✅ API structure and endpoints (13 endpoints)
5. ✅ Hardcoded paths and server-specific configurations
6. ✅ Dependencies on local GPU/compute resources
7. ✅ Docker and deployment configurations

---

## Key Findings

### System Architecture

**ADAM** is a production-grade **Retrieval-Augmented Generation** system:

```
PDF Upload → Docling Extraction → Document Cleaning → Semantic Chunking
                                                            ↓
                                                  Metadata Extraction (LLM)
                                                            ↓
                                   Parent-Child Chunks → ChromaDB Storage
                                                            ↓
Query → Hybrid Search (BM25 + e5-large-v2) → Semantic Reranking → LLM Answer
        (30 candidates)                          (top 5)          (with citations)
```

### Component Details

| Component | Technology | Config Location | Status |
|-----------|-----------|-----------------|--------|
| **Web Framework** | FastAPI 0.110.1 | airgapped_rag_advanced.py | ✅ Modern async |
| **Embedding Model** | e5-large-v2 (1024 dims) | parent_child_store.py:182 | ✅ SOTA retrieval |
| **LLM Service** | Ollama + Mistral Small 22B | Multiple (HARDCODED) | ⚠️ Remote, hardcoded |
| **Vector Database** | ChromaDB (local persistent) | /data/airgapped_rag/chromadb_advanced | ✅ Dual collection (parent-child) |
| **SQL Backend** | MS SQL Server 2019 | config/databases.yaml | ⚠️ Hardcoded hostname |
| **Document Processing** | Docling + Tesseract | Dockerfile.advanced | ✅ Structure-aware PDF extraction |
| **Hybrid Search** | BM25 + Semantic | parent_child_store.py | ✅ Keyword + semantic weighting |
| **Feedback System** | SQLite | /data/airgapped_rag/feedback.db | ✅ Quality scoring from user ratings |

### Hardcoded Configuration Issues (CRITICAL for AWS)

**Issue 1: Ollama Host** 
- Default: `http://adam.amentumspacemissions.com:11434`
- Location: `metadata_extractor.py`, line 69 (function default parameter)
- Impact: Cannot change without code modification
- **Fix**: Use environment variable + AWS Parameter Store

**Issue 2: SQL Server Address**
- Default: `CLGDBS02`
- Location: `config/databases.yaml`
- Impact: Requires YAML file edit to change servers
- **Fix**: Move to AWS Secrets Manager + environment variable substitution

**Issue 3: Data Directory Paths**
- Default: `/data/airgapped_rag`
- Location: All modules (hardcoded defaults)
- Impact: Won't work with cloud storage (S3, EFS)
- **Fix**: Prioritize environment variable (already supported)

### Resource Requirements

```
GPU/CPU:
  - Embedding Model (e5-large-v2): 2-3GB VRAM (10-50x faster than CPU)
  - LLM (Mistral 22B): 22GB VRAM (remote)
  - Document Processing: 8+ CPU cores, 2GB RAM
  
Memory:
  - Minimum: 8GB RAM
  - Recommended: 32GB RAM total
  
Storage:
  - ChromaDB: Grows with documents (~10MB per 1000 chunks)
  - Models: e5-large-v2 = 1.3GB
  - Documents: Varies with PDF size
```

---

## Deliverables Created

### 1. AWS_DEPLOYMENT_ANALYSIS.md (29KB)
**Location**: `/home/user/Adam-api/AWS_DEPLOYMENT_ANALYSIS.md`

Comprehensive guide covering:
- Detailed model loading architecture
- Configuration management strategy
- Database options comparison (ChromaDB vs RDS vs OpenSearch)
- Code modifications required (Priority 1-3)
- AWS deployment architectures (ECS Fargate recommended)
- Cost analysis ($1,280/month for recommended setup)
- Migration checklist with 30+ items
- Expected outcomes and success metrics

### 2. DEPLOYMENT_QUICK_REFERENCE.md (5KB)
**Location**: `/home/user/Adam-api/DEPLOYMENT_QUICK_REFERENCE.md`

Quick reference guide with:
- System architecture diagram
- Critical configuration points
- Component summary table
- AWS decision matrix (vector DB options)
- Priority code modifications
- Testing checklist
- Recommended deployment architecture
- Timeline and cost breakdown

### 3. This Exploration Summary
**Location**: `/home/user/Adam-api/EXPLORATION_SUMMARY.md`

---

## Architecture Overview

### Current Setup (On-Premises)
```
┌─────────────────┐
│  FastAPI RAG    │ (Container port 8000)
│  e5-large-v2    │ (GPU: 2-3GB VRAM)
│  16KB context   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐         ┌──────────────────┐
│ Remote Ollama   │ ◄─────► │ MS SQL Server    │
│ adam.amentum... │         │ Corporate DB     │
│ 22GB Mistral    │         │ Employee data    │
│ (HARDCODED)     │         │ (HARDCODED)      │
└────────┬────────┘         └──────────────────┘
         │
         ▼
┌──────────────────────┐
│ ChromaDB             │
│ /data/airgapped_rag/ │
├──────────────────────┤
│ - child_chunks       │ (with embeddings)
│ - parent_chunks      │ (context)
│ - feedback.db        │ (quality scores)
│ - conversations.db   │ (chat history)
└──────────────────────┘
```

### Recommended AWS Setup
```
┌──────────────────────────────────────┐
│    Application Load Balancer         │
└───────────────┬──────────────────────┘
                │
        ┌───────┴───────┐
        │               │
    ┌───▼───┐       ┌──▼────┐
    │ ECS-1 │       │ ECS-2  │ (Auto-scaling)
    │ RAG   │       │ RAG    │
    │+Ollama│       │+Ollama │
    └───┬───┘       └──┬─────┘
        │               │
        └───────┬───────┘
                ▼
        ┌──────────────┐
        │ EFS Storage  │
        │ - Documents  │
        │ - ChromaDB   │
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │ RDS PostgreSQL
        │ + pgvector   │
        └──────────────┘
```

---

## API Endpoints

### Document Management (6 endpoints)
- `POST /upload-document` - Ingest PDF
- `GET /documents` - List indexed docs
- `PUT /documents/{id}/questions` - Update answerable questions
- `DELETE /documents/{id}` - Remove document
- `DELETE /documents` - Clear all
- `GET /debug/document/{id}` - Debug view

### Query & Search (2 endpoints)
- `POST /query` - Synchronous Q&A (returns answer + citations)
- `POST /query-stream` - Streaming Q&A (Server-Sent Events)

### System Management (4 endpoints)
- `GET /health` - Health check
- `GET /statistics` - System stats
- `GET /acronyms` - Acronym mappings
- `PUT /acronyms` - Update acronyms

### SQL Integration (1 endpoint)
- `POST /query-employee` - Natural language SQL queries

### Debug (1 endpoint)
- `POST /debug/extract-markdown` - Test PDF extraction

---

## Key Files Reference

### Configuration
- `config/databases.yaml` - SQL server configs (HARDCODED)
- `config/acronyms.json` - Domain terminology
- `cleaning_config.yaml` - Document cleaning patterns

### Core Application
- `airgapped_rag_advanced.py` (1700+ lines) - Main FastAPI application + RAG pipeline
- `parent_child_store.py` (700+ lines) - ChromaDB + hybrid search (BM25 + semantic)
- `semantic_chunker.py` (500+ lines) - Parent-child document chunking
- `metadata_extractor.py` (600+ lines) - LLM-based metadata extraction
- `document_cleaner.py` (350+ lines) - Noise removal + regex patterns

### SQL & Integration
- `sql_query_handler.py` (1200+ lines) - Text-to-SQL conversion + execution
- `sql_routes.py` (600+ lines) - SQL API endpoints
- `conversation_manager.py` (400+ lines) - Chat history tracking
- `feedback_store.py` (400+ lines) - User feedback storage

### Utilities
- `query_classifier.py` (400+ lines) - System vs document query detection
- `run_advanced.py` (40 lines) - Application entry point
- `verify_gpu_setup.py` (60 lines) - GPU validation
- `diagnose_gpu.py` (150+ lines) - GPU diagnostics

### Deployment
- `Dockerfile.advanced` - Production container with PDF support
- `docker-compose.yml` - All-in-one setup
- `docker-compose.advanced.yml` - API-only (external Ollama)
- `docker-compose.airgapped.yml` - RHEL9 production setup

### Documentation (18 files, 5000+ lines)
- `README.md` - Overview
- `README_ADVANCED_RAG.md` - Complete system documentation
- `DEPLOYMENT.md` - Deployment guide
- `DOCKER_BUILD.md` - Docker instructions
- `GPU_EMBEDDING_SETUP.md` - GPU configuration
- Plus 13 other deployment/setup guides

---

## AWS Deployment Recommendations

### Recommended Approach: **Multi-Tier Architecture**

**Tier 1: API Service**
- ECS Fargate tasks (auto-scaling)
- 2 vCPU, 4GB memory each
- Application Load Balancer

**Tier 2: LLM Service**
- EC2 g4dn.2xlarge (GPU)
- Ollama container
- Cost: $600/month (or use SageMaker for $1200+)

**Tier 3: Vector Database**
- RDS PostgreSQL with pgvector extension
- Cost: $300/month

**Tier 4: Storage**
- EFS for shared documents/ChromaDB
- S3 for backups
- Cost: $150-200/month

**Tier 5: Configuration Management**
- AWS Secrets Manager (credentials)
- Systems Manager Parameter Store (config)

### Migration Path (2 weeks)

```
Week 1:
  Day 1-2: Fix hardcoded values in code
  Day 3-4: Set up AWS services (ECR, RDS, EFS)
  Day 5: Deploy to AWS (test environment)

Week 2:
  Day 1-2: Load testing & optimization
  Day 3-4: Data migration
  Day 5: Production cutover
```

### Cost Comparison

```
On-Premises:        $0/month (sunk costs)
AWS (This Design):  $1,280/month (with optimizations: ~$640/month)
SageMaker:          $1,480+/month
Lambda (not viable): Not suitable for GPU models
```

---

## Critical Issues Found

### 🔴 HIGH PRIORITY

1. **Hardcoded Ollama Host** in `metadata_extractor.py:69`
   - **Impact**: Cannot change without code modification
   - **Fix**: Use environment variables + Parameter Store
   - **Effort**: 2 hours

2. **Hardcoded SQL Server** in `config/databases.yaml`
   - **Impact**: Cannot connect to different servers without editing config
   - **Fix**: Move to environment variables + Secrets Manager
   - **Effort**: 4 hours

3. **No Cloud Configuration Management**
   - **Impact**: Cannot scale on AWS
   - **Fix**: Implement AWS SDK integration
   - **Effort**: 8 hours

### 🟡 MEDIUM PRIORITY

4. **Local File-Based Storage**
   - **Impact**: Won't work with auto-scaling
   - **Fix**: Migrate ChromaDB to RDS PostgreSQL
   - **Effort**: 16 hours

5. **Missing CloudWatch Integration**
   - **Impact**: No observability in AWS
   - **Fix**: Add JSON logging for CloudWatch
   - **Effort**: 4 hours

6. **Remote Ollama Dependency**
   - **Impact**: Network latency for every metadata extraction
   - **Fix**: Co-locate Ollama in AWS ECS
   - **Effort**: 8 hours

### 🟢 LOW PRIORITY

7. **No Request Rate Limiting**
   - **Effort**: 4 hours
   
8. **No Caching Layer**
   - **Effort**: 6 hours

---

## What Works Well

✅ **Hybrid Search Architecture**
- Combines BM25 (keyword) with semantic (embeddings)
- Smart reranking by semantic similarity
- Domain-specific acronym expansion

✅ **Parent-Child Chunking Strategy**
- Small precise chunks for retrieval
- Large context chunks for LLM
- Optimal balance between precision and context

✅ **Document Processing Pipeline**
- Structure-aware PDF extraction (Docling)
- Configurable cleaning patterns
- LLM-based metadata extraction

✅ **Feedback & Learning System**
- User feedback captured
- Chunk quality scoring
- Feedback-weighted retrieval

✅ **Error Handling & Logging**
- Comprehensive logging
- Graceful error handling
- Health check endpoint

✅ **Modular Architecture**
- Separate concerns (chunking, embedding, storage, LLM)
- Easy to modify components
- Good separation of duties

---

## Next Steps

1. **Review AWS_DEPLOYMENT_ANALYSIS.md** (detailed guide)
2. **Review DEPLOYMENT_QUICK_REFERENCE.md** (decision matrix)
3. **Fix Critical Issues** (hardcoded values) - ~14 hours
4. **Set up AWS Infrastructure** - ~20 hours
5. **Migrate Data** - ~24 hours
6. **Load Test** - ~16 hours
7. **Production Deployment** - ~8 hours

**Total Effort**: ~84 hours / ~2-3 weeks for one engineer

---

## Files Analyzed

**Total**: 60+ files across 9 categories:

1. **Python Application** (16 files, 10,000+ lines)
2. **Configuration** (3 files)
3. **Docker** (5 files)
4. **Documentation** (18 files, 5,000+ lines)
5. **Requirements** (4 files)
6. **Shell Scripts** (2 files)
7. **Git** (1 file - .gitignore)
8. **Other** (11 files)

---

## Document Locations

```
/home/user/Adam-api/
├── AWS_DEPLOYMENT_ANALYSIS.md          ← 29KB comprehensive guide
├── DEPLOYMENT_QUICK_REFERENCE.md       ← 5KB quick reference
├── EXPLORATION_SUMMARY.md              ← This file
├── airgapped_rag_advanced.py          ← Main application
├── parent_child_store.py               ← Vector DB + hybrid search
├── config/
│   ├── databases.yaml                  ← SQL configs (hardcoded)
│   └── acronyms.json
├── docker-compose.*.yml                ← 4 compose files
├── Dockerfile.*                        ← 3 Dockerfiles
└── DEPLOYMENT.md                       ← Current deployment guide
```

---

## Summary

The **ADAM RAG** system is a **well-architected, production-ready** document intelligence platform with:

- Advanced hybrid search (semantic + keyword)
- GPU-accelerated embeddings
- Comprehensive document processing
- Feedback-driven quality improvement
- SQL integration for structured data

**For AWS Deployment**, the primary changes needed are:

1. Extract hardcoded configuration values
2. Add AWS SDK integration (Secrets Manager, Parameter Store)
3. Migrate vector database from local ChromaDB to RDS PostgreSQL
4. Deploy Ollama to EC2 (or use SageMaker for $1200+/month)
5. Implement CloudWatch logging

**Estimated Cost**: $1,280/month → $640/month with optimizations  
**Estimated Effort**: 2-3 weeks for one engineer

---

**Report Generated**: November 12, 2025  
**Analyzed By**: Claude Code (File Search Specialist)  
**Status**: ✅ Complete & Ready for Implementation
