"""
Air-Gapped RAG API with Haystack, Ollama and ChromaDB

A robust RAG system using:
- Haystack 2.x for RAG pipelines
- Ollama for local embeddings and generation
- ChromaDB for vector storage
- Hybrid retrieval: BM25 (keyword) + Semantic (embeddings)
- Reranking for better result quality

Author: Claude AI (Refactored with Haystack)
Date: 2025-10-24
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# PDF Processing
try:
    import fitz  # PyMuPDF
    PDF_BACKEND = "pymupdf"
except ImportError:
    from pypdf import PdfReader
    PDF_BACKEND = "pypdf"

# Haystack
from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice

# ChromaDB document store (persistent)
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever, ChromaQueryTextRetriever

# Ollama (for generation)
import ollama

# -------------------- Configuration --------------------

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "/data/airgapped_rag"))
DOCS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chromadb"  # ChromaDB persistence directory

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3:8b")

# Embedding model (Sentence Transformers compatible with Ollama)
# We'll use sentence-transformers for embeddings (better than Ollama's nomic-embed-text for retrieval)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Reranker model
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieval settings
TOP_K_RETRIEVAL = 10  # Retrieve top 10 from each retriever
TOP_K_RERANK = 3      # Rerank to top 3

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------- Ensure Directories --------------------

def ensure_directories():
    """Create necessary directories if they don't exist."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

ensure_directories()

# -------------------- Pydantic Models --------------------

class Citation(BaseModel):
    """Citation with source URL and supporting excerpt."""
    source_url: str
    excerpt: str

class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    prompt: str = Field(..., description="The user's question")
    top_k: Optional[int] = Field(default=3, description="Number of documents to retrieve (1-10)")
    use_hybrid: Optional[bool] = Field(default=True, description="Use hybrid retrieval (BM25 + semantic)")

class QueryResponse(BaseModel):
    """Response model with answer and citations."""
    answer: str
    citations: List[Citation]

class UploadResponse(BaseModel):
    """Response after document upload."""
    document_id: str
    message: str
    source_url: str

class DocumentInfo(BaseModel):
    """Document information."""
    document_id: str
    source_url: str
    filename: str
    created_at: str
    char_count: int

# -------------------- PDF to Markdown Conversion --------------------

def pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert PDF to Markdown format.
    Uses PyMuPDF if available, falls back to pypdf.
    """
    logger.info(f"Converting PDF to Markdown using {PDF_BACKEND}")

    if PDF_BACKEND == "pymupdf":
        return _pdf_to_markdown_pymupdf(pdf_path)
    else:
        return _pdf_to_markdown_pypdf(pdf_path)

def _pdf_to_markdown_pymupdf(pdf_path: str) -> str:
    """Convert PDF to Markdown using PyMuPDF."""
    doc = fitz.open(pdf_path)
    markdown_parts = []

    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if page_num == 1:
            lines = text.split('\n')
            if lines:
                markdown_parts.append(f"# {lines[0].strip()}\n")
                markdown_parts.append('\n'.join(lines[1:]))
        else:
            markdown_parts.append(f"\n## Page {page_num}\n")
            markdown_parts.append(text)

    doc.close()
    return '\n'.join(markdown_parts)

def _pdf_to_markdown_pypdf(pdf_path: str) -> str:
    """Convert PDF to Markdown using pypdf."""
    reader = PdfReader(pdf_path)
    markdown_parts = []

    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if page_num == 1:
            lines = text.split('\n')
            if lines:
                markdown_parts.append(f"# {lines[0].strip()}\n")
                markdown_parts.append('\n'.join(lines[1:]))
        else:
            markdown_parts.append(f"\n## Page {page_num}\n")
            markdown_parts.append(text)

    return '\n'.join(markdown_parts)

# -------------------- Document Metadata Extraction --------------------

def extract_document_metadata(content: str) -> Dict[str, str]:
    """
    Extract metadata from document content.
    Returns: {doc_number, doc_title, key_terms}
    """
    metadata = {
        'doc_number': '',
        'doc_title': '',
        'key_terms': ''
    }

    sample = content[:2000]
    lines = sample.split('\n')

    # Extract document number
    doc_number_patterns = [
        r'Document\s+No[.:]?\s*([A-Z]{2}-[A-Z]{2}-\d{4})',
        r'Doc\s+#?\s*:?\s*([A-Z]{2}-[A-Z]{2}-\d{4})',
        r'([A-Z]{2}-[A-Z]{2}-\d{4})'
    ]

    for pattern in doc_number_patterns:
        match = re.search(pattern, sample, re.IGNORECASE)
        if match:
            metadata['doc_number'] = match.group(1)
            break

    # Extract title (prioritize lines with "Policy" or "Procedure")
    doc_title_candidates = []

    for i, line in enumerate(lines[:40]):
        line = line.strip()
        line_lower = line.lower()

        if (len(line) < 5 or len(line) > 80 or
            line.startswith('Page') or
            'proprietary' in line_lower or
            'copyright' in line_lower or
            'confidential' in line_lower or
            'amentum' in line_lower):
            continue

        if re.match(r'^[A-Z]{2}-[A-Z]{2}-\d{4}$', line):
            continue

        if any(keyword in line_lower for keyword in ['policy', 'procedure', 'guideline', 'standard']):
            doc_title_candidates.append((i, line))
        elif (line[0].isupper() and
              1 <= line.count(' ') <= 8 and
              10 <= len(line) <= 60):
            if line_lower not in ['management system', 'document', 'revision', 'effective date']:
                doc_title_candidates.append((i, line))

    # Prefer titles with policy/procedure keywords
    for idx, title in doc_title_candidates:
        title_lower = title.lower()
        if any(kw in title_lower for kw in ['policy', 'procedure', 'guideline']):
            metadata['doc_title'] = title
            break

    # Fallback to first candidate
    if not metadata['doc_title'] and doc_title_candidates:
        metadata['doc_title'] = doc_title_candidates[0][1]

    return metadata

# -------------------- Haystack Setup --------------------

class HaystackRAG:
    """Haystack-based RAG system with hybrid retrieval and reranking."""

    def __init__(self):
        """Initialize Haystack document store and pipelines."""
        logger.info("Initializing Haystack RAG system...")

        # ChromaDB document store (PERSISTENT - survives restarts!)
        self.document_store = ChromaDocumentStore(
            persist_path=str(CHROMA_DIR),
            collection_name="documents"
        )
        logger.info(f"ChromaDB initialized at: {CHROMA_DIR}")
        logger.info(f"Existing documents in store: {self.document_store.count_documents()}")

        # Embedder for documents (indexing)
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model=EMBEDDING_MODEL,
            device=ComponentDevice.from_str("cpu"),
            progress_bar=False
        )
        self.doc_embedder.warm_up()

        # Embedder for queries
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=EMBEDDING_MODEL,
            device=ComponentDevice.from_str("cpu")
        )
        self.text_embedder.warm_up()

        # Retrievers (ChromaDB-based)
        self.embedding_retriever = ChromaEmbeddingRetriever(
            document_store=self.document_store,
            top_k=TOP_K_RETRIEVAL
        )

        # For BM25/text-based retrieval, ChromaDB uses text queries
        self.bm25_retriever = ChromaQueryTextRetriever(
            document_store=self.document_store,
            top_k=TOP_K_RETRIEVAL
        )

        # Reranker
        self.ranker = TransformersSimilarityRanker(
            model=RERANKER_MODEL,
            top_k=TOP_K_RERANK,
            device=ComponentDevice.from_str("cpu")
        )
        self.ranker.warm_up()

        # Document joiner (for hybrid retrieval)
        self.joiner = DocumentJoiner()

        # Build pipelines
        self._build_indexing_pipeline()
        self._build_query_pipeline()

        logger.info("Haystack RAG system initialized successfully")

    def _build_indexing_pipeline(self):
        """Build the document indexing pipeline."""
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("embedder", self.doc_embedder)
        self.indexing_pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
        self.indexing_pipeline.connect("embedder.documents", "writer.documents")

    def _build_query_pipeline(self):
        """Build the hybrid query pipeline with BM25 + semantic search + reranking."""
        self.query_pipeline = Pipeline()

        # Add components
        self.query_pipeline.add_component("text_embedder", self.text_embedder)
        self.query_pipeline.add_component("embedding_retriever", self.embedding_retriever)
        self.query_pipeline.add_component("bm25_retriever", self.bm25_retriever)
        self.query_pipeline.add_component("joiner", self.joiner)
        self.query_pipeline.add_component("ranker", self.ranker)

        # Connect components
        self.query_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
        self.query_pipeline.connect("embedding_retriever.documents", "joiner.documents")
        self.query_pipeline.connect("bm25_retriever.documents", "joiner.documents")
        self.query_pipeline.connect("joiner.documents", "ranker.documents")

    def index_document(self, content: str, source_url: str, filename: str, doc_metadata: Dict[str, str]) -> str:
        """
        Index a document into the store.

        Args:
            content: Full document content (markdown)
            source_url: Source URL for citation
            filename: Original filename
            doc_metadata: Extracted metadata (doc_number, doc_title)

        Returns:
            document_id
        """
        # Create Haystack document
        doc_id = f"{doc_metadata.get('doc_number', filename)}_{datetime.now().timestamp()}"

        # Build rich metadata
        meta = {
            'source_url': source_url,
            'filename': filename,
            'doc_number': doc_metadata.get('doc_number', ''),
            'doc_title': doc_metadata.get('doc_title', ''),
            'created_at': datetime.now().isoformat(),
            'char_count': len(content)
        }

        # Create document
        haystack_doc = Document(
            id=doc_id,
            content=content,
            meta=meta
        )

        # Index through pipeline (embeds and stores)
        logger.info(f"Indexing document: {filename} ({len(content)} chars)")
        self.indexing_pipeline.run({"embedder": {"documents": [haystack_doc]}})

        logger.info(f"Document indexed successfully: {doc_id}")
        return doc_id

    def query(self, question: str, top_k: int = 3, use_hybrid: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system with hybrid retrieval and reranking.

        Args:
            question: User's question
            top_k: Number of documents to return after reranking
            use_hybrid: Use hybrid retrieval (BM25 + semantic)

        Returns:
            {documents: List[Document], answer: str, citations: List[Citation]}
        """
        logger.info(f"Processing query: {question}")

        # Update ranker top_k
        self.ranker.top_k = top_k

        # Run query pipeline
        if use_hybrid:
            # Hybrid: BM25 + semantic
            result = self.query_pipeline.run({
                "text_embedder": {"text": question},
                "bm25_retriever": {"query": question},
                "ranker": {"query": question}
            })
        else:
            # Semantic only - we need to bypass BM25 retriever entirely
            # Use a simplified pipeline for semantic-only mode
            logger.info("Using semantic-only retrieval (no BM25)")

            # Run just embedding retriever + ranker (skip BM25 and joiner)
            embedding = self.text_embedder.run(text=question)
            retrieved = self.embedding_retriever.run(
                query_embedding=embedding["embedding"],
                top_k=TOP_K_RETRIEVAL
            )
            reranked = self.ranker.run(
                query=question,
                documents=retrieved["documents"],
                top_k=top_k
            )
            result = {"ranker": reranked}

        # Get ranked documents
        documents = result.get("ranker", {}).get("documents", [])

        logger.info(f"Retrieved {len(documents)} documents after reranking")

        # Generate answer using Ollama
        answer = self._generate_answer(question, documents)

        # Extract citations
        citations = self._extract_citations(answer, documents, question)

        return {
            "documents": documents,
            "answer": answer,
            "citations": citations
        }

    def _generate_answer(self, question: str, documents: List[Document]) -> str:
        """Generate answer using Ollama with retrieved documents."""
        if not documents:
            return "No relevant documents found to answer this question."

        # Build context from documents - use MORE content per document
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.content[:4000]  # Increased from 2000 to 4000 chars
            doc_num = doc.meta.get('doc_number', 'Unknown')
            doc_title = doc.meta.get('doc_title', 'Unknown')
            source = doc.meta.get('source_url', 'Unknown')
            context_parts.append(f"""Document {i}:
Source: {source}
Document Number: {doc_num}
Title: {doc_title}
Content:
{content}
""")

        context = "\n" + ("=" * 80) + "\n".join(context_parts)

        # Build MORE DIRECTIVE prompt
        prompt = f"""You are answering questions based on policy documents. READ THE DOCUMENTS CAREFULLY.

QUESTION: {question}

DOCUMENTS PROVIDED:
{context}

INSTRUCTIONS:
1. READ the documents above CAREFULLY
2. SEARCH for relevant information in the document content
3. If you find the answer, provide it and cite the document: "According to Document N (doc_number)..."
4. If multiple documents have information, mention all relevant ones
5. ONLY say "no information found" if you truly cannot find ANY relevant information after carefully reading
6. Pay special attention to section numbers (like 4.3, 5.4) as they often contain specific policies

ANSWER (be thorough and cite your sources):"""

        # Call Ollama with higher temperature for better reasoning
        try:
            response = ollama.generate(
                model=OLLAMA_LLM_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.3,  # Increased from 0.1 for better reasoning
                    'num_predict': 500   # Allow longer responses
                }
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

    def _extract_citations(self, answer: str, documents: List[Document], question: str) -> List[Citation]:
        """Extract citations from documents mentioned in the answer."""
        citations = []

        # Find "Document N" references in answer
        mentioned_docs = set()
        doc_references = re.findall(r'Document\s+#?(\d+)', answer, re.IGNORECASE)
        for doc_num in doc_references:
            doc_idx = int(doc_num) - 1
            if 0 <= doc_idx < len(documents):
                mentioned_docs.add(doc_idx)

        # Check if answer says "no information"
        no_info_patterns = [
            r'do(?:es)?\s+not\s+contain',
            r'no\s+information',
            r'not\s+found',
            r'does(?:n\'t)?\s+address'
        ]

        answer_lower = answer.lower()
        if any(re.search(pattern, answer_lower) for pattern in no_info_patterns):
            return []

        # If no explicit mentions but answer has content, use first doc
        if not mentioned_docs and len(answer) > 50:
            mentioned_docs.add(0)

        # Extract excerpts from mentioned documents
        for doc_idx in sorted(mentioned_docs):
            if doc_idx < len(documents):
                doc = documents[doc_idx]
                excerpt = self._find_best_excerpt(doc.content, question)

                citations.append(Citation(
                    source_url=doc.meta.get('source_url', 'Unknown'),
                    excerpt=excerpt
                ))

        logger.info(f"Extracted {len(citations)} citations")
        return citations

    def _find_best_excerpt(self, content: str, question: str, max_length: int = 300) -> str:
        """Find the most relevant excerpt from content."""
        # Find numbered sections (5.1, 5.2, etc.)
        sections = re.findall(r'(\d+\.\d+\s+[^\n]+\n[^\n]{50,500})', content)

        if sections:
            # Score sections by question term overlap
            question_terms = set(question.lower().split())
            best_section = max(sections, key=lambda s: len(set(s.lower().split()) & question_terms))
            excerpt = best_section[:max_length]
            if len(best_section) > max_length:
                last_period = excerpt.rfind('.')
                if last_period > max_length * 0.7:
                    excerpt = excerpt[:last_period + 1]
            return excerpt + "..."

        # Fallback: return first part
        return content[:max_length] + "..."

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents."""
        docs = self.document_store.filter_documents()
        return [
            {
                "document_id": doc.id,
                "source_url": doc.meta.get('source_url', 'Unknown'),
                "filename": doc.meta.get('filename', 'Unknown'),
                "doc_number": doc.meta.get('doc_number', ''),
                "doc_title": doc.meta.get('doc_title', ''),
                "created_at": doc.meta.get('created_at', ''),
                "char_count": doc.meta.get('char_count', 0)
            }
            for doc in docs
        ]

    def delete_document(self, document_id: str):
        """Delete a document from the store."""
        self.document_store.delete_documents([document_id])

# -------------------- Initialize Haystack RAG --------------------

haystack_rag = HaystackRAG()

# -------------------- FastAPI Application --------------------

app = FastAPI(
    title="Air-Gapped RAG API (Haystack)",
    description="RAG system using Haystack with hybrid retrieval and reranking",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- API Endpoints --------------------

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 60)
    logger.info("Air-Gapped RAG API (Haystack) Starting")
    logger.info("=" * 60)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"ChromaDB directory: {CHROMA_DIR}")
    logger.info(f"Document count: {len(haystack_rag.list_documents())} (PERSISTENT)")
    logger.info(f"Ollama base URL: {OLLAMA_BASE_URL}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Reranker model: {RERANKER_MODEL}")
    logger.info(f"LLM model: {OLLAMA_LLM_MODEL}")
    logger.info("=" * 60)

    # Test Ollama connection
    try:
        ollama.list()
        logger.info("✓ Ollama connection successful")
    except Exception as e:
        logger.error(f"✗ Ollama connection failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "document_count": len(haystack_rag.list_documents()),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": OLLAMA_LLM_MODEL
    }

@app.post("/upload-document", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    source_url: str = ""
):
    """
    Upload a PDF document for indexing.

    Steps:
    1. Convert PDF to Markdown
    2. Extract metadata (doc number, title)
    3. Index document with hybrid search (BM25 + semantic embeddings)
    """
    # Validate
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not source_url:
        raise HTTPException(status_code=400, detail="source_url is required")

    try:
        # Save PDF temporarily
        temp_pdf = DOCS_DIR / f"temp_{file.filename}"
        with open(temp_pdf, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Processing document: {file.filename} ({len(content)} bytes)")

        # Convert to Markdown
        markdown_content = pdf_to_markdown(str(temp_pdf))

        # Clean up temp file
        temp_pdf.unlink()

        # Extract metadata
        logger.info("Extracting document metadata...")
        doc_metadata = extract_document_metadata(markdown_content)
        logger.info(f"Extracted metadata: {doc_metadata}")

        # Index document
        doc_id = haystack_rag.index_document(
            content=markdown_content,
            source_url=source_url,
            filename=file.filename,
            doc_metadata=doc_metadata
        )

        return UploadResponse(
            document_id=doc_id,
            message=f"Document indexed successfully with Haystack hybrid retrieval",
            source_url=source_url
        )

    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with hybrid retrieval.

    Uses:
    - BM25 retrieval (keyword matching)
    - Semantic retrieval (embedding similarity)
    - Document joiner (combines results)
    - Reranking (improves final results)
    """
    try:
        question = request.prompt.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        top_k = max(1, min(request.top_k or 3, 10))

        # Query with hybrid retrieval
        result = haystack_rag.query(
            question=question,
            top_k=top_k,
            use_hybrid=request.use_hybrid
        )

        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"]
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all indexed documents."""
    try:
        docs = haystack_rag.list_documents()
        return [DocumentInfo(**doc) for doc in docs]
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the system."""
    try:
        haystack_rag.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug-search")
async def debug_search(request: QueryRequest):
    """
    Debug endpoint to show retrieval details before reranking.
    """
    try:
        question = request.prompt.strip()

        # Run retrieval without answer generation
        if request.use_hybrid:
            result = haystack_rag.query_pipeline.run({
                "text_embedder": {"text": question},
                "bm25_retriever": {"query": question},
                "ranker": {"query": question}
            })
        else:
            result = haystack_rag.query_pipeline.run({
                "text_embedder": {"text": question},
                "embedding_retriever": {"top_k": 10},
                "bm25_retriever": {"query": "", "top_k": 0},
                "ranker": {"query": question}
            })

        # Get documents
        documents = result.get("ranker", {}).get("documents", [])

        # Format debug info
        debug_info = {
            "query": question,
            "use_hybrid": request.use_hybrid,
            "retrieved_documents": len(documents),
            "documents": [
                {
                    "rank": i + 1,
                    "doc_id": doc.id,
                    "doc_number": doc.meta.get('doc_number', 'N/A'),
                    "doc_title": doc.meta.get('doc_title', 'N/A'),
                    "source_url": doc.meta.get('source_url', 'N/A'),
                    "score": doc.score if hasattr(doc, 'score') else 0,
                    "content_preview": doc.content[:200] + "..."
                }
                for i, doc in enumerate(documents)
            ]
        }

        return debug_info

    except Exception as e:
        logger.error(f"Debug search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Run Application --------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("airgapped_rag_haystack:app", host="0.0.0.0", port=8000, reload=False)
