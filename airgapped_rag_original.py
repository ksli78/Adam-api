"""
Air-Gapped RAG API with Ollama and ChromaDB

A robust, air-gapped-compatible RAG system that:
1. Uses Ollama for embeddings (nomic-embed-text) and generation (Llama 3)
2. Uses ChromaDB for vector storage
3. Stores and retrieves FULL documents (no chunking)
4. Uses topic-based indexing (one embedding per document)
5. Provides accurate source citations with excerpts

Author: Claude AI
Date: 2025-10-24
"""

import os
import json
import uuid
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
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

# Vector Database
import chromadb
from chromadb.config import Settings

# Ollama
import ollama

# -------------------- Configuration --------------------

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "/data/airgapped_rag"))
DOCS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chromadb"

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3:8b")

# Collection name for ChromaDB
CHROMA_COLLECTION = "document_topics"

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
    top_k: Optional[int] = Field(default=1, description="Number of documents to retrieve (1-5)")

class QueryResponse(BaseModel):
    """Response model with answer and citations."""
    answer: str
    citations: List[Citation]

class UploadResponse(BaseModel):
    """Response after document upload."""
    document_id: str
    message: str
    topic: str
    source_url: str

class DocumentInfo(BaseModel):
    """Information about a stored document."""
    document_id: str
    topic: str
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
        # Extract text
        text = page.get_text()

        # Basic markdown formatting
        if page_num == 1:
            # First page title
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
            # First page title
            lines = text.split('\n')
            if lines:
                markdown_parts.append(f"# {lines[0].strip()}\n")
                markdown_parts.append('\n'.join(lines[1:]))
        else:
            markdown_parts.append(f"\n## Page {page_num}\n")
            markdown_parts.append(text)

    return '\n'.join(markdown_parts)

# -------------------- Document Storage --------------------

class DocumentStore:
    """
    Simple local document storage using JSON.
    Stores full document content and metadata.
    """

    def __init__(self, storage_path: Path = DOCS_DIR):
        self.storage_path = storage_path
        self.index_file = storage_path / "index.json"
        self._load_index()

    def _load_index(self):
        """Load the document index."""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def _save_index(self):
        """Save the document index."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    def store_document(
        self,
        doc_id: str,
        content: str,
        source_url: str,
        filename: str,
        topic: str
    ) -> None:
        """Store a document with its metadata."""
        # Save full content
        doc_file = self.storage_path / f"{doc_id}.md"
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Update index
        self.index[doc_id] = {
            'source_url': source_url,
            'filename': filename,
            'topic': topic,
            'created_at': datetime.now().isoformat(),
            'char_count': len(content)
        }
        self._save_index()
        logger.info(f"Stored document {doc_id} ({len(content)} chars)")

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        if doc_id not in self.index:
            return None

        doc_file = self.storage_path / f"{doc_id}.md"
        if not doc_file.exists():
            logger.error(f"Document file missing: {doc_file}")
            return None

        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            'id': doc_id,
            'content': content,
            **self.index[doc_id]
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents."""
        return [
            {'document_id': doc_id, **metadata}
            for doc_id, metadata in self.index.items()
        ]

# -------------------- Vector Store (ChromaDB) --------------------

class VectorStore:
    """
    ChromaDB-based vector store for document topic embeddings.
    """

    def __init__(self, persist_directory: Path = CHROMA_DIR):
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"description": "Document topic embeddings for air-gapped RAG"}
        )
        logger.info(f"ChromaDB collection '{CHROMA_COLLECTION}' ready")

    def add_document(
        self,
        doc_id: str,
        topic_embedding: List[float],
        source_url: str,
        topic: str
    ) -> None:
        """Add a document's topic embedding to the collection."""
        self.collection.add(
            ids=[doc_id],
            embeddings=[topic_embedding],
            metadatas=[{
                'source_url': source_url,
                'topic': topic
            }]
        )
        logger.info(f"Added topic embedding for document {doc_id}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 1
    ) -> Dict[str, Any]:
        """Search for similar document topics."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 5)
        )
        return results

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

# -------------------- Ollama Integration --------------------

class OllamaClient:
    """
    Client for interacting with Ollama for embeddings and generation.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        embed_model: str = OLLAMA_EMBED_MODEL,
        llm_model: str = OLLAMA_LLM_MODEL
    ):
        """Initialize Ollama client."""
        self.base_url = base_url
        self.embed_model = embed_model
        self.llm_model = llm_model

        # Configure ollama client
        ollama._client._base_url = base_url

        logger.info(f"Ollama client initialized: {base_url}")
        logger.info(f"  Embedding model: {embed_model}")
        logger.info(f"  LLM model: {llm_model}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama."""
        try:
            response = ollama.embeddings(
                model=self.embed_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Embedding generation failed: {str(e)}"
            )

    def generate_topic(self, content: str, max_chars: int = 3000) -> str:
        """
        Generate a concise topic/summary for a document.
        Uses the first portion of the document.
        """
        # Use first portion for topic generation
        sample = content[:max_chars]

        prompt = f"""Read the following document excerpt and generate a concise, descriptive title or topic summary (10-15 words maximum) that captures the main subject:

Document Excerpt:
{sample}

Topic/Title:"""

        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 50
                }
            )
            topic = response['response'].strip()
            # Clean up the topic
            topic = re.sub(r'^(Topic:|Title:|Summary:)\s*', '', topic, flags=re.IGNORECASE)
            topic = topic.split('\n')[0]  # Take first line only
            return topic[:200]  # Limit length
        except Exception as e:
            logger.error(f"Failed to generate topic: {e}")
            # Fallback: use first line
            lines = content.strip().split('\n')
            return lines[0][:200] if lines else "Untitled Document"

    def generate_answer_with_citations(
        self,
        question: str,
        documents: List[Dict[str, Any]]
    ) -> Tuple[str, List[Citation]]:
        """
        Generate an answer using retrieved documents and extract citations.
        """
        # Build context from full documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source_url = doc.get('source_url', 'Unknown')
            content = doc.get('content', '')
            context_parts.append(
                f"[Document {i} - Source: {source_url}]\n{content}\n"
            )

        context = "\n---\n".join(context_parts)

        # System instruction emphasizing citations
        system_instruction = """You are an expert assistant. Answer the user's question ONLY using the provided documents as context.

CRITICAL REQUIREMENTS:
1. For EVERY piece of information in your answer, you MUST identify which document it came from
2. Include direct quotes (excerpts) from the source documents to support your statements
3. Format citations as: [Document N: "exact quote from document"]
4. If information appears in multiple documents, cite all relevant sources
5. Do NOT make up information not present in the documents
6. If the documents don't contain enough information to answer, say so clearly"""

        prompt = f"""{system_instruction}

Documents:
{context}

User Question: {question}

Your Answer (with inline citations):"""

        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 500
                }
            )
            answer = response['response'].strip()

            # Extract citations from the answer
            citations = self._extract_citations(answer, documents)

            return answer, citations

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Answer generation failed: {str(e)}"
            )

    def _extract_citations(
        self,
        answer: str,
        documents: List[Dict[str, Any]]
    ) -> List[Citation]:
        """
        Extract citations from the generated answer.

        Looks for patterns like [Document N: "quote"] or extracts
        key sentences as excerpts.
        """
        citations = []

        # Pattern 1: Explicit citations [Document N: "quote"]
        citation_pattern = r'\[Document (\d+): "([^"]+)"\]'
        matches = re.findall(citation_pattern, answer)

        cited_docs = set()
        for doc_num, excerpt in matches:
            doc_idx = int(doc_num) - 1
            if 0 <= doc_idx < len(documents):
                doc = documents[doc_idx]
                citations.append(Citation(
                    source_url=doc.get('source_url', 'Unknown'),
                    excerpt=excerpt
                ))
                cited_docs.add(doc_idx)

        # Pattern 2: If no explicit citations found, extract key sentences
        # and match them to documents
        if not citations:
            # Extract meaningful sentences from the answer
            sentences = re.split(r'[.!?]+', answer)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

            # For each document, find best matching excerpt
            for doc_idx, doc in enumerate(documents):
                content = doc.get('content', '')
                source_url = doc.get('source_url', 'Unknown')

                # Find a good excerpt (first substantial paragraph)
                paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
                if paragraphs:
                    # Take first meaningful paragraph as excerpt
                    excerpt = paragraphs[0][:300]
                    # If too long, try to end at sentence boundary
                    if len(paragraphs[0]) > 300:
                        last_period = excerpt.rfind('.')
                        if last_period > 200:
                            excerpt = excerpt[:last_period + 1]

                    citations.append(Citation(
                        source_url=source_url,
                        excerpt=excerpt + "..."
                    ))

        return citations

# -------------------- Initialize Components --------------------

# Global instances
doc_store = DocumentStore()
vector_store = VectorStore()
ollama_client = OllamaClient()

# -------------------- FastAPI Application --------------------

app = FastAPI(
    title="Air-Gapped RAG API",
    description="Retrieval-Augmented Generation system using Ollama and ChromaDB",
    version="1.0.0"
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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ollama_base_url": OLLAMA_BASE_URL,
        "embed_model": OLLAMA_EMBED_MODEL,
        "llm_model": OLLAMA_LLM_MODEL,
        "documents_count": vector_store.count()
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
    2. Generate topic/summary for the document
    3. Generate embedding for the topic
    4. Store full document content
    5. Store topic embedding in ChromaDB
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not source_url:
        raise HTTPException(status_code=400, detail="source_url is required")

    # Generate document ID
    doc_id = uuid.uuid4().hex

    try:
        # Save uploaded PDF temporarily
        temp_pdf = DOCS_DIR / f"temp_{doc_id}.pdf"
        with open(temp_pdf, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Processing document: {file.filename} ({len(content)} bytes)")

        # Convert PDF to Markdown
        markdown_content = pdf_to_markdown(str(temp_pdf))

        # Clean up temp file
        temp_pdf.unlink()

        # Generate topic/summary
        logger.info("Generating topic summary...")
        topic = ollama_client.generate_topic(markdown_content)
        logger.info(f"Generated topic: {topic}")

        # Generate embedding for the topic
        logger.info("Generating topic embedding...")
        topic_embedding = ollama_client.generate_embedding(topic)

        # Store full document
        doc_store.store_document(
            doc_id=doc_id,
            content=markdown_content,
            source_url=source_url,
            filename=file.filename,
            topic=topic
        )

        # Store topic embedding in vector store
        vector_store.add_document(
            doc_id=doc_id,
            topic_embedding=topic_embedding,
            source_url=source_url,
            topic=topic
        )

        return UploadResponse(
            document_id=doc_id,
            message=f"Document indexed successfully. Topic: {topic}",
            topic=topic,
            source_url=source_url
        )

    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        # Clean up temp file if it exists
        if temp_pdf.exists():
            temp_pdf.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.

    Steps:
    1. Generate embedding for the user's question
    2. Search ChromaDB for top-K similar document topics
    3. Retrieve full documents from document store
    4. Generate answer using Ollama with full document context
    5. Extract and return citations
    """
    try:
        question = request.prompt.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        top_k = max(1, min(request.top_k or 1, 5))

        logger.info(f"Processing query: {question}")

        # Generate embedding for question
        logger.info("Generating query embedding...")
        query_embedding = ollama_client.generate_embedding(question)

        # Search for similar document topics
        logger.info(f"Searching for top {top_k} similar documents...")
        search_results = vector_store.search(query_embedding, top_k=top_k)

        if not search_results['ids'][0]:
            return QueryResponse(
                answer="No relevant documents found in the system. Please upload documents first.",
                citations=[]
            )

        # Retrieve full documents
        doc_ids = search_results['ids'][0]
        logger.info(f"Retrieved document IDs: {doc_ids}")

        documents = []
        for doc_id in doc_ids:
            doc = doc_store.get_document(doc_id)
            if doc:
                documents.append(doc)

        if not documents:
            return QueryResponse(
                answer="Documents not found in storage. The index may be corrupted.",
                citations=[]
            )

        # Generate answer with citations
        logger.info("Generating answer with Ollama...")
        answer, citations = ollama_client.generate_answer_with_citations(
            question=question,
            documents=documents
        )

        return QueryResponse(
            answer=answer,
            citations=citations
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all indexed documents."""
    try:
        docs = doc_store.list_documents()
        return [DocumentInfo(**doc) for doc in docs]
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the system."""
    try:
        # Remove from vector store
        vector_store.collection.delete(ids=[document_id])

        # Remove from document store
        doc_file = DOCS_DIR / f"{document_id}.md"
        if doc_file.exists():
            doc_file.unlink()

        # Update index
        if document_id in doc_store.index:
            del doc_store.index[document_id]
            doc_store._save_index()

        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Startup --------------------

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 60)
    logger.info("Air-Gapped RAG API Starting")
    logger.info("=" * 60)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Documents indexed: {vector_store.count()}")
    logger.info(f"Ollama base URL: {OLLAMA_BASE_URL}")
    logger.info(f"Embedding model: {OLLAMA_EMBED_MODEL}")
    logger.info(f"LLM model: {OLLAMA_LLM_MODEL}")
    logger.info("=" * 60)

    # Test Ollama connection
    try:
        ollama.list()
        logger.info("✓ Ollama connection successful")
    except Exception as e:
        logger.warning(f"⚠ Ollama connection failed: {e}")
        logger.warning("  Make sure Ollama is running and accessible")

# -------------------- Main --------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "airgapped_rag:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
