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
    rewrite_query: Optional[bool] = Field(default=True, description="Enable query rewriting for better retrieval")

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

# -------------------- Document Metadata Extraction --------------------

def extract_document_metadata(content: str) -> Dict[str, str]:
    """
    Extract metadata from document content.

    Tries to find:
    - Document number (e.g., EN-PO-0071)
    - Document title
    - Key terms
    """
    metadata = {
        'doc_number': '',
        'doc_title': '',
        'key_terms': ''
    }

    # Get first 2000 chars for metadata extraction
    sample = content[:2000]
    lines = sample.split('\n')

    # Try to find document number (common patterns)
    doc_number_patterns = [
        r'Document\s+No[.:]?\s*([A-Z]{2}-[A-Z]{2}-\d{4})',  # EN-PO-0071
        r'Doc\s+#?\s*:?\s*([A-Z]{2}-[A-Z]{2}-\d{4})',
        r'([A-Z]{2}-[A-Z]{2}-\d{4})'  # Just the number
    ]

    for pattern in doc_number_patterns:
        match = re.search(pattern, sample, re.IGNORECASE)
        if match:
            metadata['doc_number'] = match.group(1)
            break

    # Try to find title (usually after document number or near top)
    # Look for lines that look like titles (5-60 chars, capitalized)
    for i, line in enumerate(lines[:30]):  # Check first 30 lines
        line = line.strip()
        if 5 < len(line) < 60 and not line.startswith('Page') and line[0].isupper():
            # Skip if it's just boilerplate
            if 'proprietary' not in line.lower() and 'copyright' not in line.lower():
                # Check if it looks like a title
                if line.count(' ') >= 1 and line.count(' ') <= 8:
                    metadata['doc_title'] = line
                    break

    return metadata

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
        topic: str,
        doc_metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Add a document's topic embedding to the collection."""
        # Build rich metadata for better search
        metadata = {
            'source_url': source_url,
            'topic': topic
        }

        # Add document metadata if provided
        if doc_metadata:
            if doc_metadata.get('doc_number'):
                metadata['doc_number'] = doc_metadata['doc_number']
            if doc_metadata.get('doc_title'):
                metadata['doc_title'] = doc_metadata['doc_title']

        self.collection.add(
            ids=[doc_id],
            embeddings=[topic_embedding],
            metadatas=[metadata]
        )
        logger.info(f"Added topic embedding for document {doc_id}")
        if doc_metadata and doc_metadata.get('doc_number'):
            logger.info(f"  Document number: {doc_metadata['doc_number']}")
            logger.info(f"  Document title: {doc_metadata.get('doc_title', 'N/A')}")

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

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a vague or short query to be more specific and better for retrieval.

        Examples:
        - "PTO Policy" → "What is the company policy for Paid Time Off (PTO)?"
        - "telecommuting" → "What are the company policies and procedures for telecommuting and remote work?"
        """
        # If query is already detailed (>50 chars with multiple words), don't rewrite
        if len(query) > 50 and len(query.split()) > 7:
            logger.info("Query is detailed enough, skipping rewrite")
            return query

        prompt = f"""You are helping to improve a search query for a corporate policy document database.

Original Query: "{query}"

Rewrite this query to be more specific and better for retrieving relevant policy documents.

Guidelines:
1. Expand abbreviations (e.g., "PTO" → "Paid Time Off")
2. Add context about what kind of information is needed
3. Frame as a clear question if it's not already
4. Keep it concise (under 100 words)
5. Focus on the core topic

Rewritten Query:"""

        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 80
                }
            )
            rewritten = response['response'].strip()
            # Clean up any quotes
            rewritten = rewritten.strip('"\'')

            logger.info(f"Query rewrite: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}, using original")
            return query

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

    def generate_topic(self, content: str, doc_metadata: Dict[str, str], max_chars: int = 3000) -> str:
        """
        Generate a rich, searchable topic/summary for a document.
        Uses document metadata and content.
        """
        # Use first portion for topic generation
        sample = content[:max_chars]

        # Build context from metadata
        metadata_context = ""
        if doc_metadata.get('doc_number'):
            metadata_context += f"Document Number: {doc_metadata['doc_number']}\n"
        if doc_metadata.get('doc_title'):
            metadata_context += f"Document Title: {doc_metadata['doc_title']}\n"

        prompt = f"""You are analyzing a corporate policy document. Create a searchable description that will help find this document when users ask questions.

{metadata_context}

Document Content (first portion):
{sample}

Create a comprehensive searchable description using this format:
"[Document Title] [Document Number]: [Main Topic] - [Key Details and Terms]"

Include:
- The document title and number
- Main policy topic
- Key terms that users might search for
- Important concepts covered

Keep it under 200 characters but make it specific and searchable.

Searchable Description:"""

        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'num_predict': 100
                }
            )
            topic = response['response'].strip()
            # Clean up the topic
            topic = re.sub(r'^(Topic:|Title:|Summary:|Description:)\s*', '', topic, flags=re.IGNORECASE)
            topic = topic.split('\n')[0]  # Take first line only

            # If LLM failed to include metadata, add it
            if doc_metadata.get('doc_number') and doc_metadata['doc_number'] not in topic:
                topic = f"{doc_metadata.get('doc_title', 'Policy')} {doc_metadata['doc_number']}: {topic}"

            return topic[:300]  # Increased limit for more detailed topics
        except Exception as e:
            logger.error(f"Failed to generate topic: {e}")
            # Fallback: construct from metadata
            fallback = doc_metadata.get('doc_title', 'Document')
            if doc_metadata.get('doc_number'):
                fallback += f" {doc_metadata['doc_number']}"
            return fallback[:200]

    def generate_answer_with_citations(
        self,
        question: str,
        documents: List[Dict[str, Any]]
    ) -> Tuple[str, List[Citation]]:
        """
        Generate an answer using retrieved documents and extract citations.
        Only includes relevant citations that are actually used.
        """
        # Build context from full documents with clear labeling
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source_url = doc.get('source_url', 'Unknown')
            content = doc.get('content', '')
            # Extract first 1000 chars to get document title/number
            header = content[:1000]

            context_parts.append(
                f"=== DOCUMENT {i} ===\n"
                f"Source: {source_url}\n"
                f"Content:\n{content}\n"
            )

        context = "\n---\n".join(context_parts)

        # Improved system instruction
        system_instruction = """You are a precise policy assistant. Answer ONLY using the provided documents.

INSTRUCTIONS:
1. Read ALL documents carefully
2. Identify which document(s) answer the question
3. Extract the SPECIFIC information that answers the question
4. Cite ONLY the documents you actually use
5. Use this format: "According to Document N, [information]"
6. Include brief quotes when helpful
7. If NO document answers the question, say "The provided documents do not contain information about [topic]"

IMPORTANT: Only cite documents that directly answer the question. Don't mention documents that aren't relevant."""

        prompt = f"""{system_instruction}

{context}

User Question: {question}

Answer (cite only relevant documents):"""

        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 400
                }
            )
            answer = response['response'].strip()

            # Extract citations - only from documents actually mentioned
            citations = self._extract_relevant_citations(answer, documents, question)

            return answer, citations

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Answer generation failed: {str(e)}"
            )

    def _extract_relevant_citations(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
        question: str
    ) -> List[Citation]:
        """
        Extract citations ONLY from documents actually mentioned in the answer.

        Looks for:
        1. "Document N" references in the answer
        2. Extracts excerpts only from those documents
        3. Does NOT include documents that weren't used
        """
        citations = []

        # Find which documents are actually mentioned in the answer
        mentioned_docs = set()

        # Pattern: "Document N" or "Document #N"
        doc_references = re.findall(r'Document\s+#?(\d+)', answer, re.IGNORECASE)
        for doc_num in doc_references:
            doc_idx = int(doc_num) - 1
            if 0 <= doc_idx < len(documents):
                mentioned_docs.add(doc_idx)

        # If no explicit document mentions, check if answer says "no information found"
        no_info_patterns = [
            r'do(?:es)?\s+not\s+contain',
            r'no\s+information',
            r'not\s+found',
            r'does(?:n\'t)?\s+address'
        ]

        answer_lower = answer.lower()
        if any(re.search(pattern, answer_lower) for pattern in no_info_patterns):
            # Answer says no relevant info found - return empty citations
            logger.info("Answer indicates no relevant information found")
            return []

        # If no documents mentioned explicitly, try to infer from answer content
        if not mentioned_docs:
            # Check if answer contains specific information (not just "no info")
            if len(answer) > 50 and not any(phrase in answer_lower for phrase in ['not found', 'no information', 'do not contain']):
                # Answer has content but no explicit citations
                # Use the first document as it's likely most relevant (highest score)
                mentioned_docs.add(0)
                logger.info("No explicit citations found, using first (most relevant) document")
            else:
                # Answer is vague or says no info - don't add citations
                return []

        # Extract relevant excerpts only from mentioned documents
        for doc_idx in sorted(mentioned_docs):
            if doc_idx < len(documents):
                doc = documents[doc_idx]
                content = doc.get('content', '')
                source_url = doc.get('source_url', 'Unknown')

                # Find most relevant excerpt from this document
                excerpt = self._find_best_excerpt(content, question, answer)

                if excerpt:
                    citations.append(Citation(
                        source_url=source_url,
                        excerpt=excerpt
                    ))

        logger.info(f"Extracted {len(citations)} citations from {len(mentioned_docs)} mentioned documents")
        return citations

    def _extract_citations(
        self,
        answer: str,
        documents: List[Dict[str, Any]]
    ) -> List[Citation]:
        """
        OLD METHOD - kept for compatibility.
        Use _extract_relevant_citations instead.
        """
        return self._extract_relevant_citations(answer, documents, "")

    def _find_best_excerpt(self, content: str, question: str, answer: str, max_length: int = 350) -> str:
        """
        Find the most relevant excerpt from a document based on question and answer.

        Tries to find sections that:
        1. Contain key terms from the question
        2. Relate to information in the answer
        3. Are not boilerplate/headers
        """
        # Extract key terms from question (simple approach)
        question_terms = set(re.findall(r'\b[A-Za-z]{4,}\b', question.lower()))
        # Common stop words to ignore
        stop_words = {'what', 'when', 'where', 'which', 'policy', 'company', 'does', 'have', 'about'}
        question_terms = question_terms - stop_words

        # Skip first 500 chars (headers)
        if len(content) > 500:
            content_sample = content[500:]
        else:
            content_sample = content

        # Split into paragraphs
        paragraphs = [p.strip() for p in content_sample.split('\n\n') if len(p.strip()) > 80]

        # Score paragraphs by relevance
        best_para = None
        best_score = 0

        boilerplate_terms = ['proprietary', 'copyright', 'uncontrolled', 'permission']

        for para in paragraphs[:20]:  # Check first 20 paragraphs
            # Skip boilerplate
            if any(term in para.lower() for term in boilerplate_terms):
                continue

            # Count matching question terms
            para_lower = para.lower()
            score = sum(1 for term in question_terms if term in para_lower)

            # Bonus for numbered sections (policy content)
            if re.search(r'^\d+\.\d+\s', para):
                score += 2

            if score > best_score:
                best_score = score
                best_para = para

        if best_para:
            excerpt = best_para[:max_length]
            if len(best_para) > max_length:
                # Try to end at sentence boundary
                last_period = excerpt.rfind('.')
                if last_period > max_length * 0.6:
                    excerpt = excerpt[:last_period + 1]
            return excerpt + ("..." if len(best_para) > max_length else "")

        # Fallback to _extract_relevant_excerpt
        return self._extract_relevant_excerpt(content, max_length)

    def _extract_relevant_excerpt(self, content: str, max_length: int = 300) -> str:
        """
        Extract a relevant excerpt from document content, skipping boilerplate.

        Tries to find:
        1. Numbered policy sections (5.0, 5.1, etc.)
        2. Substantial paragraphs after skipping headers
        3. Content with definitions or procedures
        """
        # Skip first 500 chars (usually headers/copyright)
        if len(content) > 500:
            content_sample = content[500:]
        else:
            content_sample = content

        # Try to find numbered sections (e.g., "5.1 General" or "1.0 Purpose")
        section_pattern = r'(\d+\.\d+\s+[A-Z][^\n]+(?:\n[^\n]+){1,3})'
        section_matches = re.findall(section_pattern, content_sample[:3000])

        if section_matches:
            # Use first significant section
            excerpt = section_matches[0].strip()
            if len(excerpt) > max_length:
                excerpt = excerpt[:max_length]
                # Try to end at sentence
                last_period = excerpt.rfind('.')
                if last_period > max_length * 0.7:
                    excerpt = excerpt[:last_period + 1]
            return excerpt + "..."

        # Fallback: Find substantial paragraphs
        paragraphs = [p.strip() for p in content_sample.split('\n\n') if len(p.strip()) > 100]

        # Skip paragraphs that are mostly boilerplate
        boilerplate_terms = ['proprietary', 'copyright', 'uncontrolled', 'permission']

        for para in paragraphs[:5]:  # Check first 5 paragraphs
            # Check if it's not boilerplate
            if not any(term in para.lower() for term in boilerplate_terms):
                excerpt = para[:max_length]
                if len(para) > max_length:
                    last_period = excerpt.rfind('.')
                    if last_period > max_length * 0.7:
                        excerpt = excerpt[:last_period + 1]
                return excerpt + "..."

        # Last resort: just take from content
        return content_sample[:max_length] + "..."

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

        # Extract document metadata
        logger.info("Extracting document metadata...")
        doc_metadata = extract_document_metadata(markdown_content)
        logger.info(f"Extracted metadata: {doc_metadata}")

        # Generate topic/summary using metadata
        logger.info("Generating topic summary...")
        topic = ollama_client.generate_topic(markdown_content, doc_metadata)
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

        # Store topic embedding in vector store with metadata
        vector_store.add_document(
            doc_id=doc_id,
            topic_embedding=topic_embedding,
            source_url=source_url,
            topic=topic,
            doc_metadata=doc_metadata
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

        # Optionally rewrite query for better retrieval
        search_query = question
        if request.rewrite_query:
            logger.info("Rewriting query for better retrieval...")
            search_query = ollama_client.rewrite_query(question)

        # Generate embedding for question (use rewritten version for search)
        logger.info("Generating query embedding...")
        query_embedding = ollama_client.generate_embedding(search_query)

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
