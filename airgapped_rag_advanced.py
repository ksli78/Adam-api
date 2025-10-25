"""
Advanced Air-Gapped RAG System with Production Features

Complete pipeline with:
- Docling for structure-aware PDF extraction
- Document cleaning (CUI banners, headers, noise removal)
- Semantic parent-child chunking
- LLM-based metadata extraction
- Dual ChromaDB collections (children for retrieval, parents for context)
- Local Ollama for answer generation

All processing runs locally - no 3rd party APIs required.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Docling for PDF extraction
from docling.document_converter import DocumentConverter

# Local services
from document_cleaner import get_document_cleaner
from semantic_chunker import get_semantic_chunker, DocumentSection
from metadata_extractor import get_metadata_extractor
from parent_child_store import get_parent_child_store

# Ollama for answer generation
import ollama

# FastAPI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "/data/airgapped_rag"))
DOCS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chromadb_advanced"

# Create directories
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")

# FastAPI app
app = FastAPI(
    title="Advanced Air-Gapped RAG API",
    description="Production RAG with semantic chunking, metadata extraction, and parent-child retrieval",
    version="2.0.0"
)


class AdvancedRAGPipeline:
    """
    Advanced RAG pipeline orchestrator.

    Coordinates all services for document ingestion and retrieval.
    """

    def __init__(self):
        """Initialize the advanced RAG pipeline."""
        logger.info("Initializing Advanced RAG Pipeline...")

        # Initialize services
        self.doc_cleaner = get_document_cleaner()
        self.chunker = get_semantic_chunker(
            parent_chunk_size=1500,
            child_chunk_size=300,
            chunk_overlap=50
        )
        self.metadata_extractor = get_metadata_extractor(
            model_name=LLM_MODEL,
            ollama_host=OLLAMA_HOST
        )
        self.document_store = get_parent_child_store(
            persist_directory=str(CHROMA_DIR)
        )

        # Initialize Docling converter
        logger.info("Initializing Docling converter...")
        self.docling_converter = DocumentConverter()

        # Initialize Ollama client for answer generation
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)

        logger.info("Advanced RAG Pipeline initialized successfully!")
        logger.info(f"Document store stats: {self.document_store.get_statistics()}")

    async def ingest_document(
        self,
        file_path: str,
        source_url: str,
        document_title: str = ""
    ) -> Dict[str, Any]:
        """
        Ingest a PDF document through the complete pipeline.

        Pipeline stages:
        1. Extract with Docling (preserves structure)
        2. Clean sections (remove noise)
        3. Extract metadata with LLM
        4. Create parent-child chunks
        5. Store in ChromaDB

        Args:
            file_path: Path to PDF file
            source_url: Source URL for the document
            document_title: Optional document title

        Returns:
            Dict with ingestion statistics
        """
        start_time = datetime.now()
        document_id = str(uuid.uuid4())

        logger.info(f"Starting document ingestion: {document_title or source_url}")

        try:
            # Stage 1: Extract with Docling
            logger.info("Stage 1: Extracting PDF with Docling...")
            docling_result = self.docling_converter.convert(file_path)

            # Convert to markdown
            markdown_text = docling_result.document.export_to_markdown()
            logger.info(f"Extracted {len(markdown_text)} characters of markdown")

            # Stage 2: Extract sections from markdown
            logger.info("Stage 2: Extracting document sections...")
            sections = self.chunker.extract_sections_from_markdown(markdown_text)
            logger.info(f"Extracted {len(sections)} sections")

            # Stage 3: Clean sections
            logger.info("Stage 3: Cleaning sections (removing noise)...")
            cleaned_sections = []
            for section in sections:
                cleaned_result = self.doc_cleaner.clean_section(
                    section_text=section.text,
                    section_title=section.title
                )

                if cleaned_result["is_valid"]:
                    # Update section with cleaned text
                    section.text = cleaned_result["cleaned_text"]
                    cleaned_sections.append(section)
                else:
                    logger.debug(
                        f"Skipping section '{section.title}' - "
                        f"too short after cleaning ({cleaned_result['cleaned_length']} chars)"
                    )

            logger.info(f"Retained {len(cleaned_sections)} valid sections after cleaning")

            if not cleaned_sections:
                raise ValueError("No valid content after cleaning - document may be empty or all noise")

            # Stage 4: Extract metadata with LLM
            logger.info("Stage 4: Extracting metadata with LLM...")
            full_text = "\n\n".join(s.text for s in cleaned_sections)
            doc_metadata = self.metadata_extractor.extract(
                document_text=full_text,
                document_title=document_title,
                document_filename=Path(file_path).name
            )
            logger.info(
                f"Metadata extracted: type={doc_metadata.document_type}, "
                f"topics={doc_metadata.primary_topics}, "
                f"confidence={doc_metadata.confidence:.2f}"
            )

            # Stage 5: Create parent-child chunks
            logger.info("Stage 5: Creating parent-child chunks...")
            parent_chunks, child_chunks = self.chunker.chunk_document(
                sections=cleaned_sections,
                document_title=document_title or Path(file_path).name,
                document_id=document_id
            )
            logger.info(
                f"Created {len(parent_chunks)} parent chunks and "
                f"{len(child_chunks)} child chunks"
            )

            # Stage 6: Store in ChromaDB
            logger.info("Stage 6: Storing in ChromaDB...")
            store_metadata = {
                "document_id": document_id,
                "source_url": source_url,
                "document_title": document_title,
                "document_type": doc_metadata.document_type,
                "summary": doc_metadata.summary,
                "primary_topics": ", ".join(doc_metadata.primary_topics),
                "keywords": ", ".join(doc_metadata.keywords),
                "departments": ", ".join(doc_metadata.departments),
                "ingestion_date": datetime.now().isoformat(),
                "extraction_confidence": doc_metadata.confidence
            }

            storage_stats = self.document_store.add_document_chunks(
                parent_chunks=parent_chunks,
                child_chunks=child_chunks,
                document_metadata=store_metadata
            )

            elapsed_time = (datetime.now() - start_time).total_seconds()

            result = {
                "document_id": document_id,
                "source_url": source_url,
                "document_title": document_title,
                "message": "Document ingested successfully",
                "statistics": {
                    "sections_extracted": len(sections),
                    "sections_after_cleaning": len(cleaned_sections),
                    "parent_chunks": len(parent_chunks),
                    "child_chunks": len(child_chunks),
                    "ingestion_time_seconds": round(elapsed_time, 2),
                    "document_metadata": doc_metadata.to_dict(),
                    **storage_stats
                }
            }

            logger.info(f"Document ingestion completed in {elapsed_time:.1f}s: {result}")
            return result

        except Exception as e:
            logger.error(f"Error ingesting document: {e}", exc_info=True)
            raise

    async def query(
        self,
        question: str,
        top_k: int = 10,
        parent_limit: int = 3,
        metadata_filter: Dict[str, Any] = None,
        temperature: float = 0.3,
        use_hybrid: bool = True,
        bm25_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Query the RAG system with parent-child retrieval.

        Process:
        1. Retrieve top_k child chunks (hybrid: BM25 + semantic)
        2. Expand to parent chunks (context)
        3. Pass parent chunks to LLM
        4. Generate answer with citations

        Args:
            question: User's question
            top_k: Number of child chunks to retrieve
            parent_limit: Maximum number of parent chunks for LLM
            metadata_filter: Optional metadata filter
            temperature: LLM temperature
            use_hybrid: Use hybrid search (BM25 + semantic)
            bm25_weight: Weight for BM25 scores (0.0-1.0)

        Returns:
            Dict with answer and citations
        """
        logger.info(f"Processing query: {question} (hybrid={use_hybrid}, bm25_weight={bm25_weight})")

        try:
            # Retrieve with parent expansion (using hybrid search)
            child_results, parent_results = self.document_store.retrieve_with_parent_expansion(
                query=question,
                top_k=top_k,
                expand_to_parents=True,
                parent_limit=parent_limit,
                metadata_filter=metadata_filter,
                use_hybrid=use_hybrid,
                bm25_weight=bm25_weight
            )

            if not parent_results:
                logger.warning("No relevant documents found")
                return {
                    "answer": "I don't have enough information to answer this question based on the available documents.",
                    "citations": [],
                    "confidence": 0.0
                }

            # Build context from parent chunks
            context_parts = []
            for i, parent in enumerate(parent_results, 1):
                context_parts.append(
                    f"[Document {i}]\n"
                    f"Title: {parent['metadata'].get('document_title', 'Unknown')}\n"
                    f"Section: {parent['metadata'].get('section_title', 'Unknown')}\n"
                    f"Content:\n{parent['text']}\n"
                )

            context = "\n\n---\n\n".join(context_parts)

            # Generate answer with LLM
            answer = await self._generate_answer(question, context, temperature)

            # Build citations
            citations = []
            for parent in parent_results:
                citations.append({
                    "source_url": parent['metadata'].get('source_url', ''),
                    "document_title": parent['metadata'].get('document_title', 'Unknown'),
                    "section_title": parent['metadata'].get('section_title', ''),
                    "section_number": parent['metadata'].get('section_number', ''),
                    "excerpt": parent['text'][:500] + "..." if len(parent['text']) > 500 else parent['text']
                })

            result = {
                "answer": answer,
                "citations": citations,
                "retrieval_stats": {
                    "child_chunks_retrieved": len(child_results),
                    "parent_chunks_used": len(parent_results),
                    "metadata_filter": metadata_filter
                }
            }

            logger.info("Query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise

    async def _generate_answer(
        self,
        question: str,
        context: str,
        temperature: float
    ) -> str:
        """Generate answer using Ollama LLM."""

        prompt = f"""You are a helpful assistant that answers questions based on provided documents.

QUESTION: {question}

CONTEXT FROM DOCUMENTS:
{context}

INSTRUCTIONS:
1. Read the context carefully
2. Answer the question based ONLY on the information in the context
3. Cite which document number you're referencing (e.g., "According to Document 1...")
4. If the context doesn't contain enough information, say so clearly
5. Be specific and include relevant details (section numbers, amounts, dates, etc.)
6. Keep your answer focused and concise

ANSWER:"""

        logger.debug("Calling Ollama to generate answer...")

        try:
            response = self.ollama_client.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": 500
                }
            )

            answer = response['response'].strip()
            logger.debug(f"Generated answer: {answer[:200]}...")

            return answer

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Error generating answer: {str(e)}"


# Initialize pipeline (singleton)
rag_pipeline = AdvancedRAGPipeline()


# API Models
class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 10
    parent_limit: int = 3
    temperature: float = 0.3
    metadata_filter: Optional[Dict[str, Any]] = None
    use_hybrid: bool = True  # Use hybrid search (BM25 + semantic) by default
    bm25_weight: float = 0.5  # Weight for BM25 vs semantic (0.5 = equal weight)


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]


# API Endpoints

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    source_url: str = Form(...)
):
    """
    Upload and process a PDF document.

    Complete pipeline: extract, clean, chunk, extract metadata, store.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file
    file_path = DOCS_DIR / f"{uuid.uuid4()}_{file.filename}"

    try:
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        # Process document
        result = await rag_pipeline.ingest_document(
            file_path=str(file_path),
            source_url=source_url,
            document_title=file.filename
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with hybrid search (BM25 + semantic).

    Retrieves child chunks using hybrid search, expands to parents, generates answer.

    Hybrid search combines:
    - BM25: Keyword/lexical matching (good for exact terms)
    - Semantic: Embedding similarity (good for concepts)

    Set use_hybrid=False for pure semantic search.
    Adjust bm25_weight (0.0-1.0) to control BM25 vs semantic influence.
    """
    try:
        result = await rag_pipeline.query(
            question=request.prompt,
            top_k=request.top_k,
            parent_limit=request.parent_limit,
            metadata_filter=request.metadata_filter,
            temperature=request.temperature,
            use_hybrid=request.use_hybrid,
            bm25_weight=request.bm25_weight
        )

        return result

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all documents in the system."""
    try:
        documents = rag_pipeline.document_store.list_documents()
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    try:
        result = rag_pipeline.document_store.delete_document(document_id)
        return result
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """Get system statistics."""
    try:
        stats = rag_pipeline.document_store.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "airgapped_rag_advanced:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
