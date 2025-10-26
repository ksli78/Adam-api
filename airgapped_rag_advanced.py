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

        # Initialize query classifier for system queries
        from query_classifier import get_query_classifier
        self.query_classifier = get_query_classifier(
            ollama_host=OLLAMA_HOST,
            model_name=LLM_MODEL
        )

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
            # Step 1: Classify the query (system vs document query)
            classification = self.query_classifier.classify_query(question)

            # Step 2: Handle system queries (about the RAG system itself)
            if classification['query_type'] == 'system':
                logger.info(f"Detected system query, generating system response")
                system_answer = self.query_classifier.generate_system_response(question)

                # Replace newlines with HTML line breaks for display
                system_answer = system_answer.replace('\n', '<br>')

                return {
                    "answer": system_answer,
                    "citations": [],
                    "retrieval_stats": {
                        "query_type": "system",
                        "classification_confidence": classification['confidence'],
                        "child_chunks_retrieved": 0,
                        "parent_chunks_used": 0,
                        "message": "System query - no document retrieval performed"
                    }
                }

            # Step 3: Handle document queries (normal RAG retrieval)
            logger.info("Detected document query, proceeding with retrieval")

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

            # Check if we have insufficient results (weak matches)
            # If we retrieved very few chunks, it means the strict BM25 filter
            # eliminated most documents, indicating poor keyword match
            # With BM25 threshold at 0.95, even 1 chunk is a strong signal
            MIN_CHUNKS_THRESHOLD = 1

            if not parent_results or len(child_results) < MIN_CHUNKS_THRESHOLD:
                logger.warning(f"Insufficient results found: {len(child_results)} child chunks")
                return {
                    "answer": (
                        "I couldn't find relevant information about your question in the available documents.<br><br>"
                        "Here are some suggestions:<br>"
                        "• Try rephrasing your question with different keywords<br>"
                        "• Check the <a href='https://portal.amentumspacemissions.com/MS/Pages/MSDefaultHomePage.aspx' target='_blank'>Management System</a> "
                        "where all policy documents are housed<br>"
                        "• If you're looking for a specific policy, try including the policy number (e.g., EN-PO-XXXX)"
                    ),
                    "citations": [],
                    "confidence": 0.0,
                    "retrieval_stats": {
                        "query_type": "document",
                        "classification_confidence": classification.get('confidence', 'high'),
                        "child_chunks_retrieved": len(child_results),
                        "parent_chunks_used": 0,
                        "message": "Insufficient relevant documents found"
                    }
                }

            # Build context from parent chunks (include URLs for inline citations)
            context_parts = []
            for i, parent in enumerate(parent_results, 1):
                context_parts.append(
                    f"[Document {i}]\n"
                    f"Title: {parent['metadata'].get('document_title', 'Unknown')}\n"
                    f"URL: {parent['metadata'].get('source_url', '')}\n"
                    f"Section: {parent['metadata'].get('section_title', 'Unknown')}\n"
                    f"Content:\n{parent['text']}\n"
                )

            context = "\n\n---\n\n".join(context_parts)

            # Generate answer with LLM
            answer = await self._generate_answer(question, context, temperature)

            # Replace newlines with HTML line breaks for better display
            answer = answer.replace('\n', '<br>')

            # Build citations
            citations = []
            parent_chunk_ids = []
            for parent in parent_results:
                citations.append({
                    "source_url": parent['metadata'].get('source_url', ''),
                    "document_title": parent['metadata'].get('document_title', 'Unknown'),
                    "section_title": parent['metadata'].get('section_title', ''),
                    "section_number": parent['metadata'].get('section_number', ''),
                    "excerpt": parent['text'][:500] + "..." if len(parent['text']) > 500 else parent['text']
                })
                parent_chunk_ids.append(parent['id'])

            result = {
                "answer": answer,
                "citations": citations,
                "retrieval_stats": {
                    "query_type": "document",
                    "classification_confidence": classification.get('confidence', 'high'),
                    "child_chunks_retrieved": len(child_results),
                    "parent_chunks_used": len(parent_results),
                    "parent_chunk_ids": parent_chunk_ids,  # For feedback tracking
                    "metadata_filter": metadata_filter,
                    "use_hybrid": use_hybrid,
                    "bm25_weight": bm25_weight
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

        prompt = f"""You are a friendly, helpful assistant that answers questions using information from company documents.

QUESTION: {question}

AVAILABLE DOCUMENTS:
{context}

INSTRUCTIONS:
1. Answer the question naturally and conversationally - no need for phrases like "Based on the provided documents" or "According to Document X"
2. Use ONLY information from the documents above
3. When referencing a document, use inline HTML citations in this exact format: <span><a href="URL">FileName.pdf</a></span>
4. ONLY cite documents you actually use in your answer - don't mention documents you didn't reference
5. Include specific details like section numbers, amounts, dates when relevant
6. If the documents don't contain enough information, say so clearly
7. Keep your answer focused and helpful

EXAMPLE of inline citation:
"PTO is a paid time off program<span><a href="https://example.com/EN-PO-0301.pdf">EN-PO-0301.pdf</a></span> that varies based on years of service."

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
    temperature: float = 0.1  # Lower temperature for more consistent, deterministic responses
    metadata_filter: Optional[Dict[str, Any]] = None
    use_hybrid: bool = True  # Use hybrid search (BM25 + semantic) by default
    bm25_weight: float = 0.5  # Weight for BM25 vs semantic (0.5 = equal weight)


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    feedback_type: str  # "good" or "bad"
    citations: Optional[List[Dict[str, Any]]] = None
    retrieval_stats: Optional[Dict[str, Any]] = None
    user_comment: Optional[str] = None
    session_id: Optional[str] = None


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


@app.delete("/documents")
async def clear_all_documents():
    """Clear all documents from the document store."""
    try:
        # Get all documents
        documents = rag_pipeline.document_store.list_documents()

        # Delete each document
        deleted_count = 0
        for doc in documents:
            rag_pipeline.document_store.delete_document(doc['document_id'])
            deleted_count += 1

        logger.info(f"Cleared {deleted_count} documents from document store")

        return {
            "message": f"Successfully cleared {deleted_count} documents",
            "deleted_count": deleted_count,
            "statistics": rag_pipeline.document_store.get_statistics()
        }
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
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


@app.get("/debug/document/{document_id}")
async def debug_document(document_id: str):
    """
    Debug endpoint to inspect actual chunk content for a document.

    Shows what text is stored in parent and child chunks.
    """
    try:
        # Get parent chunks
        parent_chunks = rag_pipeline.document_store.parent_collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        # Get child chunks
        child_chunks = rag_pipeline.document_store.child_collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        result = {
            "document_id": document_id,
            "parent_chunks": {
                "count": len(parent_chunks['ids']),
                "chunks": [
                    {
                        "chunk_id": parent_chunks['ids'][i],
                        "metadata": parent_chunks['metadatas'][i],
                        "text": parent_chunks['documents'][i][:1000] + "..." if len(parent_chunks['documents'][i]) > 1000 else parent_chunks['documents'][i]
                    }
                    for i in range(len(parent_chunks['ids']))
                ]
            },
            "child_chunks": {
                "count": len(child_chunks['ids']),
                "chunks": [
                    {
                        "chunk_id": child_chunks['ids'][i],
                        "metadata": child_chunks['metadatas'][i],
                        "text": child_chunks['documents'][i][:500] + "..." if len(child_chunks['documents'][i]) > 500 else child_chunks['documents'][i]
                    }
                    for i in range(min(10, len(child_chunks['ids'])))  # Limit to first 10 child chunks
                ]
            }
        }

        return result

    except Exception as e:
        logger.error(f"Error debugging document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/extract-markdown")
async def debug_extract_markdown(file: UploadFile = File(...)):
    """
    Debug endpoint to see raw Docling markdown extraction.

    Upload a PDF and see what markdown Docling produces and how it gets
    parsed into sections. Useful for diagnosing section extraction issues.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file
    file_path = DOCS_DIR / f"debug_{uuid.uuid4()}_{file.filename}"

    try:
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        # Extract with Docling
        logger.info(f"Extracting PDF with Docling: {file.filename}")
        docling_result = rag_pipeline.docling_converter.convert(str(file_path))
        markdown_text = docling_result.document.export_to_markdown()

        # Extract sections
        sections = rag_pipeline.chunker.extract_sections_from_markdown(markdown_text)

        result = {
            "filename": file.filename,
            "raw_markdown_length": len(markdown_text),
            "raw_markdown_preview": markdown_text[:2000] + "..." if len(markdown_text) > 2000 else markdown_text,
            "sections_extracted": len(sections),
            "sections": [
                {
                    "title": s.title,
                    "section_number": s.section_number,
                    "level": s.level,
                    "text_length": len(s.text),
                    "text_preview": s.text[:500] + "..." if len(s.text) > 500 else s.text
                }
                for s in sections
            ]
        }

        # Clean up debug file
        file_path.unlink()

        return result

    except Exception as e:
        logger.error(f"Error extracting markdown: {e}")
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on a query-answer pair.

    Feedback is used to improve future retrievals by tracking which
    chunks produce good vs bad responses.

    Args:
        query: The original user query
        answer: The RAG system's answer
        feedback_type: "good" or "bad"
        citations: Optional list of citations from the response
        retrieval_stats: Optional retrieval statistics
        user_comment: Optional user comment
        session_id: Optional session identifier

    Returns:
        feedback_id and success message
    """
    try:
        from feedback_store import get_feedback_store

        # Extract chunk IDs from retrieval stats
        chunks_used = []
        if request.retrieval_stats and 'parent_chunk_ids' in request.retrieval_stats:
            # Use parent chunk IDs from retrieval stats (most reliable)
            chunks_used = request.retrieval_stats['parent_chunk_ids']
        elif request.citations:
            # Fallback: use source URLs as identifiers
            for citation in request.citations:
                source_url = citation.get('source_url', '')
                if source_url:
                    chunks_used.append(source_url)

        feedback_store = get_feedback_store()
        feedback_id = feedback_store.add_feedback(
            query=request.query,
            answer=request.answer,
            feedback_type=request.feedback_type,
            chunks_used=chunks_used,
            citations=request.citations,
            retrieval_stats=request.retrieval_stats,
            retrieval_method="hybrid" if request.retrieval_stats and request.retrieval_stats.get('use_hybrid') else "semantic",
            user_comment=request.user_comment,
            session_id=request.session_id
        )

        logger.info(
            f"Received {request.feedback_type} feedback (ID: {feedback_id}) "
            f"for query: '{request.query[:50]}...'"
        )

        return {
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully",
            "feedback_type": request.feedback_type
        }

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/analytics")
async def get_feedback_analytics(days: int = 30):
    """
    Get feedback analytics and statistics.

    Shows:
    - Overall satisfaction rate
    - Feedback breakdown by retrieval method
    - Best/worst performing chunks

    Args:
        days: Number of days to analyze (default: 30)

    Returns:
        Analytics summary
    """
    try:
        from feedback_store import get_feedback_store

        feedback_store = get_feedback_store()
        analytics = feedback_store.get_feedback_summary(days=days)

        return analytics

    except Exception as e:
        logger.error(f"Error getting feedback analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/recent")
async def get_recent_feedback(limit: int = 50):
    """
    Get recent feedback entries.

    Args:
        limit: Maximum number of entries to return (default: 50)

    Returns:
        List of recent feedback entries
    """
    try:
        from feedback_store import get_feedback_store

        feedback_store = get_feedback_store()
        recent = feedback_store.get_recent_feedback(limit=limit)

        return {"feedback": recent, "count": len(recent)}

    except Exception as e:
        logger.error(f"Error getting recent feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "airgapped_rag_advanced:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
