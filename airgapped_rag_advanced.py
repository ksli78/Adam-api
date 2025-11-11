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
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

# Fix for Windows symlink permission issue with Hugging Face Hub
# Must be set before importing any HF-dependent libraries (like docling)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Docling for PDF extraction
from docling.document_converter import DocumentConverter

# Local services
from document_cleaner import get_document_cleaner
from semantic_chunker import get_semantic_chunker, DocumentSection
from metadata_extractor import get_metadata_extractor
from parent_child_store import get_parent_child_store
from sql_routes import sql_router

# Ollama for answer generation
import ollama

# FastAPI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
# Remote Ollama server on development/production machine with 32GB VRAM (2 GPUs)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://adam.amentumspacemissions.com:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small:22b")  # Mistral Small 22B - excellent for RAG (~22GB VRAM)

# LLM Context window configuration
# Mistral Small supports up to 128K tokens, we use 16K for optimal VRAM usage
# 16K is sufficient for ~20 documents with questions in Stage 2 selection
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "16384"))  # 16K tokens

# FastAPI app
app = FastAPI(
    title="Advanced Air-Gapped RAG API",
    description="Production RAG with semantic chunking, metadata extraction, and parent-child retrieval",
    version="2.0.0"
)

# Add CORS middleware to allow cross-origin requests
# This is needed for the HTML demo page and frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict this in production to specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include SQL query routes
app.include_router(sql_router)


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
            ollama_host=OLLAMA_HOST,
            context_window=LLM_CONTEXT_WINDOW
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
                "answerable_questions": " || ".join(doc_metadata.answerable_questions),  # Use || as separator
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
            # Step 1: Use hybrid search to get top 30 candidates (casts wide net)
            child_results, _ = self.document_store.retrieve_with_parent_expansion(
                query=question,
                top_k=30,  # Get more candidates for reranking
                expand_to_parents=False,  # Don't expand yet - we'll rerank first
                metadata_filter=metadata_filter,
                use_hybrid=use_hybrid,
                bm25_weight=bm25_weight
            )

            # Check if we have insufficient results
            MIN_CHUNKS_THRESHOLD = 1
            if len(child_results) < MIN_CHUNKS_THRESHOLD:
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
                        "child_chunks_retrieved": len(child_results),
                        "parent_chunks_used": 0,
                        "message": "Insufficient relevant documents found"
                    }
                }

            # Step 2: SEMANTIC RERANKING - Sort by semantic score only (ignore BM25 noise)
            logger.info(f"Reranking {len(child_results)} chunks by semantic similarity...")
            child_results_reranked = sorted(child_results, key=lambda x: x.get('semantic_score', 0), reverse=True)

            # Take top 5 by semantic similarity
            SEMANTIC_TOP_K = 5
            top_semantic_chunks = child_results_reranked[:SEMANTIC_TOP_K]

            logger.info(f"Top {SEMANTIC_TOP_K} chunks by semantic score:")
            for i, chunk in enumerate(top_semantic_chunks, 1):
                logger.info(
                    f"  Rank {i}: {chunk['metadata'].get('document_title', 'Unknown')} - "
                    f"{chunk['metadata'].get('section_title', '')} "
                    f"(semantic={chunk.get('semantic_score', 0):.3f})"
                )

            # Step 3: Expand top semantic chunks to parents using existing method
            # Extract child IDs from top semantic chunks
            top_child_ids = [chunk['id'] for chunk in top_semantic_chunks]

            # Get parent chunks for these child IDs
            parent_ids_seen = set()
            parent_results = []

            for child_id in top_child_ids:
                # Find the child chunk to get its parent_id
                child_chunk = next((c for c in top_semantic_chunks if c['id'] == child_id), None)
                if not child_chunk:
                    continue

                # Get parent_id from child metadata (field name might be 'parent_id' or 'parent_chunk_id')
                parent_id = child_chunk['metadata'].get('parent_id') or child_chunk['metadata'].get('parent_chunk_id')

                logger.debug(f"Child {child_id[:8]}... has parent_id: {parent_id}")

                if parent_id and parent_id not in parent_ids_seen:
                    try:
                        # Fetch the parent chunk
                        parent_data = self.document_store.parent_collection.get(
                            ids=[parent_id],
                            include=["documents", "metadatas"]
                        )

                        if parent_data and parent_data['ids']:
                            parent_results.append({
                                "id": parent_data['ids'][0],
                                "text": parent_data['documents'][0],
                                "metadata": parent_data['metadatas'][0]
                            })
                            parent_ids_seen.add(parent_id)
                            logger.debug(f"Added parent {parent_id[:8]}...")

                            # Limit to parent_limit
                            if len(parent_results) >= parent_limit:
                                break
                    except Exception as e:
                        logger.warning(f"Failed to fetch parent {parent_id}: {e}")

            logger.info(f"Expanded to {len(parent_results)} parent chunks")

            if not parent_results:
                logger.warning(f"No parent chunks found after reranking. Child metadata keys: {list(top_semantic_chunks[0]['metadata'].keys()) if top_semantic_chunks else 'none'}")
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
                    "semantic_reranked_top": SEMANTIC_TOP_K,
                    "parent_chunks_used": len(parent_results),
                    "retrieval_method": "hybrid_search_with_semantic_reranking",
                    "metadata_filter": metadata_filter
                }
            }

            logger.info("Query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise

    async def query_stream(
        self,
        question: str,
        top_k: int = 30,
        parent_limit: int = 5,
        metadata_filter: Dict[str, Any] = None,
        temperature: float = 0.3,
        use_hybrid: bool = True,
        bm25_weight: float = 0.2
    ) -> AsyncGenerator[str, None]:
        """
        Stream query response with real-time status updates and token-by-token LLM generation.

        Yields Server-Sent Events (SSE) format messages:
        - status: Retrieval progress updates
        - sources: Retrieved document citations
        - token: Individual tokens from LLM generation
        - done: Completion signal with final stats
        - error: Error message if something fails

        Args:
            Same as query() method

        Yields:
            SSE-formatted JSON messages
        """
        try:
            # Yield initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Finding relevant documents...'})}\n\n"

            logger.info(f"Processing streaming query: {question[:100]}... (hybrid={use_hybrid}, bm25_weight={bm25_weight})")

            # Step 1: Use hybrid search to get top 30 candidates (casts wide net)
            child_results, _ = self.document_store.retrieve_with_parent_expansion(
                query=question,
                top_k=30,  # Get more candidates for reranking
                expand_to_parents=False,  # Don't expand yet - we'll rerank first
                metadata_filter=metadata_filter,
                use_hybrid=use_hybrid,
                bm25_weight=bm25_weight
            )

            # Check if we have insufficient results
            MIN_CHUNKS_THRESHOLD = 1
            if len(child_results) < MIN_CHUNKS_THRESHOLD:
                logger.warning(f"Insufficient results found: {len(child_results)} child chunks")
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant documents found'})}\n\n"
                return

            # Step 2: SEMANTIC RERANKING - Sort by semantic score only (ignore BM25 noise)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing document relevance...'})}\n\n"

            logger.info(f"Reranking {len(child_results)} chunks by semantic similarity...")
            child_results_reranked = sorted(child_results, key=lambda x: x.get('semantic_score', 0), reverse=True)

            # Take top 5 by semantic similarity
            SEMANTIC_TOP_K = 5
            top_semantic_chunks = child_results_reranked[:SEMANTIC_TOP_K]

            # Step 3: Expand top semantic chunks to parents
            top_child_ids = [chunk['id'] for chunk in top_semantic_chunks]
            parent_ids_seen = set()
            parent_results = []

            for child_id in top_child_ids:
                child_chunk = next((c for c in top_semantic_chunks if c['id'] == child_id), None)
                if not child_chunk:
                    continue

                parent_id = child_chunk['metadata'].get('parent_id') or child_chunk['metadata'].get('parent_chunk_id')

                if parent_id and parent_id not in parent_ids_seen:
                    try:
                        parent_data = self.document_store.parent_collection.get(
                            ids=[parent_id],
                            include=["documents", "metadatas"]
                        )

                        if parent_data and parent_data['ids']:
                            parent_results.append({
                                "id": parent_data['ids'][0],
                                "text": parent_data['documents'][0],
                                "metadata": parent_data['metadatas'][0]
                            })
                            parent_ids_seen.add(parent_id)

                            if len(parent_results) >= parent_limit:
                                break
                    except Exception as e:
                        logger.warning(f"Failed to fetch parent {parent_id}: {e}")

            logger.info(f"Expanded to {len(parent_results)} parent chunks")

            if not parent_results:
                logger.warning("No parent chunks found after reranking")
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant information found'})}\n\n"
                return

            # Build context from parent chunks (MUST include URLs for inline citations!)
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

            # Build citations (include source_url for clickable links)
            citations = []
            for parent in parent_results:
                citations.append({
                    "source_url": parent['metadata'].get('source_url', ''),
                    "document_title": parent['metadata'].get('document_title', 'Unknown'),
                    "section_title": parent['metadata'].get('section_title', ''),
                    "section_number": parent['metadata'].get('section_number', ''),
                    "excerpt": parent['text'][:500] + "..." if len(parent['text']) > 500 else parent['text']
                })

            yield f"data: {json.dumps({'type': 'sources', 'citations': citations})}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"

            # Step 4: Stream LLM response token by token
            # Use EXACT SAME PROMPT as regular endpoint for consistent quality
            prompt = f"""Answer the following question using ONLY the information from the documents provided below.

QUESTION:
{question}

DOCUMENTS:
{context}

INSTRUCTIONS:
- Provide a direct, helpful answer to the question
- Use information ONLY from the documents above
- Include specific details (section numbers, dates, amounts) when relevant
- IMPORTANT: Add inline citations after EACH claim or bullet point using this format: (<span><a href="URL">FileName.pdf</a></span>)
- Place citations immediately after the relevant statement, before the period
- If information is missing, clearly state what cannot be answered

CITATION EXAMPLE:
✓ CORRECT: "Employees must submit requests via the Decisions tool (<span><a href="https://...">EN-PO-0301.pdf</a></span>)."
✗ WRONG: "Employees must submit requests via the Decisions tool. For more details, see EN-PO-0301.pdf."

Now provide your answer with inline citations after each point:"""

            logger.info("Starting LLM streaming generation...")
            logger.info("Calling Ollama API with stream=True...")

            # Call Ollama with streaming enabled
            response = self.ollama_client.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": 2000,
                    "num_ctx": LLM_CONTEXT_WINDOW
                },
                stream=True  # Enable streaming!
            )

            logger.info("Ollama stream started, beginning iteration...")

            # Stream tokens as they're generated
            full_answer = ""
            token_count = 0
            chunk_count = 0

            for chunk in response:
                chunk_count += 1

                # Log EVERY chunk to debug buffering issue (chunk is a Pydantic GenerateResponse object)
                if chunk_count <= 5 or chunk_count % 10 == 0:
                    logger.info(f"Received chunk #{chunk_count}: done={getattr(chunk, 'done', None)}, has_response={hasattr(chunk, 'response')}")

                # Access response attribute directly (chunk is GenerateResponse object, not dict)
                if hasattr(chunk, 'response') and chunk.response:
                    token = chunk.response
                    full_answer += token
                    token_count += 1

                    # Replace newlines with <br> for HTML display (same as regular endpoint)
                    display_token = token.replace('\n', '<br>')

                    # Yield each token immediately
                    yield f"data: {json.dumps({'type': 'token', 'content': display_token})}\n\n"

                    # Log frequently to verify streaming is working
                    if token_count in [1, 5, 10, 20, 50, 100] or token_count % 50 == 0:
                        logger.info(f"✓ Streamed {token_count} tokens so far (chunk #{chunk_count})...")

            logger.info(f"Streaming complete: generated {token_count} tokens in {chunk_count} chunks, {len(full_answer)} characters")

            # Yield completion with final stats
            yield f"data: {json.dumps({'type': 'done', 'stats': {'child_chunks_retrieved': len(child_results), 'parent_chunks_used': len(parent_results), 'answer_length': len(full_answer), 'tokens_streamed': token_count}})}\n\n"

        except Exception as e:
            logger.error(f"Error processing streaming query: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    async def query_with_llm_selection(
        self,
        question: str,
        max_documents: int = 3,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Query the RAG system using LLM-based document selection.

        New flow:
        1. Use LLM to select relevant documents based on answerable_questions
        2. Fetch full text of selected documents
        3. Send full documents + question to LLM for answer generation
        4. Return answer with document-level citations

        Args:
            question: User's question
            max_documents: Maximum number of documents to select
            temperature: LLM temperature for answer generation

        Returns:
            Dict with answer and citations
        """
        logger.info(f"Processing query with LLM document selection: {question}")

        try:
            # Step 1: Use LLM to select relevant documents
            selected_doc_ids = await self._select_documents_with_llm(
                user_question=question,
                max_documents=max_documents
            )

            if not selected_doc_ids:
                logger.warning("No documents selected by LLM")
                return {
                    "answer": (
                        "I couldn't find any documents that can answer your question.<br><br>"
                        "Here are some suggestions:<br>"
                        "• Try rephrasing your question with different keywords<br>"
                        "• Check the <a href='https://portal.amentumspacemissions.com/MS/Pages/MSDefaultHomePage.aspx' target='_blank'>Management System</a> "
                        "where all policy documents are housed<br>"
                        "• If you're looking for a specific policy, try including the policy number (e.g., EN-PO-XXXX)"
                    ),
                    "citations": [],
                    "confidence": 0.0,
                    "retrieval_stats": {
                        "documents_selected": 0,
                        "selection_method": "llm_based",
                        "message": "No relevant documents found"
                    }
                }

            # Step 2: Fetch full text for selected documents
            logger.info(f"Fetching full text for {len(selected_doc_ids)} documents")
            documents_with_text = []

            for doc_id in selected_doc_ids:
                full_text = self.document_store.get_full_document_text(doc_id)
                if full_text:
                    # Get document metadata
                    all_docs = self.document_store.get_all_documents_with_metadata()
                    doc_metadata = next(
                        (d for d in all_docs if d['document_id'] == doc_id),
                        None
                    )

                    if doc_metadata:
                        documents_with_text.append({
                            "document_id": doc_id,
                            "text": full_text,
                            "metadata": doc_metadata
                        })

            if not documents_with_text:
                logger.error("Failed to fetch text for selected documents")
                return {
                    "answer": "Error: Could not retrieve document content.",
                    "citations": [],
                    "confidence": 0.0,
                    "retrieval_stats": {
                        "documents_selected": len(selected_doc_ids),
                        "documents_retrieved": 0,
                        "selection_method": "llm_based",
                        "message": "Failed to fetch document text"
                    }
                }

            # Step 3: Build context from full documents
            context_parts = []
            for i, doc in enumerate(documents_with_text, 1):
                context_parts.append(
                    f"[Document {i}]\n"
                    f"Title: {doc['metadata']['document_title']}\n"
                    f"URL: {doc['metadata']['source_url']}\n"
                    f"Type: {doc['metadata']['document_type']}\n"
                    f"Summary: {doc['metadata']['summary']}\n\n"
                    f"Full Content:\n{doc['text']}\n"
                )

            context = "\n\n" + "="*80 + "\n\n".join(context_parts)

            # Step 4: Generate answer with LLM
            answer = await self._generate_answer(question, context, temperature)

            # Replace newlines with HTML line breaks
            answer = answer.replace('\n', '<br>')

            # Step 5: Build citations (document-level, not chunk-level)
            citations = []
            for doc in documents_with_text:
                citations.append({
                    "source_url": doc['metadata']['source_url'],
                    "document_title": doc['metadata']['document_title'],
                    "document_type": doc['metadata']['document_type'],
                    "summary": doc['metadata']['summary']
                })

            result = {
                "answer": answer,
                "citations": citations,
                "retrieval_stats": {
                    "documents_selected": len(selected_doc_ids),
                    "documents_used": len(documents_with_text),
                    "selection_method": "llm_based",
                    "selected_document_ids": selected_doc_ids
                }
            }

            logger.info("Query with LLM selection processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query with LLM selection: {e}", exc_info=True)
            raise

    async def _select_documents_with_llm(
        self,
        user_question: str,
        max_documents: int = 5
    ) -> List[str]:
        """
        Use LLM to intelligently select which documents can answer the user's question.

        Uses a two-stage approach:
        1. Stage 1: Use BM25/semantic search to narrow down to top 30 candidates
        2. Stage 2: Use LLM to select the most relevant documents from candidates

        This approach:
        - Handles large document collections (100s of documents)
        - Stays within LLM context window limits
        - Combines fast keyword matching with semantic reasoning

        Args:
            user_question: The user's natural language question
            max_documents: Maximum number of documents to select

        Returns:
            List of document IDs that can answer the question
        """
        logger.info(f"Using LLM to select documents for question: {user_question}")

        # Get all documents with metadata
        all_documents = self.document_store.get_all_documents_with_metadata()

        if not all_documents:
            logger.warning("No documents found in the store")
            return []

        logger.info(f"Total documents in system: {len(all_documents)}")

        # STAGE 1: Use hybrid search to narrow down candidates
        # This handles large document collections efficiently
        CANDIDATE_LIMIT = 20  # Number of candidates to send to LLM (reduced from 30 to fit in 16K context)

        logger.info(f"Stage 1: Using hybrid search to find top {CANDIDATE_LIMIT} candidate documents")

        # Use the existing hybrid search to get candidate chunks
        candidate_chunks, _ = self.document_store.retrieve_with_parent_expansion(
            query=user_question,
            top_k=100,  # Get more chunks to cover more documents
            expand_to_parents=False,  # Don't need parent expansion yet
            use_hybrid=True,
            bm25_weight=0.3  # Balance between BM25 and semantic
        )

        # Extract unique document IDs from candidate chunks
        candidate_doc_ids = []
        seen_doc_ids = set()
        for chunk in candidate_chunks:
            doc_id = chunk['metadata'].get('document_id')
            if doc_id and doc_id not in seen_doc_ids:
                candidate_doc_ids.append(doc_id)
                seen_doc_ids.add(doc_id)
                if len(candidate_doc_ids) >= CANDIDATE_LIMIT:
                    break

        if not candidate_doc_ids:
            logger.warning("No candidate documents found in Stage 1 (hybrid search)")
            return []

        # Filter all_documents to only include candidates
        candidate_documents = [doc for doc in all_documents if doc['document_id'] in candidate_doc_ids]

        logger.info(f"Stage 1 complete: Narrowed to {len(candidate_documents)} candidate documents")

        # Log the candidate document titles for debugging
        candidate_titles = [f"{doc['document_title']} ({doc['document_type']})" for doc in candidate_documents[:10]]
        logger.info(f"Top 10 candidates: {', '.join(candidate_titles)}")
        if len(candidate_documents) > 10:
            logger.info(f"... and {len(candidate_documents) - 10} more candidates")

        # STAGE 2: Use LLM to select best documents from candidates
        logger.info(f"Stage 2: Using LLM to select best documents from candidates")

        # Build document catalog for LLM (using only candidates from Stage 1)
        doc_catalog = []
        doc_number_to_id = {}  # Map document numbers to IDs for fallback parsing

        for i, doc in enumerate(candidate_documents, 1):
            doc_number_to_id[str(i)] = doc['document_id']

            doc_info = f"""
Document {i}:
- ID: {doc['document_id']}
- Title: {doc['document_title']}
- Type: {doc['document_type']}
- Summary: {doc['summary']}
- Topics: {doc['primary_topics']}
- Questions this document can answer:
"""
            for q in doc['answerable_questions']:
                doc_info += f"  * {q}\n"

            doc_catalog.append(doc_info)

        catalog_text = "\n".join(doc_catalog)

        logger.info(f"Document catalog size: {len(catalog_text)} characters, {len(candidate_documents)} documents")

        # Log the catalog for debugging
        if len(catalog_text) > 3000:
            logger.info(f"Document catalog (first 3000 chars):\n{catalog_text[:3000]}...")
        else:
            logger.info(f"Full document catalog:\n{catalog_text}")

        # Build prompt for LLM
        prompt = f"""You are a document selection expert. Your task is to identify which documents can answer a user's question.

USER'S QUESTION:
{user_question}

AVAILABLE DOCUMENTS:
{catalog_text}

INSTRUCTIONS:
1. Read the user's question carefully
2. For each document, check if ANY of its "Questions this document can answer" are similar to or can answer the user's question
3. Also consider if the document summary or topics are relevant
4. Select ONLY documents that can actually answer the user's question
5. If you find matching documents, select up to {max_documents} of the most relevant ones
6. Return a JSON array with the "ID" field values (the long alphanumeric strings)

CRITICAL - DOCUMENT IDENTIFICATION:
- Each document has an "ID" field (e.g., "13fb5d84-7eeb-40b2-9410-a26d9aafe7d6")
- You MUST return these exact ID values in your JSON array
- DO NOT return the document number (e.g., "Document 1", "Document 2")
- Example: ["13fb5d84-7eeb-40b2-9410-a26d9aafe7d6", "11ceb2fc-df0a-4460-8802-9ed6199b0809"]

IMPORTANT:
- Match questions by MEANING, not just exact words (e.g., "work hours" matches "working time", "schedule", "overtime")
- If NO documents match, return an empty array: []
- Return ONLY the JSON array, no explanations

Your response (JSON array of ID values only):"""

        logger.info(f"Sending document selection prompt to LLM (question: '{user_question}')")

        try:
            response = self.ollama_client.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for more deterministic selection
                    "num_predict": 300,  # Increased to allow for more IDs
                    "num_ctx": LLM_CONTEXT_WINDOW  # Context window size (32K for llama3.1:8b)
                }
            )

            # Parse the response to extract document IDs
            response_text = response['response'].strip()
            logger.info(f"LLM document selection raw response: {response_text}")

            # Try to parse as JSON
            import json
            import re

            # Extract JSON array from response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                selected_ids = json.loads(json_match.group(0))

                # Handle case where LLM returns document numbers instead of IDs
                # Convert any numeric strings to document IDs using our mapping
                converted_ids = []
                for item in selected_ids:
                    if str(item) in doc_number_to_id:
                        # LLM returned a document number, convert it
                        actual_id = doc_number_to_id[str(item)]
                        converted_ids.append(actual_id)
                        logger.info(f"LLM returned document number '{item}', converted to ID: {actual_id}")
                    elif item in [doc['document_id'] for doc in candidate_documents]:
                        # LLM correctly returned a document ID
                        converted_ids.append(item)
                    else:
                        logger.warning(f"Ignoring invalid ID/number: {item}")

                selected_ids = converted_ids

                # Log which documents were selected with their titles
                if selected_ids:
                    selected_titles = []
                    for doc_id in selected_ids:
                        doc = next((d for d in candidate_documents if d['document_id'] == doc_id), None)
                        if doc:
                            selected_titles.append(f"{doc['document_title']} (type: {doc['document_type']})")

                    logger.info(f"Stage 2 complete: LLM selected {len(selected_ids)} final documents:")
                    for title in selected_titles:
                        logger.info(f"  - {title}")
                else:
                    logger.warning("LLM returned empty selection - no matching documents found in candidates")

                return selected_ids
            else:
                logger.warning("Could not parse JSON from LLM response, falling back to empty list")
                logger.warning(f"Unparseable response was: {response_text}")
                return []

        except Exception as e:
            logger.error(f"Error in LLM document selection: {e}", exc_info=True)
            return []

    async def _generate_answer(
        self,
        question: str,
        context: str,
        temperature: float
    ) -> str:
        """Generate answer using Ollama LLM."""

        prompt = f"""Answer the following question using ONLY the information from the documents provided below.

QUESTION:
{question}

DOCUMENTS:
{context}

INSTRUCTIONS:
- Provide a direct, helpful answer to the question
- Use information ONLY from the documents above
- Include specific details (section numbers, dates, amounts) when relevant
- IMPORTANT: Add inline citations after EACH claim or bullet point using this format: (<span><a href="URL">FileName.pdf</a></span>)
- Place citations immediately after the relevant statement, before the period
- If information is missing, clearly state what cannot be answered

CITATION EXAMPLE:
✓ CORRECT: "Employees must submit requests via the Decisions tool (<span><a href="https://...">EN-PO-0301.pdf</a></span>)."
✗ WRONG: "Employees must submit requests via the Decisions tool. For more details, see EN-PO-0301.pdf."

Now provide your answer with inline citations after each point:"""

        logger.debug("Calling Ollama to generate answer...")

        try:
            response = self.ollama_client.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": 2000,  # High limit for comprehensive answers that tie together multiple policy documents
                    "num_ctx": LLM_CONTEXT_WINDOW  # Context window size (32K for llama3.1:8b)
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
    top_k: int = 30  # Get 30 candidates for semantic reranking
    parent_limit: int = 5  # Top 5 parents after semantic reranking
    temperature: float = 0.3  # LLM temperature for answer generation
    metadata_filter: Optional[Dict[str, Any]] = None
    use_hybrid: bool = True  # Use hybrid search (BM25 + semantic) by default
    bm25_weight: float = 0.2  # Weight for BM25 (0.2 = 80% semantic, 20% BM25)
    use_llm_selection: bool = False  # DEPRECATED: Use semantic reranking instead


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
    Query the RAG system using HYBRID SEARCH + SEMANTIC RERANKING:

    Process:
    1. Stage 1: Hybrid search (BM25 + semantic) retrieves 30 candidate chunks
    2. Stage 2: Semantic reranking picks top 5 chunks by semantic similarity
    3. Stage 3: Expand to parent chunks for rich context
    4. Stage 4: LLM generates answer with inline citations

    Benefits:
    - BM25 casts wide net (handles keyword variations)
    - Semantic reranking eliminates BM25 noise (focuses on meaning)
    - Fast (< 15 seconds) and accurate
    - Only top 5 most relevant sections sent to LLM

    Parameters:
    - prompt: User's question
    - top_k: Candidates for reranking (default: 30, don't change)
    - parent_limit: Max parents after reranking (default: 5)
    - use_hybrid: Enable hybrid search (default: True)
    - bm25_weight: BM25 weight in Stage 1 (default: 0.2)
    - temperature: LLM temperature (default: 0.3)
    """
    try:
        # Route to appropriate query method based on use_llm_selection
        if request.use_llm_selection:
            logger.info("Using LLM-based document selection mode")
            result = await rag_pipeline.query_with_llm_selection(
                question=request.prompt,
                max_documents=request.max_documents,
                temperature=request.temperature
            )
        else:
            logger.info("Using hybrid search mode")
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


@app.post("/query-stream")
async def query_stream(request: QueryRequest):
    """
    Stream query response with real-time token generation.

    Returns Server-Sent Events (SSE) with the following message types:

    1. status: Progress updates during retrieval
       {"type": "status", "message": "Finding relevant documents..."}

    2. sources: Document citations found
       {"type": "sources", "citations": [{document_title, section_title, ...}]}

    3. token: Individual tokens from LLM as they're generated
       {"type": "token", "content": "word"}

    4. done: Completion signal with stats
       {"type": "done", "stats": {child_chunks_retrieved, parent_chunks_used, answer_length}}

    5. error: Error message if something fails
       {"type": "error", "message": "Error description"}

    Benefits over /query endpoint:
    - Users see response immediately as it's generated
    - Perceived speed improvement (feels 3x faster)
    - Professional UX like ChatGPT
    - Same quality as regular endpoint (2000 tokens, 5 parent chunks)

    Client Example (JavaScript):
        const eventSource = new EventSource('/query-stream', {
            method: 'POST',
            body: JSON.stringify({prompt: "What is the PTO policy?"})
        });

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'token') {
                appendToAnswer(data.content);
            } else if (data.type === 'sources') {
                displayCitations(data.citations);
            } else if (data.type === 'done') {
                eventSource.close();
            }
        };
    """
    try:
        logger.info("Using streaming query mode")

        # Generate streaming response
        return StreamingResponse(
            rag_pipeline.query_stream(
                question=request.prompt,
                top_k=request.top_k,
                parent_limit=request.parent_limit,
                metadata_filter=request.metadata_filter,
                temperature=request.temperature,
                use_hybrid=request.use_hybrid,
                bm25_weight=request.bm25_weight
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except Exception as e:
        logger.error(f"Error processing streaming query: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/documents")
async def list_documents():
    """
    List all documents in the system with full metadata including answerable questions.

    Returns document information including:
    - Basic info (ID, title, URL, type)
    - Summary and topics
    - Answerable questions (for LLM-based document selection)
    """
    try:
        documents = rag_pipeline.document_store.get_all_documents_with_metadata()
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


@app.put("/documents/{document_id}/questions")
async def update_document_questions(
    document_id: str,
    questions: List[str]
):
    """
    Update the answerable questions for a document without re-ingesting.

    This allows you to manually refine or add questions after document ingestion.
    Useful for:
    - Adding domain-specific questions the LLM might have missed
    - Removing low-quality auto-generated questions
    - Tuning questions based on user query patterns

    Args:
        document_id: The document ID to update
        questions: New list of questions (replaces existing questions)

    Example request body:
    ```json
    [
        "How do I request vacation time?",
        "What is the PTO accrual rate?",
        "Who approves time off requests?"
    ]
    ```
    """
    try:
        if not questions:
            raise HTTPException(
                status_code=400,
                detail="Questions list cannot be empty"
            )

        # Validate questions
        for q in questions:
            if not isinstance(q, str) or not q.strip():
                raise HTTPException(
                    status_code=400,
                    detail="All questions must be non-empty strings"
                )

        result = rag_pipeline.document_store.update_document_questions(
            document_id=document_id,
            answerable_questions=questions
        )

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document questions: {e}")
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


@app.get("/acronyms")
async def get_acronyms():
    """
    Get all acronym expansions.

    Returns the current acronym expansion mappings used for query expansion.
    These help the system understand domain-specific acronyms like PTO, HR, etc.
    """
    try:
        from parent_child_store import get_acronym_expansions
        acronyms = get_acronym_expansions()

        return {
            "acronyms": acronyms,
            "count": len(acronyms),
            "message": "Acronym expansions loaded successfully"
        }
    except Exception as e:
        logger.error(f"Error getting acronyms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/acronyms")
async def update_acronyms(acronyms: Dict[str, str]):
    """
    Update acronym expansions.

    Replaces the entire acronym expansion dictionary. To add/modify a single
    acronym, include all existing acronyms plus the new one.

    Request body should be a JSON object with acronym -> expansion mappings:
    {
        "pto": "Paid Time Off (PTO)",
        "hr": "Human Resources (HR)",
        "new_acronym": "New Expansion (NEW_ACRONYM)"
    }

    The changes are saved to config/acronyms.json and take effect immediately.
    """
    try:
        from parent_child_store import save_acronym_expansions

        # Validate that all values are non-empty strings
        for acronym, expansion in acronyms.items():
            if not isinstance(acronym, str) or not acronym.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid acronym key: '{acronym}' must be a non-empty string"
                )
            if not isinstance(expansion, str) or not expansion.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid expansion for '{acronym}': must be a non-empty string"
                )

        # Convert all acronyms to lowercase for consistency
        normalized_acronyms = {k.lower(): v for k, v in acronyms.items()}

        # Save to file (this also updates the global cache)
        save_acronym_expansions(normalized_acronyms)

        logger.info(f"Updated {len(normalized_acronyms)} acronym expansions")

        return {
            "message": "Acronym expansions updated successfully",
            "count": len(normalized_acronyms),
            "acronyms": normalized_acronyms
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating acronyms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "airgapped_rag_advanced:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
