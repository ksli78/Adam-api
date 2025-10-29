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
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Spell correction
from spellchecker import SpellChecker

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
# DATA_DIR structure:
#   - documents/           - Uploaded PDF files
#   - chromadb_advanced/   - Vector database (parent and child collections)
# Set DATA_DIR environment variable to customize location (e.g., D:\data\airgapped_rag on Windows)
default_data_dir = "D:/data/airgapped_rag" if os.name == 'nt' else "/data/airgapped_rag"
DATA_DIR = Path(os.getenv("DATA_DIR", default_data_dir))
DOCS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chromadb_advanced"

# Create directories
DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Using data directory: {DATA_DIR}")
logger.info(f"ChromaDB location: {CHROMA_DIR}")

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")

# FastAPI app
app = FastAPI(
    title="Advanced Air-Gapped RAG API",
    description="Production RAG with semantic chunking, metadata extraction, and parent-child retrieval",
    version="2.0.0"
)

# Include SQL query routes
from sql_routes import sql_router
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

        # Initialize conversation manager for contextual follow-ups
        from conversation_manager import get_conversation_manager
        self.conversation_manager = get_conversation_manager(
            db_path=str(DATA_DIR / "conversations.db")
        )

        # Initialize spell checker with custom domain dictionary
        logger.info("Initializing spell checker with custom dictionary...")
        self.spell_checker = SpellChecker()
        # Add domain-specific terms that should not be "corrected"
        custom_words = {
            'amentum', 'safeup', 'pto', 'cui', 'cto', 'itar', 'eeo', 'ada',
            'fmla', 'osha', 'covid', 'hr', 'dod', 'nasa', 'usaf',
            'timesheet', 'timesheets', 'mgmt', 'pdf', 'mgr', 'dept',
            'reimbursement', 'reimbursements', 'workflow', 'workflows',
            'onboarding', 'offboarding', 'ppe', 'sop', 'sops'
        }
        self.spell_checker.word_frequency.load_words(custom_words)
        logger.info(f"Spell checker initialized with {len(custom_words)} custom terms")

        logger.info("Advanced RAG Pipeline initialized successfully!")
        logger.info(f"Document store stats: {self.document_store.get_statistics()}")

        # Warm up the LLM to keep it hot on GPU
        self._warmup_model()

    def _warmup_model(self):
        """
        Warm up the LLM model to keep it loaded in GPU memory.

        This prevents the first query from being slow due to model loading.
        Runs a simple generation to load the model into VRAM.
        """
        logger.info("Warming up LLM model on GPU...")
        try:
            # Run a quick, simple generation to load model
            self.ollama_client.generate(
                model=LLM_MODEL,
                prompt="Hello",
                options={
                    "num_predict": 5,  # Just a few tokens
                    "temperature": 0.1
                },
                keep_alive=-1  # Keep model loaded indefinitely
            )
            logger.info("LLM model warmed up and ready on GPU")
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")

    def _correct_spelling(self, text: str) -> tuple[str, List[tuple[str, str]]]:
        """
        Correct spelling errors in the query text.

        Fixes typos and misspellings while preserving domain-specific terms
        like "SafeUp", "Amentum", company acronyms, etc.

        Args:
            text: Query text that may contain typos

        Returns:
            Tuple of (corrected_text, list of (original, correction) pairs)
        """
        import string

        # Split into words while preserving punctuation positions
        words = text.split()
        corrections = []
        corrected_words = []

        for word in words:
            # Strip punctuation for spell checking but remember it
            stripped = word.strip(string.punctuation)
            leading_punct = word[:len(word) - len(word.lstrip(string.punctuation))]
            trailing_punct = word[len(word.rstrip(string.punctuation)):]

            # Skip very short words (likely ok) and words with numbers
            if len(stripped) <= 2 or any(c.isdigit() for c in stripped):
                corrected_words.append(word)
                continue

            # Check if word is misspelled (case-insensitive check)
            if stripped.lower() not in self.spell_checker:
                # Get correction
                correction = self.spell_checker.correction(stripped.lower())

                # Only apply correction if we found one and it's different
                if correction and correction != stripped.lower():
                    # Preserve original capitalization pattern
                    if stripped.isupper():
                        corrected = correction.upper()
                    elif stripped[0].isupper():
                        corrected = correction.capitalize()
                    else:
                        corrected = correction

                    corrections.append((stripped, corrected))
                    corrected_words.append(leading_punct + corrected + trailing_punct)
                else:
                    # No good correction found, keep original
                    corrected_words.append(word)
            else:
                # Word is spelled correctly
                corrected_words.append(word)

        corrected_text = ' '.join(corrected_words)

        # Log corrections if any were made
        if corrections:
            corrections_str = ', '.join([f"'{orig}' → '{corr}'" for orig, corr in corrections])
            logger.info(f"Spell corrections applied: {corrections_str}")

        return corrected_text, corrections

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
        conversation_id: Optional[str] = None,
        top_k: int = 10,
        parent_limit: int = 3,
        metadata_filter: Dict[str, Any] = None,
        temperature: float = 0.3,
        use_hybrid: bool = True,
        bm25_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Query the RAG system with parent-child retrieval and conversation context.

        Process:
        1. Check conversation history for context
        2. Classify query (system vs document)
        3. Retrieve top_k child chunks (hybrid: BM25 + semantic)
        4. Expand to parent chunks (context)
        5. Pass parent chunks + conversation history to LLM
        6. Generate answer with citations
        7. Store message in conversation history

        Args:
            question: User's question
            conversation_id: Optional conversation ID for context
            top_k: Number of child chunks to retrieve
            parent_limit: Maximum number of parent chunks for LLM
            metadata_filter: Optional metadata filter
            temperature: LLM temperature
            use_hybrid: Use hybrid search (BM25 + semantic)
            bm25_weight: Weight for BM25 scores (0.0-1.0)

        Returns:
            Dict with answer, citations, and conversation info
        """
        logger.info(f"Processing query: {question} (conversation_id={conversation_id}, hybrid={use_hybrid})")

        try:
            # Step 0: Handle conversation ID
            if not conversation_id:
                conversation_id = self.conversation_manager.create_conversation()
                logger.info(f"Created new conversation: {conversation_id}")

            # Check if conversation limit reached
            conv_limit_reached = self.conversation_manager.should_start_new_conversation(
                conversation_id,
                max_messages=50
            )

            if conv_limit_reached:
                logger.warning(f"Conversation {conversation_id} reached message limit")

            # Get conversation context
            conversation_context = self.conversation_manager.get_conversation_context(
                conversation_id,
                max_messages=10
            )
            previous_chunks = self.conversation_manager.get_previous_chunks(
                conversation_id,
                max_messages=3
            )

            # Add user message to conversation
            self.conversation_manager.add_message(
                conversation_id,
                role="user",
                content=question
            )

            # Step 0.5: Apply spell correction to fix typos and misspellings
            # This happens before expansion and classification
            # Example: "saftey policy>" → "safety policy>"
            corrected_question, spelling_corrections = self._correct_spelling(question)

            # Use corrected question for all downstream processing
            if spelling_corrections:
                logger.info(f"Original query: '{question}'")
                logger.info(f"Corrected query: '{corrected_question}'")
                question = corrected_question

            # Step 1: Expand query with conversation context FIRST (before classification)
            # This is critical: "How is it calculated?" → "How is PTO calculated?"
            # Must happen before classification so expanded query is classified correctly
            expanded_query = question
            if conversation_context:
                expanded_query = await self._expand_query_with_context(
                    question,
                    conversation_context
                )
                logger.info(f"Expanded query: '{question}' -> '{expanded_query}'")

            # Step 2: Classify the EXPANDED query (system vs document query)
            # Use expanded query so "How is PTO calculated?" is correctly classified as document
            # Without expansion, "How is it calculated?" might be misclassified as system query
            query_to_classify = expanded_query if conversation_context else question
            classification = self.query_classifier.classify_query(query_to_classify)

            # Step 3: Handle system queries (about the RAG system itself)
            if classification['query_type'] == 'system':
                logger.info(f"Detected system query, generating system response")
                system_answer = self.query_classifier.generate_system_response(question)

                # Replace newlines with HTML line breaks for display
                system_answer = system_answer.replace('\n', '<br>')

                # Store assistant message in conversation
                self.conversation_manager.add_message(
                    conversation_id,
                    role="assistant",
                    content=system_answer,
                    query_type="system"
                )

                return {
                    "answer": system_answer,
                    "citations": [],
                    "conversation_id": conversation_id,
                    "conversation_warning": "Conversation limit reached - please start a new conversation" if conv_limit_reached else None,
                    "retrieval_stats": {
                        "query_type": "system",
                        "classification_confidence": classification['confidence'],
                        "child_chunks_retrieved": 0,
                        "parent_chunks_used": 0,
                        "message": "System query - no document retrieval performed",
                        "spelling_corrections": [{"from": orig, "to": corr} for orig, corr in spelling_corrections] if spelling_corrections else None
                    }
                }

            # Step 4: Handle document queries (normal RAG retrieval)
            logger.info("Detected document query, proceeding with retrieval")

            # Retrieve with parent expansion (using expanded query for better context)
            child_results, parent_results = self.document_store.retrieve_with_parent_expansion(
                query=expanded_query,  # Use expanded query for better retrieval
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

                insufficient_answer = (
                    "I couldn't find relevant information about your question in the available documents.<br><br>"
                    "Here are some suggestions:<br>"
                    "• Try rephrasing your question with different keywords<br>"
                    "• Check the <a href='https://portal.amentumspacemissions.com/MS/Pages/MSDefaultHomePage.aspx' target='_blank'>Management System</a> "
                    "where all policy documents are housed<br>"
                    "• If you're looking for a specific policy, try including the policy number (e.g., EN-PO-XXXX)"
                )

                # Store assistant message in conversation
                self.conversation_manager.add_message(
                    conversation_id,
                    role="assistant",
                    content=insufficient_answer,
                    query_type="document"
                )

                return {
                    "answer": insufficient_answer,
                    "citations": [],
                    "confidence": 0.0,
                    "conversation_id": conversation_id,
                    "conversation_warning": "Conversation limit reached - please start a new conversation" if conv_limit_reached else None,
                    "retrieval_stats": {
                        "query_type": "document",
                        "classification_confidence": classification.get('confidence', 'high'),
                        "child_chunks_retrieved": len(child_results),
                        "parent_chunks_used": 0,
                        "message": "Insufficient relevant documents found",
                        "spelling_corrections": [{"from": orig, "to": corr} for orig, corr in spelling_corrections] if spelling_corrections else None
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

            # Generate answer with LLM (include conversation context)
            answer = await self._generate_answer(
                question,
                context,
                temperature,
                conversation_context=conversation_context if conversation_context else None
            )

            # Replace newlines with HTML line breaks for better display
            answer = answer.replace('\n', '<br>')

            # Extract URLs that were actually cited in the answer
            # Inline citation format: <span><a href="URL">FileName.pdf</a></span>
            cited_urls = set(re.findall(r'<a\s+href=["\'](https?://[^"\']+)["\']', answer))
            logger.info(f"Found {len(cited_urls)} URLs cited in answer: {cited_urls}")

            # Build citations - only include documents that were actually cited
            citations = []
            parent_chunk_ids = []
            for parent in parent_results:
                source_url = parent['metadata'].get('source_url', '')

                # Only include if this document was actually cited in the answer
                if source_url in cited_urls:
                    citations.append({
                        "source_url": source_url,
                        "document_title": parent['metadata'].get('document_title', 'Unknown'),
                        "section_title": parent['metadata'].get('section_title', ''),
                        "section_number": parent['metadata'].get('section_number', ''),
                        "excerpt": parent['text'][:500] + "..." if len(parent['text']) > 500 else parent['text']
                    })
                    parent_chunk_ids.append(parent['id'])

            # Store assistant message in conversation with chunk IDs
            child_chunk_ids = [c['id'] for c in child_results]
            self.conversation_manager.add_message(
                conversation_id,
                role="assistant",
                content=answer,
                query_type="document",
                chunk_ids=child_chunk_ids,
                parent_chunk_ids=parent_chunk_ids
            )

            result = {
                "answer": answer,
                "citations": citations,
                "conversation_id": conversation_id,
                "conversation_warning": "Conversation limit reached - please start a new conversation" if conv_limit_reached else None,
                "retrieval_stats": {
                    "query_type": "document",
                    "classification_confidence": classification.get('confidence', 'high'),
                    "child_chunks_retrieved": len(child_results),
                    "parent_chunks_used": len(parent_results),
                    "parent_chunk_ids": parent_chunk_ids,  # For feedback tracking
                    "metadata_filter": metadata_filter,
                    "use_hybrid": use_hybrid,
                    "bm25_weight": bm25_weight,
                    "spelling_corrections": [{"from": orig, "to": corr} for orig, corr in spelling_corrections] if spelling_corrections else None
                }
            }

            logger.info("Query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise

    async def _expand_query_with_context(
        self,
        question: str,
        conversation_context: str
    ) -> str:
        """
        Expand a query using conversation context to make it standalone.

        Converts follow-up questions like "How do I request it?" into
        "How do I request PTO?" by using the conversation history.
        Also adds context for topic references like "Explain SafeUp" when
        SafeUp was mentioned in previous answers.

        Args:
            question: User's current question (may have pronouns/references)
            conversation_context: Previous conversation messages

        Returns:
            Expanded, standalone query suitable for retrieval
        """
        expansion_prompt = f"""{conversation_context}

CURRENT QUESTION: {question}

TASK: Rewrite the current question to be a standalone search query for document retrieval.

CRITICAL RULES:
1. Replace pronouns (it, that, this, them, these) with their referents from conversation
2. If the question mentions a topic/term from the conversation, add minimal context about what domain it's from
3. Do NOT add document names (like "EN-PO-0301.pdf") or URLs
4. Keep the SAME basic sentence structure as the original
5. Keep the query SHORT and SIMPLE - just enough context to search
6. If already standalone, return it UNCHANGED
7. Output ONLY the rewritten question - no explanations

EXAMPLES:
Conversation: "User: What is the PTO policy?"
Current: "How do I request it?"
Rewritten: "How do I request PTO?"

Conversation: "User: What is the PTO policy?"
Current: "How is it calculated?"
Rewritten: "How is PTO calculated?"

Conversation: "User: Does Amentum have a dress code?"
Current: "What about shoes?"
Rewritten: "What about shoes in the dress code?"

Conversation: "User: What are the safety procedures?"
Current: "Tell me more"
Rewritten: "Tell me more about safety procedures"

Conversation: "User: What is the safety policy? Assistant: The safety policy includes SafeUp®, Amentum's safety program..."
Current: "Explain SafeUp"
Rewritten: "Explain SafeUp safety program"

Conversation: "User: What benefits does Amentum offer? Assistant: Amentum offers 401k, health insurance, and tuition assistance..."
Current: "Tell me about tuition assistance"
Rewritten: "Tell me about tuition assistance benefits"

Conversation: "User: What is the timesheet policy? Assistant: EN-PO-0501 describes timesheet submission..."
Current: "Who approves them?"
Rewritten: "Who approves timesheets?"

REWRITTEN QUESTION:"""

        try:
            response = self.ollama_client.generate(
                model=LLM_MODEL,
                prompt=expansion_prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent expansion
                    "num_predict": 50
                },
                keep_alive=-1
            )

            expanded = response['response'].strip()

            # Remove quotes if LLM added them
            expanded = expanded.strip('"').strip("'")

            # If expansion failed or is empty, return original
            if not expanded or len(expanded) < 3:
                logger.warning(f"Query expansion failed, using original: {question}")
                return question

            return expanded

        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return question  # Fallback to original query

    async def _generate_answer(
        self,
        question: str,
        context: str,
        temperature: float,
        conversation_context: Optional[str] = None
    ) -> str:
        """Generate answer using Ollama LLM with optional conversation context."""

        # Build prompt with conversation context if available
        prompt_parts = ["You are a friendly, helpful assistant that answers questions using information from company documents."]

        if conversation_context:
            prompt_parts.append(f"\n{conversation_context}\n")
            prompt_parts.append("\nNOTE: Consider the conversation history above when answering. This may be a follow-up question.\n")

        prompt_parts.append(f"""
CURRENT QUESTION: {question}

AVAILABLE DOCUMENTS:
{context}

INSTRUCTIONS:
1. Answer the question naturally and conversationally - no need for phrases like "Based on the provided documents" or "According to Document X"
2. Use ONLY information from the documents above
3. If this is a follow-up question, consider the conversation history but STILL answer based on the documents
4. When referencing a document, use inline HTML citations in this exact format: <span><a href="URL">FileName.pdf</a></span>
5. ONLY cite documents you actually use in your answer - don't mention documents you didn't reference
6. Include specific details like section numbers, amounts, dates when relevant
7. If the documents don't contain enough information, say so clearly
8. Keep your answer focused and helpful

EXAMPLE of inline citation:
"PTO is a paid time off program<span><a href="https://example.com/EN-PO-0301.pdf">EN-PO-0301.pdf</a></span> that varies based on years of service."

ANSWER:""")

        prompt = "".join(prompt_parts)

        logger.debug("Calling Ollama to generate answer...")

        try:
            response = self.ollama_client.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": 500
                },
                keep_alive=-1  # Keep model loaded in GPU memory
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
    conversation_id: Optional[str] = None  # For conversation context
    top_k: int = 10
    parent_limit: int = 3
    temperature: float = 0.1  # Lower temperature for more consistent, deterministic responses
    metadata_filter: Optional[Dict[str, Any]] = None
    use_hybrid: bool = True  # Use hybrid search (BM25 + semantic) by default
    bm25_weight: float = 0.5  # Weight for BM25 vs semantic (0.5 = equal weight)


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    conversation_id: Optional[str] = None
    conversation_warning: Optional[str] = None
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
    Query the RAG system with hybrid search (BM25 + semantic) and conversation context.

    Retrieves child chunks using hybrid search, expands to parents, generates answer.
    Supports conversation history for follow-up questions.

    Hybrid search combines:
    - BM25: Keyword/lexical matching (good for exact terms)
    - Semantic: Embedding similarity (good for concepts)

    Set use_hybrid=False for pure semantic search.
    Adjust bm25_weight (0.0-1.0) to control BM25 vs semantic influence.

    Pass conversation_id to maintain context across multiple questions.
    If not provided, a new conversation will be created.
    """
    try:
        result = await rag_pipeline.query(
            question=request.prompt,
            conversation_id=request.conversation_id,
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


@app.post("/conversation/new")
async def create_conversation():
    """
    Create a new conversation.

    Returns:
        conversation_id for the new conversation
    """
    try:
        conversation_id = rag_pipeline.conversation_manager.create_conversation()
        return {
            "conversation_id": conversation_id,
            "message": "New conversation created"
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str, limit: Optional[int] = None):
    """
    Get conversation history.

    Args:
        conversation_id: Conversation ID
        limit: Optional limit on number of messages

    Returns:
        List of messages in the conversation
    """
    try:
        history = rag_pipeline.conversation_manager.get_conversation_history(
            conversation_id,
            limit=limit
        )
        stats = rag_pipeline.conversation_manager.get_conversation_stats(conversation_id)

        return {
            "conversation_id": conversation_id,
            "messages": history,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{conversation_id}/stats")
async def get_conversation_stats(conversation_id: str):
    """
    Get conversation statistics.

    Args:
        conversation_id: Conversation ID

    Returns:
        Conversation stats including message count
    """
    try:
        stats = rag_pipeline.conversation_manager.get_conversation_stats(conversation_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting conversation stats: {e}")
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
