"""
Metadata Extractor Service

Extracts structured metadata from documents using local LLM (Ollama).
Runs once per document, not per chunk.

Extracts:
- Document type (policy, procedure, form, guideline, etc.)
- Primary topics and keywords
- Department/organization
- Key entities mentioned
- Brief summary
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import ollama

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Structured metadata extracted from a document."""

    # Document classification
    document_type: str = "unknown"  # policy, procedure, form, guideline, memo, etc.

    # Content summary
    summary: str = ""  # 2-3 sentence summary
    primary_topics: List[str] = field(default_factory=list)  # 3-5 main topics
    keywords: List[str] = field(default_factory=list)  # 5-10 keywords

    # Organizational info
    departments: List[str] = field(default_factory=list)  # HR, IT, Finance, etc.
    org_units: List[str] = field(default_factory=list)  # Specific teams/groups

    # Key entities
    entities: List[str] = field(default_factory=list)  # PTO, benefits, salary, equipment, etc.

    # Answerable questions (for LLM-based document selection)
    answerable_questions: List[str] = field(default_factory=list)  # 5-10 questions this document can answer

    # Document properties
    language: str = "en"
    confidence: float = 1.0  # Confidence in extraction (0-1)

    # Raw LLM response (for debugging)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetadataExtractor:
    """
    Extracts structured metadata from documents using local Ollama LLM.

    Uses a carefully crafted prompt to extract relevant metadata
    in a structured JSON format.
    """

    def __init__(
        self,
        model_name: str = "mistral-small:22b",
        ollama_host: str = "http://adam.amentumspacemissions.com:11434",
        max_input_chars: int = 6000,
        temperature: float = 0.1,
        context_window: int = 32768
    ):
        """
        Initialize the metadata extractor.

        Args:
            model_name: Ollama model to use
            ollama_host: Ollama server URL
            max_input_chars: Maximum characters to send to LLM
            temperature: LLM temperature (0.0-1.0, lower = more deterministic)
            context_window: LLM context window size in tokens
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.max_input_chars = max_input_chars
        self.temperature = temperature
        self.context_window = context_window

        # Configure ollama client
        self.client = ollama.Client(host=ollama_host)

        logger.info(
            f"MetadataExtractor initialized: model={model_name}, "
            f"host={ollama_host}, max_chars={max_input_chars}, context_window={context_window}"
        )

    def extract(
        self,
        document_text: str,
        document_title: str = "",
        document_filename: str = ""
    ) -> DocumentMetadata:
        """
        Extract metadata from document text.

        Args:
            document_text: Full document text (will be truncated if too long)
            document_title: Optional document title
            document_filename: Optional filename (can help with context)

        Returns:
            DocumentMetadata object with extracted information
        """
        logger.info(f"Extracting metadata for document: {document_title or document_filename}")

        # Truncate text if too long (keep beginning and end for context)
        truncated_text = self._truncate_text(document_text)

        # Build prompt
        prompt = self._build_extraction_prompt(
            text=truncated_text,
            title=document_title,
            filename=document_filename
        )

        # Call LLM
        try:
            response = self._call_llm(prompt)
            logger.debug(f"LLM response: {response[:200]}...")

            # Parse response
            metadata = self._parse_response(response)
            metadata.raw_response = response

            logger.info(
                f"Extracted metadata: type={metadata.document_type}, "
                f"topics={len(metadata.primary_topics)}, keywords={len(metadata.keywords)}"
            )

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return self._create_fallback_metadata(
                document_text, document_title, document_filename
            )

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text intelligently if too long.

        Keeps beginning and end for context.
        """
        if len(text) <= self.max_input_chars:
            return text

        # Keep first 70% and last 30% of allowed length
        first_part_size = int(self.max_input_chars * 0.7)
        last_part_size = int(self.max_input_chars * 0.3)

        first_part = text[:first_part_size]
        last_part = text[-last_part_size:]

        truncated = f"{first_part}\n\n[... middle content truncated ...]\n\n{last_part}"

        logger.debug(f"Truncated text from {len(text)} to {len(truncated)} chars")
        return truncated

    def _build_extraction_prompt(
        self,
        text: str,
        title: str,
        filename: str
    ) -> str:
        """Build prompt for metadata extraction."""

        prompt = f"""You are a document analysis expert. Analyze the following document and extract structured metadata in JSON format.

**Document Title:** {title or "Unknown"}
**Filename:** {filename or "Unknown"}

**Document Text:**
{text}

---

Extract the following information and return ONLY a valid JSON object with these fields:

{{
  "document_type": "policy|procedure|form|guideline|memo|manual|report|other",
  "summary": "A concise 2-3 sentence summary of the document's main purpose and content",
  "primary_topics": ["topic1", "topic2", "topic3"],  // 3-5 main topics
  "keywords": ["keyword1", "keyword2", ...],  // 5-10 important keywords
  "departments": ["HR", "IT", ...],  // Departments mentioned or related
  "org_units": ["team1", "group1", ...],  // Specific organizational units
  "entities": ["PTO", "benefits", ...],  // Key entities, concepts, or terms
  "answerable_questions": [
    "How do I request vacation time?",
    "What are the PTO accrual rates for full-time employees?",
    ...
  ],  // 8-15 natural language questions this document can answer
  "language": "en",  // Document language code
  "confidence": 0.95  // Your confidence in this analysis (0.0-1.0)
}}

**Guidelines:**
- Be specific and accurate
- Extract only information clearly present in the document
- Use lowercase for consistency (except acronyms)
- Focus on substantive topics, not formatting details
- For document_type, choose the most appropriate category
- Confidence should reflect how clear and complete the document is

**CRITICAL: Answerable Questions Guidelines**

Generate 8-15 questions that a user WHO DOESN'T KNOW THIS DOCUMENT EXISTS would naturally ask.

Think about what problems or information needs this document solves, then write questions from that perspective.

**Question Types to Include:**
- ACTION questions: "How do I [do something]?", "What are the steps to [accomplish task]?"
- ELIGIBILITY questions: "Who is eligible for [benefit/program]?", "Can I [do something]?"
- REQUIREMENT questions: "What do I need to [do something]?", "What are the requirements for [process]?"
- DEADLINE questions: "When is the deadline for [task]?", "How far in advance must I [do something]?"
- AMOUNT/RATE questions: "How much [resource] do I get?", "What is the rate for [item]?"
- APPROVAL questions: "Who approves [request]?", "What is the approval process for [task]?"
- CONSEQUENCE questions: "What happens if I [do/don't do something]?"
- LOCATION questions: "Where do I submit [form/request]?", "Where can I find [resource]?"

**AVOID These Bad Question Patterns:**
❌ "What is this document about?" - Too meta, requires knowing the document exists
❌ "What does this procedure cover?" - Too generic
❌ "What policy number is this?" - Meta-information, not user need
❌ "What is the purpose of this guideline?" - Meta-question
❌ Any question containing "this document", "this procedure", "this policy", "this form"

**GOOD Examples for a PTO Policy:**
✓ "How many vacation days do full-time employees get per year?"
✓ "Can I carry over unused PTO to the next year?"
✓ "How do I request time off?"
✓ "Who approves my vacation requests?"
✓ "How far in advance do I need to request vacation?"
✓ "What happens if I don't use all my PTO?"
✓ "Do part-time employees get paid time off?"
✓ "Where do I submit my PTO request?"

**GOOD Examples for a Expense Reimbursement Procedure:**
✓ "How do I submit an expense report?"
✓ "What expenses are eligible for reimbursement?"
✓ "How long does it take to get reimbursed?"
✓ "What receipts do I need to include with my expense report?"
✓ "What is the maximum meal reimbursement amount?"
✓ "Can I get reimbursed for conference travel?"

Be specific and diverse. Cover different aspects of the document.

Return ONLY the JSON object, no other text."""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call Ollama LLM with prompt.

        Args:
            prompt: Prompt text

        Returns:
            LLM response text
        """
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": 800,  # Increased for more questions (8-15 instead of 5-10)
                    "num_ctx": self.context_window  # Context window size
                }
            )

            return response['response'].strip()

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise

    def _parse_response(self, response: str) -> DocumentMetadata:
        """
        Parse LLM response into DocumentMetadata object.

        Args:
            response: JSON response from LLM

        Returns:
            DocumentMetadata object
        """
        try:
            # Try to extract JSON from response (in case LLM added extra text)
            json_match = self._extract_json(response)

            if json_match:
                data = json.loads(json_match)

                # Get and filter answerable questions
                raw_questions = data.get("answerable_questions", [])
                filtered_questions = self._filter_questions(raw_questions)

                return DocumentMetadata(
                    document_type=data.get("document_type", "unknown"),
                    summary=data.get("summary", ""),
                    primary_topics=data.get("primary_topics", []),
                    keywords=data.get("keywords", []),
                    departments=data.get("departments", []),
                    org_units=data.get("org_units", []),
                    entities=data.get("entities", []),
                    answerable_questions=filtered_questions,
                    language=data.get("language", "en"),
                    confidence=float(data.get("confidence", 0.8)),
                    raw_response=response
                )
            else:
                logger.warning("Could not extract JSON from LLM response")
                return self._parse_unstructured_response(response)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return self._parse_unstructured_response(response)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return DocumentMetadata(raw_response=response)

    def _filter_questions(self, questions: List[str]) -> List[str]:
        """
        Filter out low-quality questions using pattern matching.

        Removes:
        - Meta-questions about the document itself
        - Questions that are too generic or vague
        - Questions that assume knowledge of the document

        Args:
            questions: List of raw questions from LLM

        Returns:
            Filtered list of high-quality questions
        """
        import re

        if not questions:
            return []

        filtered = []
        removed_count = 0

        # Bad patterns to filter out
        bad_patterns = [
            r'\bthis\s+(document|procedure|policy|form|guideline|manual|report)\b',
            r'\bwhat\s+is\s+this\b',
            r'\bwhat\s+does\s+this\b',
            r'\bwhat\s+(document|procedure|policy)\s+number\b',
            r'\bpurpose\s+of\s+this\b',
            r'\babout\s+this\s+(document|procedure|policy)\b',
            r'\bwhat\s+(is|are)\s+covered\b',
            r'\bwhat\s+(topics|subjects)\s+(does\s+this|are)\b',
        ]

        # Combine into single regex for efficiency
        bad_pattern = re.compile('|'.join(bad_patterns), re.IGNORECASE)

        for question in questions:
            question = question.strip()

            # Skip empty or too short questions
            if not question or len(question) < 10:
                removed_count += 1
                continue

            # Skip if matches bad patterns
            if bad_pattern.search(question):
                logger.debug(f"Filtered out meta-question: {question}")
                removed_count += 1
                continue

            # Skip questions that are just fragments
            if not question.endswith('?'):
                removed_count += 1
                continue

            # Keep good questions
            filtered.append(question)

        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} low-quality questions, kept {len(filtered)}")

        return filtered

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text (in case LLM added extra text)."""
        import re

        # Try to find JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                # Validate it's valid JSON
                json.loads(match.group(0))
                return match.group(0)
            except json.JSONDecodeError:
                continue

        return None

    def _parse_unstructured_response(self, response: str) -> DocumentMetadata:
        """
        Fallback parser for when LLM doesn't return proper JSON.

        Extracts what we can from unstructured text.
        """
        logger.warning("Using fallback parser for unstructured response")

        metadata = DocumentMetadata(raw_response=response)

        # Try to extract document type
        response_lower = response.lower()
        for doc_type in ["policy", "procedure", "form", "guideline", "memo", "manual", "report"]:
            if doc_type in response_lower:
                metadata.document_type = doc_type
                break

        # Try to extract keywords (lines with keywords/topics)
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if "keyword" in line_lower or "topic" in line_lower:
                # Extract words from this line
                words = [w.strip('",.:;!? ') for w in line.split() if len(w) > 3]
                metadata.keywords.extend(words[:5])

        # Try to extract summary (first substantial sentence)
        sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 20]
        if sentences:
            metadata.summary = sentences[0] + '.'

        metadata.confidence = 0.5  # Low confidence for fallback

        return metadata

    def _create_fallback_metadata(
        self,
        document_text: str,
        document_title: str,
        document_filename: str
    ) -> DocumentMetadata:
        """
        Create basic fallback metadata when LLM extraction fails.

        Uses simple heuristics.
        """
        logger.warning("Creating fallback metadata (LLM extraction failed)")

        metadata = DocumentMetadata()

        # Try to infer document type from filename
        filename_lower = document_filename.lower()
        if "policy" in filename_lower or "po-" in filename_lower:
            metadata.document_type = "policy"
        elif "procedure" in filename_lower or "pr-" in filename_lower:
            metadata.document_type = "procedure"
        elif "form" in filename_lower:
            metadata.document_type = "form"

        # Extract simple keywords (most common meaningful words)
        words = document_text.lower().split()
        word_freq = {}
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those"}

        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            if len(word) > 3 and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        metadata.keywords = [word for word, freq in sorted_words[:10]]

        # Simple summary
        metadata.summary = f"Document: {document_title or document_filename}"

        metadata.confidence = 0.3  # Very low confidence for fallback

        return metadata


# Singleton instance
_extractor_instance = None


def get_metadata_extractor(**kwargs) -> MetadataExtractor:
    """
    Get or create singleton MetadataExtractor instance.

    Args:
        **kwargs: Arguments to pass to MetadataExtractor constructor

    Returns:
        MetadataExtractor instance
    """
    global _extractor_instance

    if _extractor_instance is None:
        _extractor_instance = MetadataExtractor(**kwargs)

    return _extractor_instance


if __name__ == "__main__":
    # Test the extractor
    logging.basicConfig(level=logging.DEBUG)

    test_document = """
    EN-PO-0301: Time Off Policy

    1. Purpose

    This policy establishes guidelines for paid time off (PTO) for all employees
    of the company. It applies to full-time and part-time staff.

    2. Scope

    This policy covers vacation time, sick leave, and personal days. It does not
    cover unpaid leave or leave of absence.

    4.3 PTO Accrual

    Full-time employees accrue 15 days of PTO per year, which begins accruing
    on their start date. Part-time employees accrue PTO on a pro-rated basis.

    Employees may carry over up to 5 unused days to the following year.

    5. Requesting Time Off

    All PTO requests must be submitted through the HR portal at least 2 weeks
    in advance. Requests are subject to manager approval.
    """

    extractor = get_metadata_extractor()
    metadata = extractor.extract(
        document_text=test_document,
        document_title="Time Off Policy",
        document_filename="EN-PO-0301.pdf"
    )

    print("\n=== Extracted Metadata ===")
    print(json.dumps(metadata.to_dict(), indent=2))
