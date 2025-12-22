"""
Query Classifier and System Query Handler

Classifies incoming queries to determine if they're about:
- Greetings (pleasant messages like "Good morning", "Hello")
- The system itself (meta queries like "What can you do?")
- Document content (normal RAG queries)

For greetings, generates an introduction response.
For system queries, generates helpful responses about capabilities.
Document queries are routed to the RAG pipeline.
"""

import logging
import re
from typing import Dict, Any, List
from ollama_client_lb import OllamaClient

logger = logging.getLogger(__name__)

# System information and capabilities
SYSTEM_INFO = {
    "name": "Adam",
    "full_name": "Amentum Document and Assistance Model",
    "purpose": "AI-powered document search and question answering system",
    "capabilities": [
        "Search and retrieve information from company policy documents",
        "Answer questions about procedures, policies, and guidelines",
        "Provide direct citations with source documents and section numbers",
        "Handle both keyword and semantic searches for better accuracy",
        "Learn from user feedback to improve future responses",
        "Support hybrid search combining keyword matching and AI understanding"
    ],
    "document_types": [
        "Company policies and procedures",
        "Engineering documentation",
        "Safety and compliance guidelines",
        "Process documentation",
        "Contract requirements"
    ],
    "features": [
        "Hybrid search (keyword + semantic AI)",
        "Direct source citations with document links",
        "Context-aware answers with section references",
        "User feedback to continuously improve",
        "Support for complex multi-part questions"
    ],
    "limitations": [
        "Can only answer based on uploaded documents",
        "Cannot access external information or websites",
        "Cannot make decisions or provide legal advice",
        "Requires clear, specific questions for best results"
    ],
    "usage_tips": [
        "Include specific keywords (like policy numbers) for better results",
        "Ask one question at a time for clearest answers",
        "Use feedback buttons to help improve the system",
        "Check citations to verify information in source documents"
    ]
}

# Common greeting patterns for quick detection (case-insensitive)
# These bypass the LLM classification for efficiency
GREETING_PATTERNS = [
    r'^(hi|hello|hey|howdy|hiya|yo)[\s\!\?\.\,]*$',
    r'^good\s*(morning|afternoon|evening|night|day)[\s\!\?\.\,]*$',
    r'^(morning|afternoon|evening)[\s\!\?\.\,]*$',
    r'^(greetings|salutations)[\s\!\?\.\,]*$',
    r'^what\'?s?\s*up[\s\!\?\.\,]*$',
    r'^how\s*(are|r)\s*(you|u)[\s\!\?\.\,]*$',
    r'^how\'?s?\s*it\s*going[\s\!\?\.\,]*$',
    r'^(sup|wassup|wazzup)[\s\!\?\.\,]*$',
    r'^(hola|bonjour|ciao)[\s\!\?\.\,]*$',
    r'^(thanks|thank\s*you|thx|ty)[\s\!\?\.\,]*$',
    r'^(bye|goodbye|see\s*you|later|cya)[\s\!\?\.\,]*$',
    r'^nice\s*to\s*meet\s*you[\s\!\?\.\,]*$',
    r'^pleased\s*to\s*meet\s*you[\s\!\?\.\,]*$',
]

# Compile patterns for efficiency
COMPILED_GREETING_PATTERNS = [re.compile(p, re.IGNORECASE) for p in GREETING_PATTERNS]


def is_greeting(text: str) -> bool:
    """
    Quick check if text is a greeting using regex patterns.

    Args:
        text: User input text

    Returns:
        True if the text matches a known greeting pattern
    """
    text = text.strip()
    for pattern in COMPILED_GREETING_PATTERNS:
        if pattern.match(text):
            return True
    return False


class QueryClassifier:
    """
    Classifies queries and handles system-related questions.

    Uses LLM to intelligently detect when users are asking about
    the system itself vs. asking questions about documents.
    """

    def __init__(
        self,
        ollama_hosts: List[str] = None,
        model_name: str = "mistral"
    ):
        """
        Initialize the query classifier.

        Args:
            ollama_hosts: List of Ollama server URLs for load balancing
            model_name: LLM model to use for classification
        """
        self.ollama_hosts = ollama_hosts or ["http://localhost:11434"]
        self.ollama_client = OllamaClient(hosts=self.ollama_hosts, strategy="round-robin")
        self.model_name = model_name
        logger.info(f"QueryClassifier initialized with model: {model_name}, hosts: {self.ollama_hosts}")

    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify a query as 'greeting', 'system', or 'document' query.

        Greeting queries are pleasant messages like "Good morning", "Hello"
        System queries are about the RAG system itself (e.g., "What can you do?")
        Document queries are about content in documents (e.g., "What is the PTO policy?")

        Args:
            query: User's question

        Returns:
            Dict with 'query_type', 'confidence', and 'original_query'
        """
        # Step 1: Quick pattern-based check for greetings (fast, no LLM needed)
        if is_greeting(query):
            logger.info(f"Query classified as: greeting (pattern match) - '{query[:50]}'")
            return {
                "query_type": "greeting",
                "confidence": "high",
                "original_query": query
            }

        classification_prompt = f"""You are a query classifier for a document search system named "Adam" (Amentum Document Assistant and Manager).

Your job is to determine if the user is asking about:
1. THE SYSTEM ITSELF (system query) - Questions specifically about what Adam/the search system is, what features it has, how to use Adam's interface
2. DOCUMENT CONTENT (document query) - Questions about company policies, procedures, processes, or ANY information that would be found in company documents

USER QUERY: "{query}"

CRITICAL RULES:
- If the question is about company policies, procedures, or processes → DOCUMENT
- If the question is "how do I" do something at the company (request PTO, submit forms, follow procedures) → DOCUMENT
- If the question is about using or understanding company systems/processes → DOCUMENT
- ONLY classify as SYSTEM if specifically asking about Adam's features or capabilities

EXAMPLES OF SYSTEM QUERIES (asking about Adam itself):
- "What is your name?"
- "What can you do?"
- "Introduce yourself"
- "How do I use this search system?"
- "What are you?"
- "Tell me about yourself Adam"
- "What kind of questions can you answer?"
- "How does Adam work?"
- "What features does this system have?"

EXAMPLES OF DOCUMENT QUERIES (asking about company information):
- "What is the PTO policy?"
- "How do I request time off?"
- "How do I request PTO?"
- "How do I submit a timesheet?"
- "What are the safety procedures?"
- "Does Amentum have a dress code?"
- "What is the maximum PTO accrual?"
- "How do I apply for leave?"
- "What is the process for requesting equipment?"
- "How do I report an incident?"

Respond with ONLY ONE WORD:
- "SYSTEM" if asking about Adam/the search system itself
- "DOCUMENT" if asking about company policies, procedures, or processes

Your response:"""

        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=classification_prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "num_predict": 10    # Only need one word
                },
                keep_alive=-1  # Keep model loaded in GPU memory
            )

            classification = response['response'].strip().upper()

            # Normalize response
            if "SYSTEM" in classification:
                query_type = "system"
            elif "DOCUMENT" in classification:
                query_type = "document"
            else:
                # Default to document query if unclear
                logger.warning(f"Unclear classification: {classification}, defaulting to 'document'")
                query_type = "document"

            logger.info(f"Query classified as: {query_type} - '{query[:50]}...'")

            return {
                "query_type": query_type,
                "confidence": "high" if classification in ["SYSTEM", "DOCUMENT"] else "low",
                "original_query": query
            }

        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default to document query on error
            return {
                "query_type": "document",
                "confidence": "low",
                "original_query": query,
                "error": str(e)
            }

    def generate_system_response(self, query: str) -> str:
        """
        Generate a helpful response about the system itself.

        Uses SYSTEM_INFO to create contextual, natural responses
        about the system's capabilities.

        Args:
            query: User's question about the system

        Returns:
            Natural language response about the system
        """
        system_context = f"""SYSTEM INFORMATION:
Name: {SYSTEM_INFO['name']} ({SYSTEM_INFO['full_name']})
Purpose: {SYSTEM_INFO['purpose']}

CAPABILITIES:
{chr(10).join('- ' + cap for cap in SYSTEM_INFO['capabilities'])}

FEATURES:
{chr(10).join('- ' + feat for feat in SYSTEM_INFO['features'])}

DOCUMENT TYPES I CAN SEARCH:
{chr(10).join('- ' + doc for doc in SYSTEM_INFO['document_types'])}

USAGE TIPS:
{chr(10).join('- ' + tip for tip in SYSTEM_INFO['usage_tips'])}

LIMITATIONS:
{chr(10).join('- ' + lim for lim in SYSTEM_INFO['limitations'])}
"""

        response_prompt = f"""You are {SYSTEM_INFO['name']} ({SYSTEM_INFO['full_name']}), a helpful AI assistant for searching company documents.

USER ASKED: "{query}"

Using the system information below, provide a friendly, helpful response that answers their question.

{system_context}

INSTRUCTIONS:
1. Be friendly and conversational
2. Answer their specific question
3. Highlight relevant capabilities
4. Suggest how they can use the system
5. Keep response concise (2-3 paragraphs max)
6. Use "I" to refer to yourself as {SYSTEM_INFO['name']}

YOUR RESPONSE:"""

        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=response_prompt,
                options={
                    "temperature": 0.7,  # Higher temperature for more natural responses
                    "num_predict": 300
                },
                keep_alive=-1  # Keep model loaded in GPU memory
            )

            answer = response['response'].strip()
            logger.info(f"Generated system response for: '{query[:50]}...'")

            return answer

        except Exception as e:
            logger.error(f"Error generating system response: {e}")
            # Fallback response
            return (
                f"I'm {SYSTEM_INFO['name']} ({SYSTEM_INFO['full_name']}), "
                f"your AI-powered assistant for searching company documents. "
                f"I can help you find information about policies, procedures, and guidelines. "
                f"Just ask me a question about any company document, and I'll search for the answer!"
            )

    def generate_greeting_response(self, greeting: str = "") -> str:
        """
        Generate a friendly introduction response for greetings.

        Returns a consistent, helpful introduction that explains what
        the system can do.

        Args:
            greeting: The user's greeting (for context)

        Returns:
            Friendly introduction message
        """
        # Determine appropriate greeting response based on input
        greeting_lower = greeting.lower().strip()

        # Reciprocate the greeting appropriately
        if any(x in greeting_lower for x in ['morning']):
            greeting_reply = "Good morning!"
        elif any(x in greeting_lower for x in ['afternoon']):
            greeting_reply = "Good afternoon!"
        elif any(x in greeting_lower for x in ['evening']):
            greeting_reply = "Good evening!"
        elif any(x in greeting_lower for x in ['night']):
            greeting_reply = "Good evening!"
        elif any(x in greeting_lower for x in ['thanks', 'thank']):
            greeting_reply = "You're welcome!"
        elif any(x in greeting_lower for x in ['bye', 'goodbye', 'see you', 'later', 'cya']):
            greeting_reply = "Goodbye! Feel free to come back anytime you have questions."
            return greeting_reply
        else:
            greeting_reply = "Hello!"

        # Build a helpful introduction
        introduction = (
            f"{greeting_reply} I'm {SYSTEM_INFO['name']} ({SYSTEM_INFO['full_name']}), "
            f"your AI-powered assistant for searching company documents.<br><br>"
            f"<strong>What I can help with:</strong><br>"
            f"• Search and retrieve information from company policy documents<br>"
            f"• Answer questions about procedures, policies, and guidelines<br>"
            f"• Provide direct citations with source documents and section numbers<br><br>"
            f"<strong>How to use me:</strong><br>"
            f"Just ask me a question about any company document! For example:<br>"
            f"• \"What is the PTO policy?\"<br>"
            f"• \"How do I request time off?\"<br>"
            f"• \"What are the safety procedures?\"<br><br>"
            f"What would you like to know?"
        )

        logger.info(f"Generated greeting response for: '{greeting[:50]}'")
        return introduction

    def generate_no_results_response(self, query: str) -> str:
        """
        Generate a helpful response when no relevant documents are found.

        Args:
            query: The user's original query

        Returns:
            Helpful message explaining that no results were found
        """
        response = (
            "I wasn't able to find any relevant information in the available documents "
            "to answer your question.<br><br>"
            "<strong>Suggestions:</strong><br>"
            "• Try rephrasing your question with different keywords<br>"
            "• Use more specific terms (e.g., include policy numbers like EN-PO-XXXX)<br>"
            "• Break down complex questions into simpler parts<br>"
            "• Check the <a href='https://portal.amentumspacemissions.com/MS/Pages/MSDefaultHomePage.aspx' target='_blank'>Management System</a> "
            "where all policy documents are housed<br><br>"
            "If you're looking for a document that hasn't been uploaded yet, "
            "please contact your administrator."
        )

        logger.info(f"Generated no-results response for: '{query[:50]}'")
        return response


# Singleton instance
_classifier_instance = None


def get_query_classifier(**kwargs) -> QueryClassifier:
    """
    Get or create singleton QueryClassifier instance.

    Args:
        **kwargs: Arguments to pass to QueryClassifier constructor

    Returns:
        QueryClassifier instance
    """
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = QueryClassifier(**kwargs)

    return _classifier_instance


if __name__ == "__main__":
    # Test the classifier
    logging.basicConfig(level=logging.INFO)

    classifier = get_query_classifier()

    # Test queries - including greetings, system queries, and document queries
    test_queries = [
        # Greetings (should be classified as 'greeting')
        "Hello",
        "Good morning",
        "Hi!",
        "Hey there",
        "How are you?",
        "Thanks!",
        # System queries (should be classified as 'system')
        "What is your name?",
        "What can you do?",
        "Introduce yourself",
        "Tell me about yourself",
        "What kind of documents can you search?",
        "How does this system work?",
        # Document queries (should be classified as 'document')
        "What is the PTO policy?",
        "How do I request time off?",
        "Does Amentum have a dress code?",
        "What are the safety procedures for confined spaces?"
    ]

    print("\n" + "="*80)
    print("TESTING QUERY CLASSIFIER")
    print("="*80)

    for query in test_queries:
        print(f"\nQUERY: {query}")

        # Classify
        result = classifier.classify_query(query)
        print(f"TYPE: {result['query_type']} (confidence: {result['confidence']})")

        # Generate appropriate response based on type
        if result['query_type'] == 'greeting':
            response = classifier.generate_greeting_response(query)
            print(f"RESPONSE:\n{response}")
        elif result['query_type'] == 'system':
            response = classifier.generate_system_response(query)
            print(f"RESPONSE:\n{response}")
        else:
            print("RESPONSE: [Would search documents for this query]")

        print("-" * 80)
