"""
Query Classifier and System Query Handler

Classifies incoming queries to determine if they're about:
- Greetings (pleasant messages like "Good morning", "Hello")
- The system itself (meta queries like "What can you do?")
- Off-topic requests (stories, jokes, weather, general chat)
- Gibberish/nonsense input
- Document content (normal RAG queries)

For greetings, generates an introduction response.
For system queries, generates helpful responses about capabilities.
For off-topic/gibberish, generates a polite redirect to document queries.
Document queries are routed to the RAG pipeline.
"""

import logging
import re
import string
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

# Off-topic patterns - requests that are clearly not about company documents
OFF_TOPIC_PATTERNS = [
    r'^tell\s*(me)?\s*(a|another)?\s*story',
    r'^tell\s*(me)?\s*(a|another)?\s*joke',
    r'^(what|how).*weather',
    r'^(what|who)\s*(is|are|was|were)\s*(the)?\s*(president|king|queen|prime minister)',
    r'^(write|compose|create)\s*(me)?\s*(a|an)?\s*(poem|song|story|essay|haiku)',
    r'^(sing|dance|play)',
    r'^(what|when|where)\s*(is|are|was|were)\s*(the)?\s*(world cup|super bowl|olympics|election)',
    r'^(calculate|compute|solve|what is)\s*\d+\s*[\+\-\*\/\^]\s*\d+',
    r'^(translate|say)\s*.+\s*(in|to)\s*(spanish|french|german|chinese|japanese)',
    r'^(play|start)\s*(a)?\s*(game|music|video)',
    r'^(can you|do you)\s*(feel|love|hate|dream|sleep|eat)',
    r'^(are you|do you have)\s*(alive|conscious|sentient|feelings|emotions)',
    r'^(what\'?s?|how\'?s?)\s*(the)?\s*(news|stock|bitcoin|crypto)',
    r'^(recommend|suggest)\s*(me)?\s*(a)?\s*(movie|book|restaurant|song)',
    r'^(who|what)\s*(will|is going to)\s*win',
    r'^(predict|forecast)',
]

COMPILED_OFF_TOPIC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]


def is_off_topic(text: str) -> bool:
    """
    Check if text is an off-topic request not related to company documents.

    Args:
        text: User input text

    Returns:
        True if the text matches a known off-topic pattern
    """
    text = text.strip()
    for pattern in COMPILED_OFF_TOPIC_PATTERNS:
        if pattern.search(text):
            return True
    return False


def is_gibberish(text: str) -> bool:
    """
    Detect if text is gibberish/random keyboard mashing.

    Uses multiple heuristics:
    1. Very low ratio of vowels to consonants
    2. Unusual character repetition patterns
    3. Very few real English words
    4. High ratio of unusual character combinations

    Args:
        text: User input text

    Returns:
        True if the text appears to be gibberish
    """
    text = text.strip().lower()

    # Very short inputs are not gibberish (could be abbreviations)
    if len(text) < 5:
        return False

    # Remove punctuation and spaces for analysis
    alpha_only = ''.join(c for c in text if c.isalpha())

    if len(alpha_only) < 3:
        # Mostly non-alphabetic - could be gibberish
        if len(text) > 10:
            return True
        return False

    # Count vowels and consonants
    vowels = set('aeiou')
    vowel_count = sum(1 for c in alpha_only if c in vowels)
    consonant_count = len(alpha_only) - vowel_count

    # Normal English has ~40% vowels, gibberish often has very few or too many
    vowel_ratio = vowel_count / len(alpha_only) if alpha_only else 0

    # Check for repeated character patterns (like "asdfasdf" or "jjjjkkk")
    repeated_chars = sum(1 for i in range(len(alpha_only) - 1) if alpha_only[i] == alpha_only[i + 1])
    repeat_ratio = repeated_chars / len(alpha_only) if alpha_only else 0

    # Check for common letter combinations that indicate real words
    common_bigrams = ['th', 'he', 'in', 'er', 'an', 'on', 'at', 'en', 'nd', 'ti', 'es', 'or', 'te', 'of', 'ed', 'is', 'it', 'al', 'ar', 'st', 'to', 'nt', 'ng', 'se', 're', 'ha', 'as', 'ou', 'io', 'le', 'co', 'me', 'de', 'hi', 'ri', 'ro', 'ic', 'ne', 'ea', 'ra', 'ce', 'li', 'ch', 'wh', 'ho', 'be', 'ca', 'ma', 'no', 'do']
    bigram_hits = sum(1 for bg in common_bigrams if bg in alpha_only)
    expected_bigrams = len(alpha_only) / 5  # Rough estimate of expected hits

    # Gibberish detection rules
    is_likely_gibberish = False

    # Rule 1: Very unusual vowel ratio
    if vowel_ratio < 0.1 or vowel_ratio > 0.7:
        is_likely_gibberish = True

    # Rule 2: Too many repeated characters
    if repeat_ratio > 0.4:
        is_likely_gibberish = True

    # Rule 3: Very few common English bigrams
    if len(alpha_only) > 8 and bigram_hits < expected_bigrams * 0.3:
        is_likely_gibberish = True

    # Override: If we find some common words, it's probably not gibberish
    common_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'what', 'how', 'who', 'why', 'when', 'where', 'which', 'policy', 'document', 'help', 'find', 'search', 'question']
    words_in_text = text.split()
    if any(word in common_words for word in words_in_text):
        is_likely_gibberish = False

    return is_likely_gibberish


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
        Classify a query as 'greeting', 'system', 'off_topic', 'gibberish', or 'document' query.

        Greeting queries are pleasant messages like "Good morning", "Hello"
        System queries are about the RAG system itself (e.g., "What can you do?")
        Off-topic queries are requests unrelated to documents (e.g., "Tell me a story")
        Gibberish queries are nonsense/random text
        Document queries are about content in documents (e.g., "What is the PTO policy?")

        Args:
            query: User's question

        Returns:
            Dict with 'query_type', 'confidence', and 'original_query'
        """
        # Step 1: Check for gibberish first (fast, no LLM needed)
        if is_gibberish(query):
            logger.info(f"Query classified as: gibberish - '{query[:50]}'")
            return {
                "query_type": "gibberish",
                "confidence": "high",
                "original_query": query
            }

        # Step 2: Quick pattern-based check for greetings (fast, no LLM needed)
        if is_greeting(query):
            logger.info(f"Query classified as: greeting (pattern match) - '{query[:50]}'")
            return {
                "query_type": "greeting",
                "confidence": "high",
                "original_query": query
            }

        # Step 3: Check for off-topic requests (fast, no LLM needed)
        if is_off_topic(query):
            logger.info(f"Query classified as: off_topic (pattern match) - '{query[:50]}'")
            return {
                "query_type": "off_topic",
                "confidence": "high",
                "original_query": query
            }

        # Step 4: Use LLM to classify between system, off_topic, and document queries
        classification_prompt = f"""You are a query classifier for a document search system named "Adam" (Amentum Document and Assistance Model).

Your job is to determine if the user is asking about:
1. THE SYSTEM ITSELF (SYSTEM) - Questions specifically about what Adam/the search system is, what features it has, how to use Adam's interface
2. DOCUMENT CONTENT (DOCUMENT) - Questions about company policies, procedures, processes, or ANY information that would be found in company documents
3. OFF-TOPIC REQUESTS (OFFTOPIC) - Requests that have nothing to do with company documents or the system, like stories, jokes, weather, general knowledge, math, games, etc.

USER QUERY: "{query}"

CRITICAL RULES:
- If the question is about company policies, procedures, or processes → DOCUMENT
- If the question is "how do I" do something at the company (request PTO, submit forms, follow procedures) → DOCUMENT
- If the question is about using or understanding company systems/processes → DOCUMENT
- If specifically asking about Adam's features or capabilities → SYSTEM
- If asking for stories, jokes, games, weather, general knowledge, or anything clearly unrelated to company documents → OFFTOPIC
- If the query makes no sense or seems like random text → OFFTOPIC

EXAMPLES OF SYSTEM QUERIES:
- "What is your name?"
- "What can you do?"
- "Introduce yourself"
- "How do I use this search system?"

EXAMPLES OF DOCUMENT QUERIES:
- "What is the PTO policy?"
- "How do I request time off?"
- "How do I request PTO?"
- "How do I submit a timesheet?"
- "What are the safety procedures?"
- "Does Amentum have a dress code?"

EXAMPLES OF OFF-TOPIC QUERIES:
- "Tell me a story"
- "Tell me a joke"
- "What's the weather like?"
- "Who is the president?"
- "Calculate 5 + 3"
- "Write me a poem"
- "asdfghjkl" (gibberish)
- "What's 2+2?"
- "Tell me about dinosaurs"

Respond with ONLY ONE WORD:
- "SYSTEM" if asking about Adam/the search system itself
- "DOCUMENT" if asking about company policies, procedures, or processes
- "OFFTOPIC" if asking for something unrelated to company documents

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
            elif "OFFTOPIC" in classification or "OFF" in classification:
                query_type = "off_topic"
            elif "DOCUMENT" in classification:
                query_type = "document"
            else:
                # Default to off_topic if unclear (safer than searching documents)
                logger.warning(f"Unclear classification: {classification}, defaulting to 'off_topic'")
                query_type = "off_topic"

            logger.info(f"Query classified as: {query_type} - '{query[:50]}...'")

            return {
                "query_type": query_type,
                "confidence": "high" if classification in ["SYSTEM", "DOCUMENT", "OFFTOPIC"] else "low",
                "original_query": query
            }

        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Default to off_topic on error (safer than searching documents)
            return {
                "query_type": "off_topic",
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

    def generate_off_topic_response(self, query: str = "") -> str:
        """
        Generate a polite response for off-topic queries.

        Explains that the system is designed for company document queries
        and provides guidance on what types of questions it can answer.

        Args:
            query: The user's off-topic query

        Returns:
            Polite redirect message
        """
        response = (
            f"I'm {SYSTEM_INFO['name']}, a document search assistant specifically designed "
            f"to help you find information in company documents.<br><br>"
            f"I'm not able to help with that particular request, but I <strong>can</strong> help you with:<br>"
            f"• Finding information in company policies and procedures<br>"
            f"• Answering questions about guidelines and processes<br>"
            f"• Locating specific documents or sections<br><br>"
            f"<strong>Try asking something like:</strong><br>"
            f"• \"What is the PTO policy?\"<br>"
            f"• \"How do I submit a timesheet?\"<br>"
            f"• \"What are the safety procedures?\"<br><br>"
            f"How can I help you find information in our documents?"
        )

        logger.info(f"Generated off-topic response for: '{query[:50]}'")
        return response

    def generate_gibberish_response(self) -> str:
        """
        Generate a helpful response for gibberish/nonsense input.

        Returns:
            Message asking the user to rephrase their question
        """
        response = (
            "I'm sorry, I couldn't understand that input.<br><br>"
            f"I'm {SYSTEM_INFO['name']}, a document search assistant. "
            f"Please ask me a clear question about company documents, policies, or procedures.<br><br>"
            f"<strong>Example questions:</strong><br>"
            f"• \"What is the PTO policy?\"<br>"
            f"• \"How do I request time off?\"<br>"
            f"• \"What are the safety procedures?\"<br><br>"
            f"What would you like to know?"
        )

        logger.info("Generated gibberish response")
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

    # Test queries - including all types
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
        # Off-topic queries (should be classified as 'off_topic')
        "Tell me a story",
        "Tell me another story",
        "Tell me a joke",
        "What's the weather like?",
        "Who is the president?",
        "Write me a poem",
        "What's 2+2?",
        # Gibberish queries (should be classified as 'gibberish')
        "asdfghjkl",
        "qwerty zxcvb nmkl",
        "fjdksla jfkdls fjkdla",
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
        elif result['query_type'] == 'off_topic':
            response = classifier.generate_off_topic_response(query)
            print(f"RESPONSE:\n{response}")
        elif result['query_type'] == 'gibberish':
            response = classifier.generate_gibberish_response()
            print(f"RESPONSE:\n{response}")
        else:
            print("RESPONSE: [Would search documents for this query]")

        print("-" * 80)
