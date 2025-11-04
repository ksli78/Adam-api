# System Query Handling

The RAG system now intelligently handles meta-queries where users ask about the system itself, rather than about document content.

## Overview

When users ask questions like "What is your name?" or "What can you do?", the system:
1. **Classifies** the query as either `system` or `document` type
2. **Routes** system queries to a dedicated handler (no document retrieval)
3. **Generates** natural, context-aware responses about capabilities

This happens automatically using LLM-based classification - no hardcoded rules!

## System Information

The system identifies itself as:
- **Name**: Adam
- **Full Name**: Amentum Document Assistant and Manager
- **Purpose**: AI-powered document search and question answering

## Example System Queries

These queries are automatically detected and answered without searching documents:

### Identity Questions
- "What is your name?"
- "Who are you?"
- "Introduce yourself"
- "Tell me about yourself"

### Capability Questions
- "What can you do?"
- "What are you capable of?"
- "How can you help me?"
- "What kind of questions can you answer?"

### Usage Questions
- "How do I use this?"
- "How does this work?"
- "What features do you have?"
- "What are your limitations?"

## Configuration

System information is configured in `query_classifier.py` via the `SYSTEM_INFO` dictionary:

```python
SYSTEM_INFO = {
    "name": "Adam",
    "full_name": "Amentum Document Assistant and Manager",
    "capabilities": [
        "Search and retrieve information from company policy documents",
        "Answer questions about procedures, policies, and guidelines",
        # ... more capabilities
    ],
    "limitations": [
        "Can only answer based on uploaded documents",
        "Cannot access external information",
        # ... more limitations
    ]
}
```

You can customize:
- System name and description
- Capabilities and features
- Document types handled
- Usage tips
- Known limitations

## How It Works

### 1. Classification

```
User Query → LLM Classifier → "SYSTEM" or "DOCUMENT"
```

The LLM classifier analyzes the query and determines intent:
- **System queries**: About the RAG system itself
- **Document queries**: About content in documents

### 2. Routing

```
if query_type == "system":
    → Generate system response (no retrieval)
else:
    → Normal RAG pipeline (retrieve + answer)
```

### 3. Response Generation

For system queries, the LLM generates responses using `SYSTEM_INFO`:

```python
# Context includes all system capabilities, features, limitations
system_context = build_context_from_SYSTEM_INFO()

# LLM generates natural response
response = generate_response(query, system_context)
```

## API Response Format

System queries return the same response format as document queries, but with empty citations:

```json
{
  "answer": "I'm Adam (Amentum Document Assistant and Manager)...",
  "citations": [],
  "retrieval_stats": {
    "query_type": "system",
    "classification_confidence": "high",
    "child_chunks_retrieved": 0,
    "parent_chunks_used": 0,
    "message": "System query - no document retrieval performed"
  }
}
```

## Benefits

✅ **No Hardcoding**: All responses are LLM-generated, not hardcoded
✅ **Consistent**: Uses same API format as document queries
✅ **Efficient**: No unnecessary document retrieval for meta queries
✅ **Flexible**: Easy to update system info without code changes
✅ **Natural**: Responses feel conversational, not scripted

## Testing

Test the classifier directly:

```bash
python query_classifier.py
```

This runs test queries and shows:
- Classification result (system vs document)
- Generated response for system queries

## Customization

To modify system responses:

1. **Update `SYSTEM_INFO`** in `query_classifier.py`:
   ```python
   SYSTEM_INFO = {
       "name": "YourSystemName",
       "capabilities": ["Your capability 1", ...],
       # ... etc
   }
   ```

2. **Adjust response prompt** in `generate_system_response()`:
   - Modify tone (formal vs casual)
   - Change response length
   - Add/remove information sections

3. **Fine-tune classification** in `classify_query()`:
   - Adjust temperature (currently 0.1 for consistency)
   - Modify example queries in prompt
   - Change confidence thresholds

## Examples

### Query: "What is your name?"

**Response:**
```
I'm Adam, which stands for Amentum Document Assistant and Manager!
I'm your AI-powered assistant designed to help you search through company
documents and find answers to your questions about policies, procedures,
and guidelines.

I can search across all uploaded company documents using a combination of
keyword matching and AI understanding to find the most relevant information.
When I provide answers, I always include citations so you can verify the
information in the original documents.

Just ask me any question about company policies or procedures, and I'll do
my best to help!
```

### Query: "What can you do?"

**Response:**
```
I'm here to help you find information from company documents! Here's what
I can do:

• Search through policy documents, procedures, and guidelines using both
  keyword and semantic search
• Answer questions with direct citations from source documents
• Provide context-aware responses that understand what you're asking about
• Learn from your feedback to improve future responses
• Handle complex queries about multiple topics

I work best when you ask specific questions, and I'll always show you where
the information comes from so you can verify it yourself. If you need help
with a particular policy or procedure, just ask!
```

## Notes

- Classification uses the same LLM model as answer generation
- Low temperature (0.1) ensures consistent classification
- Defaults to "document" query if classification is uncertain
- System responses use higher temperature (0.7) for natural language
