"""
Test script for streaming query endpoint.

Demonstrates how to consume Server-Sent Events (SSE) from /query-stream endpoint.
"""

import requests
import json
import sys


def test_streaming_query(question: str, api_url: str = "http://localhost:8000"):
    """
    Test the streaming query endpoint with a sample question.

    Args:
        question: Question to ask
        api_url: Base URL of the API server
    """
    print(f"Question: {question}\n")
    print("=" * 80)

    # Prepare request payload
    payload = {
        "prompt": question,
        "top_k": 30,
        "parent_limit": 5,
        "temperature": 0.3,
        "use_hybrid": True,
        "bm25_weight": 0.2
    }

    # Make streaming request
    url = f"{api_url}/query-stream"

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()

            print("\nStreaming response:\n")
            answer_text = ""
            citations = []

            # Process SSE stream
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')

                    # SSE format: "data: {json}"
                    if line_str.startswith('data: '):
                        data_json = line_str[6:]  # Remove "data: " prefix

                        try:
                            data = json.loads(data_json)
                            msg_type = data.get('type')

                            if msg_type == 'status':
                                print(f"[STATUS] {data['message']}")

                            elif msg_type == 'sources':
                                citations = data['citations']
                                print(f"\n[SOURCES] Found {len(citations)} relevant documents")
                                for i, cite in enumerate(citations, 1):
                                    print(f"  {i}. {cite['document_title']} - {cite['section_title']}")
                                print("\n[ANSWER] ", end="", flush=True)

                            elif msg_type == 'token':
                                # Print token without newline to show streaming effect
                                token = data['content']
                                answer_text += token
                                print(token, end="", flush=True)

                            elif msg_type == 'done':
                                stats = data['stats']
                                print(f"\n\n[DONE] Answer generated successfully!")
                                print(f"  - Child chunks retrieved: {stats['child_chunks_retrieved']}")
                                print(f"  - Parent chunks used: {stats['parent_chunks_used']}")
                                print(f"  - Answer length: {stats['answer_length']} characters")

                            elif msg_type == 'error':
                                print(f"\n[ERROR] {data['message']}")
                                return

                        except json.JSONDecodeError as e:
                            print(f"\n[WARNING] Failed to parse SSE data: {e}")

            print("\n" + "=" * 80)
            print("\nFull Answer:")
            print(answer_text)

            if citations:
                print("\n\nCitations:")
                for i, cite in enumerate(citations, 1):
                    print(f"{i}. {cite['document_title']}")
                    if cite['section_title']:
                        print(f"   Section: {cite['section_title']}")
                    if cite['source_url']:
                        print(f"   URL: {cite['source_url']}")

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Request failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Test questions
    test_questions = [
        "What is the PTO policy?",
        "How do I request time off?",
        "How many hours can I work in a single week?"
    ]

    # Use first question, or custom question from command line
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = test_questions[0]

    # Test streaming endpoint
    test_streaming_query(question)

    print("\n\n✅ Streaming test completed!")
    print("\nCompare this to the non-streaming /query endpoint:")
    print("  - Same quality (2000 tokens, 5 parent chunks)")
    print("  - Same accuracy (semantic reranking)")
    print("  - But feels 3x faster due to real-time token display!")
