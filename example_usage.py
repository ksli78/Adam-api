"""
Example usage script for Air-Gapped RAG API

This script demonstrates how to interact with the API programmatically.

Usage:
    python example_usage.py --upload document.pdf --url http://example.com/doc
    python example_usage.py --query "What is the PTO policy?"
    python example_usage.py --list
"""

import argparse
import json
import sys
from pathlib import Path

import requests

# Configuration
API_BASE_URL = "http://localhost:8000"


def check_health():
    """Check if the API is healthy and display info."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()

        print("✓ API is healthy")
        print(f"  Documents indexed: {data['documents_count']}")
        print(f"  Embedding model: {data['embed_model']}")
        print(f"  LLM model: {data['llm_model']}")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API")
        print(f"  Make sure the API is running at {API_BASE_URL}")
        return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def upload_document(pdf_path: str, source_url: str):
    """Upload a PDF document to the API."""
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"✗ File not found: {pdf_path}")
        return False

    if not pdf_file.suffix.lower() == '.pdf':
        print(f"✗ File must be a PDF: {pdf_path}")
        return False

    print(f"Uploading: {pdf_file.name}")
    print(f"Source URL: {source_url}")

    try:
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            data = {'source_url': source_url}

            response = requests.post(
                f"{API_BASE_URL}/upload-document",
                files=files,
                data=data,
                timeout=120  # Allow time for processing
            )
            response.raise_for_status()
            result = response.json()

        print("\n✓ Document uploaded successfully!")
        print(f"  Document ID: {result['document_id']}")
        print(f"  Topic: {result['topic']}")
        print(f"  Source: {result['source_url']}")
        return True

    except requests.exceptions.Timeout:
        print("✗ Upload timed out (document may be too large)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Upload failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"  Details: {e.response.text}")
        return False


def query_documents(question: str, top_k: int = 1):
    """Query the RAG system."""
    print(f"Question: {question}")
    print(f"Retrieving top {top_k} document(s)...")

    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                'prompt': question,
                'top_k': top_k
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(result['answer'])
        print()

        if result['citations']:
            print("=" * 70)
            print("CITATIONS")
            print("=" * 70)
            for i, citation in enumerate(result['citations'], 1):
                print(f"\n[{i}] Source: {citation['source_url']}")
                print(f"    Excerpt: {citation['excerpt'][:200]}...")
        else:
            print("(No citations)")

        return True

    except requests.exceptions.Timeout:
        print("✗ Query timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Query failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"  Details: {e.response.text}")
        return False


def list_documents():
    """List all indexed documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        response.raise_for_status()
        documents = response.json()

        if not documents:
            print("No documents indexed yet.")
            return True

        print(f"\nIndexed Documents ({len(documents)}):")
        print("=" * 100)

        for doc in documents:
            print(f"\nID: {doc['document_id']}")
            print(f"  Topic: {doc['topic']}")
            print(f"  Source: {doc['source_url']}")
            print(f"  File: {doc['filename']}")
            print(f"  Created: {doc['created_at']}")
            print(f"  Size: {doc['char_count']:,} characters")

        return True

    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to list documents: {e}")
        return False


def delete_document(doc_id: str):
    """Delete a document by ID."""
    print(f"Deleting document: {doc_id}")

    try:
        response = requests.delete(
            f"{API_BASE_URL}/documents/{doc_id}",
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        print(f"✓ {result['message']}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"✗ Delete failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"  Details: {e.response.text}")
        return False


def interactive_mode():
    """Run in interactive Q&A mode."""
    print("\n" + "=" * 70)
    print("Interactive Query Mode")
    print("=" * 70)
    print("Type your questions (or 'quit' to exit)")
    print()

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            query_documents(question, top_k=1)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Example usage script for Air-Gapped RAG API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check API health
  python example_usage.py --health

  # Upload a document
  python example_usage.py --upload policy.pdf --url http://company.com/policy

  # Query documents
  python example_usage.py --query "What is the vacation policy?"

  # List all documents
  python example_usage.py --list

  # Delete a document
  python example_usage.py --delete abc123def456

  # Interactive mode
  python example_usage.py --interactive
        """
    )

    parser.add_argument(
        '--health',
        action='store_true',
        help='Check API health status'
    )

    parser.add_argument(
        '--upload',
        type=str,
        metavar='PDF_FILE',
        help='Upload a PDF document'
    )

    parser.add_argument(
        '--url',
        type=str,
        metavar='SOURCE_URL',
        help='Source URL for the uploaded document (required with --upload)'
    )

    parser.add_argument(
        '--query',
        type=str,
        metavar='QUESTION',
        help='Query the RAG system'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=1,
        metavar='K',
        help='Number of documents to retrieve (default: 1)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all indexed documents'
    )

    parser.add_argument(
        '--delete',
        type=str,
        metavar='DOC_ID',
        help='Delete a document by ID'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive query mode'
    )

    parser.add_argument(
        '--api-url',
        type=str,
        default=API_BASE_URL,
        metavar='URL',
        help=f'API base URL (default: {API_BASE_URL})'
    )

    args = parser.parse_args()

    # Update API URL if provided
    global API_BASE_URL
    API_BASE_URL = args.api_url

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Health check (always run if requested, or before other operations)
    if args.health or any([args.upload, args.query, args.list, args.delete, args.interactive]):
        print("Checking API health...")
        if not check_health():
            print("\nPlease start the API first:")
            print("  python airgapped_rag.py")
            print("  # or")
            print("  ./start_airgapped_rag.sh")
            sys.exit(1)
        print()

    # Execute requested operation
    success = True

    if args.upload:
        if not args.url:
            print("✗ --url is required when uploading a document")
            sys.exit(1)
        success = upload_document(args.upload, args.url)

    elif args.query:
        success = query_documents(args.query, args.top_k)

    elif args.list:
        success = list_documents()

    elif args.delete:
        success = delete_document(args.delete)

    elif args.interactive:
        interactive_mode()
        success = True

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
