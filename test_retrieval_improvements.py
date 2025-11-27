#!/usr/bin/env python3
"""
Integration test for RAG retrieval improvements.
Tests: Cross-encoder reranking, Query rewriting, HyDE, Adaptive BM25

Usage:
    python test_retrieval_improvements.py

Requirements:
    - Running Ollama instance(s)
    - Loaded documents in ChromaDB
"""

import asyncio
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_improvements():
    """Test all retrieval improvements."""

    # Import after setting up logging
    from airgapped_rag_advanced import AdvancedRAGPipeline, QueryRequest

    print("=" * 60)
    print("RAG RETRIEVAL IMPROVEMENTS TEST")
    print("=" * 60)

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    try:
        pipeline = AdvancedRAGPipeline()
        print("✅ Pipeline initialized")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return False

    # Check if we have documents
    stats = pipeline.document_store.get_statistics()
    print(f"   Documents in store: {stats.get('documents', 0)}")
    print(f"   Child chunks: {stats.get('total_child_chunks', 0)}")
    print(f"   Parent chunks: {stats.get('total_parent_chunks', 0)}")

    if stats.get('total_child_chunks', 0) == 0:
        print("\n⚠️  WARNING: No documents in store. Tests will use empty results.")
        print("   Upload some documents first for meaningful tests.")

    # Test queries - mix of clean and messy
    test_queries = [
        ("What is the PTO policy?", "clean"),
        ("pto how much days", "messy"),
        ("EN-PO-0301", "keyword"),
        ("who approve my expense", "messy"),
    ]

    print("\n2. Testing queries with different configurations...\n")

    all_passed = True
    for query, query_type in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: '{query}' (type: {query_type})")
        print("=" * 50)

        # Test 1: Default configuration (reranker=True, adaptive BM25)
        print("\n[Test 1: Default config (reranker=True, adaptive BM25)]")
        try:
            result = await pipeline.query(
                question=query,
                top_k=10,
                parent_limit=3,
                use_hybrid=True,
                bm25_weight=None,  # Adaptive
                use_reranker=True,
                rewrite_query=False,
                use_hyde=False
            )
            citations_count = len(result.get('citations', []))
            answer_len = len(result.get('answer', ''))
            print(f"  ✅ Retrieved {citations_count} citations, answer length: {answer_len}")
            if answer_len > 0:
                print(f"  Answer preview: {result['answer'][:100]}...")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_passed = False

        # Test 2: With query rewriting (for messy queries)
        if query_type == "messy":
            print("\n[Test 2: With query rewriting]")
            try:
                result = await pipeline.query(
                    question=query,
                    top_k=10,
                    parent_limit=3,
                    use_hybrid=True,
                    rewrite_query=True,  # Enable
                    use_reranker=True
                )
                citations_count = len(result.get('citations', []))
                print(f"  ✅ Retrieved {citations_count} citations")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                all_passed = False

        # Test 3: With HyDE
        print("\n[Test 3: With HyDE]")
        try:
            result = await pipeline.query(
                question=query,
                top_k=10,
                parent_limit=3,
                use_hybrid=True,
                use_hyde=True,  # Enable
                use_reranker=True
            )
            citations_count = len(result.get('citations', []))
            print(f"  ✅ Retrieved {citations_count} citations")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_passed = False

        # Test 4: Without reranker (verify backward compatibility)
        print("\n[Test 4: Without reranker (backward compatibility)]")
        try:
            result = await pipeline.query(
                question=query,
                top_k=10,
                parent_limit=3,
                use_hybrid=True,
                bm25_weight=0.2,  # Explicit weight
                use_reranker=False,  # Disable
                rewrite_query=False,
                use_hyde=False
            )
            citations_count = len(result.get('citations', []))
            print(f"  ✅ Retrieved {citations_count} citations (no reranker)")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_passed = False

    # Test adaptive BM25 query type detection
    print("\n" + "=" * 60)
    print("3. Testing Adaptive BM25 Query Type Detection")
    print("=" * 60)

    test_queries_bm25 = [
        ("What is the vacation policy?", "semantic"),
        ("EN-PO-0301 section 4.2", "keyword"),
        ("PTO HR", "mixed"),
        ('"employee handbook" training', "keyword"),
    ]

    for query, expected_type in test_queries_bm25:
        query_type, bm25_weight = pipeline.document_store._estimate_query_type(query)
        status = "✅" if query_type == expected_type else "⚠️"
        print(f"{status} Query: '{query[:40]}...' -> type={query_type}, weight={bm25_weight:.2f} (expected: {expected_type})")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all_passed:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed - check logs above")
        return False


def test_cross_encoder_loading():
    """Test that CrossEncoder loads correctly."""
    print("\n" + "=" * 60)
    print("CROSS-ENCODER LOADING TEST")
    print("=" * 60)

    try:
        from sentence_transformers import CrossEncoder
        print("✅ CrossEncoder import successful")

        print("Loading cross-encoder model...")
        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            max_length=512
        )
        print("✅ CrossEncoder model loaded")

        # Test prediction
        pairs = [("What is PTO?", "Employees receive paid time off.")]
        scores = reranker.predict(pairs)
        print(f"✅ Test prediction score: {scores[0]:.4f}")

        return True
    except Exception as e:
        print(f"❌ CrossEncoder test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("RAG RETRIEVAL IMPROVEMENTS - INTEGRATION TESTS")
    print("=" * 60)

    # Test 1: CrossEncoder loading
    ce_passed = test_cross_encoder_loading()

    # Test 2: Full integration tests
    print("\n")
    integration_passed = asyncio.run(test_improvements())

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"CrossEncoder Loading: {'✅ PASSED' if ce_passed else '❌ FAILED'}")
    print(f"Integration Tests: {'✅ PASSED' if integration_passed else '❌ FAILED'}")

    sys.exit(0 if (ce_passed and integration_passed) else 1)
