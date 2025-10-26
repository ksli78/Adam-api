"""
Feedback Storage and Analysis System

Captures user feedback on RAG responses and uses it to improve future retrievals.

Features:
- Store feedback (good/bad) with full context (query, answer, chunks used)
- Track document/chunk quality scores based on feedback
- Provide analytics on retrieval quality
- Enable feedback-weighted retrieval to boost good chunks
"""

import sqlite3
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class FeedbackStore:
    """
    Manages user feedback storage and retrieval quality analytics.

    Database Schema:
    - feedback: Stores each feedback event
    - chunk_quality: Aggregated quality scores for each chunk
    """

    def __init__(self, db_path: str = "/data/airgapped_rag/feedback.db"):
        """
        Initialize the feedback store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing FeedbackStore at {db_path}")
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Feedback events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                feedback_type TEXT NOT NULL CHECK(feedback_type IN ('good', 'bad')),
                retrieval_method TEXT,
                chunks_used TEXT,  -- JSON array of chunk IDs
                citations TEXT,     -- JSON array of citations
                retrieval_stats TEXT,  -- JSON with retrieval statistics
                user_comment TEXT,
                session_id TEXT
            )
        """)

        # Chunk quality scores (aggregated from feedback)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_quality (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                good_count INTEGER DEFAULT 0,
                bad_count INTEGER DEFAULT 0,
                total_appearances INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 0.0,
                last_updated TEXT
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
            ON feedback(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_type
            ON feedback(feedback_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_quality_score
            ON chunk_quality(quality_score DESC)
        """)

        conn.commit()
        conn.close()

        logger.info("Feedback database initialized successfully")

    def add_feedback(
        self,
        query: str,
        answer: str,
        feedback_type: str,
        chunks_used: List[str],
        citations: List[Dict[str, Any]] = None,
        retrieval_stats: Dict[str, Any] = None,
        retrieval_method: str = "hybrid",
        user_comment: str = None,
        session_id: str = None
    ) -> int:
        """
        Record user feedback for a query-answer pair.

        Args:
            query: The user's question
            answer: The RAG system's answer
            feedback_type: "good" or "bad"
            chunks_used: List of chunk IDs used in retrieval
            citations: List of citation objects
            retrieval_stats: Retrieval statistics (child chunks, parent chunks, etc.)
            retrieval_method: "hybrid", "semantic", or "bm25"
            user_comment: Optional user comment
            session_id: Optional session identifier

        Returns:
            feedback_id of the inserted record
        """
        if feedback_type not in ('good', 'bad'):
            raise ValueError("feedback_type must be 'good' or 'bad'")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO feedback (
                timestamp, query, answer, feedback_type, retrieval_method,
                chunks_used, citations, retrieval_stats, user_comment, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            query,
            answer,
            feedback_type,
            retrieval_method,
            json.dumps(chunks_used),
            json.dumps(citations) if citations else None,
            json.dumps(retrieval_stats) if retrieval_stats else None,
            user_comment,
            session_id
        ))

        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Update chunk quality scores
        self._update_chunk_quality(chunks_used, feedback_type)

        logger.info(
            f"Recorded {feedback_type} feedback (ID: {feedback_id}) "
            f"for query: '{query[:50]}...'"
        )

        return feedback_id

    def _update_chunk_quality(self, chunk_ids: List[str], feedback_type: str):
        """
        Update quality scores for chunks based on feedback.

        Quality score = (good_count - bad_count) / total_appearances
        Range: -1.0 (all bad) to +1.0 (all good)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        for chunk_id in chunk_ids:
            # Get current stats or create new entry
            cursor.execute("""
                SELECT good_count, bad_count, total_appearances
                FROM chunk_quality
                WHERE chunk_id = ?
            """, (chunk_id,))

            row = cursor.fetchone()

            if row:
                good_count, bad_count, total_appearances = row
            else:
                good_count = bad_count = total_appearances = 0

            # Update counts
            if feedback_type == 'good':
                good_count += 1
            else:
                bad_count += 1
            total_appearances += 1

            # Calculate quality score
            if total_appearances > 0:
                quality_score = (good_count - bad_count) / total_appearances
            else:
                quality_score = 0.0

            # Upsert chunk quality
            cursor.execute("""
                INSERT OR REPLACE INTO chunk_quality (
                    chunk_id, good_count, bad_count, total_appearances,
                    quality_score, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk_id,
                good_count,
                bad_count,
                total_appearances,
                quality_score,
                timestamp
            ))

        conn.commit()
        conn.close()

        logger.debug(f"Updated quality scores for {len(chunk_ids)} chunks")

    def get_chunk_quality_score(self, chunk_id: str) -> float:
        """
        Get the quality score for a specific chunk.

        Returns:
            Quality score (-1.0 to +1.0), or 0.0 if no feedback exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT quality_score
            FROM chunk_quality
            WHERE chunk_id = ?
        """, (chunk_id,))

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else 0.0

    def get_chunk_quality_scores(self, chunk_ids: List[str]) -> Dict[str, float]:
        """
        Get quality scores for multiple chunks.

        Args:
            chunk_ids: List of chunk IDs

        Returns:
            Dict mapping chunk_id to quality_score
        """
        if not chunk_ids:
            return {}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query with proper number of placeholders
        placeholders = ','.join('?' * len(chunk_ids))
        cursor.execute(f"""
            SELECT chunk_id, quality_score
            FROM chunk_quality
            WHERE chunk_id IN ({placeholders})
        """, chunk_ids)

        scores = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        # Fill in 0.0 for chunks with no feedback
        for chunk_id in chunk_ids:
            if chunk_id not in scores:
                scores[chunk_id] = 0.0

        return scores

    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary statistics of feedback over the last N days.

        Args:
            days: Number of days to look back

        Returns:
            Dict with feedback statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
        cutoff_str = cutoff_date.isoformat()

        # Overall feedback counts
        cursor.execute("""
            SELECT feedback_type, COUNT(*)
            FROM feedback
            WHERE timestamp >= ?
            GROUP BY feedback_type
        """, (cutoff_str,))

        feedback_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Feedback by retrieval method
        cursor.execute("""
            SELECT retrieval_method, feedback_type, COUNT(*)
            FROM feedback
            WHERE timestamp >= ?
            GROUP BY retrieval_method, feedback_type
        """, (cutoff_str,))

        by_method = {}
        for method, feedback_type, count in cursor.fetchall():
            if method not in by_method:
                by_method[method] = {}
            by_method[method][feedback_type] = count

        # Top worst performing chunks
        cursor.execute("""
            SELECT chunk_id, good_count, bad_count, quality_score
            FROM chunk_quality
            WHERE total_appearances >= 2
            ORDER BY quality_score ASC
            LIMIT 10
        """)

        worst_chunks = [
            {
                "chunk_id": row[0],
                "good_count": row[1],
                "bad_count": row[2],
                "quality_score": row[3]
            }
            for row in cursor.fetchall()
        ]

        # Top best performing chunks
        cursor.execute("""
            SELECT chunk_id, good_count, bad_count, quality_score
            FROM chunk_quality
            WHERE total_appearances >= 2
            ORDER BY quality_score DESC
            LIMIT 10
        """)

        best_chunks = [
            {
                "chunk_id": row[0],
                "good_count": row[1],
                "bad_count": row[2],
                "quality_score": row[3]
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        good_count = feedback_counts.get('good', 0)
        bad_count = feedback_counts.get('bad', 0)
        total = good_count + bad_count

        return {
            "period_days": days,
            "total_feedback": total,
            "good_feedback": good_count,
            "bad_feedback": bad_count,
            "satisfaction_rate": (good_count / total * 100) if total > 0 else 0,
            "by_retrieval_method": by_method,
            "worst_performing_chunks": worst_chunks,
            "best_performing_chunks": best_chunks
        }

    def get_recent_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent feedback entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of feedback entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT feedback_id, timestamp, query, answer, feedback_type,
                   retrieval_method, user_comment
            FROM feedback
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        results = [
            {
                "feedback_id": row[0],
                "timestamp": row[1],
                "query": row[2],
                "answer": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                "feedback_type": row[4],
                "retrieval_method": row[5],
                "user_comment": row[6]
            }
            for row in cursor.fetchall()
        ]

        conn.close()
        return results


# Singleton instance
_feedback_store_instance = None


def get_feedback_store(**kwargs) -> FeedbackStore:
    """
    Get or create singleton FeedbackStore instance.

    Args:
        **kwargs: Arguments to pass to FeedbackStore constructor

    Returns:
        FeedbackStore instance
    """
    global _feedback_store_instance

    if _feedback_store_instance is None:
        _feedback_store_instance = FeedbackStore(**kwargs)

    return _feedback_store_instance
