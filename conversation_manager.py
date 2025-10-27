"""
Conversation History Manager

Manages conversation history for contextual follow-up questions.
Stores conversations in SQLite and provides context for RAG queries.
"""

import logging
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and context.

    Features:
    - Store conversation messages with full context
    - Retrieve conversation history
    - Track retrieved chunks per message for context
    - Implement conversation length limits
    - Support starting new conversations
    """

    def __init__(self, db_path: str = "/data/airgapped_rag/conversations.db"):
        """
        Initialize the conversation manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"ConversationManager initialized with db: {db_path}")

    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query_type TEXT,
                    chunk_ids TEXT,
                    parent_chunk_ids TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)

            # Indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, timestamp)
            """)

            conn.commit()
            logger.info("Database tables initialized")

    def create_conversation(self) -> str:
        """
        Create a new conversation.

        Returns:
            New conversation ID
        """
        conversation_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (conversation_id)
                VALUES (?)
            """, (conversation_id,))
            conn.commit()

        logger.info(f"Created new conversation: {conversation_id}")
        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        query_type: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        parent_chunk_ids: Optional[List[str]] = None
    ) -> str:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Conversation ID
            role: Message role ('user' or 'assistant')
            content: Message content
            query_type: Type of query ('system' or 'document')
            chunk_ids: Child chunk IDs used for retrieval
            parent_chunk_ids: Parent chunk IDs used for context

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Add message
            cursor.execute("""
                INSERT INTO messages (
                    message_id, conversation_id, role, content,
                    query_type, chunk_ids, parent_chunk_ids
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id,
                conversation_id,
                role,
                content,
                query_type,
                json.dumps(chunk_ids) if chunk_ids else None,
                json.dumps(parent_chunk_ids) if parent_chunk_ids else None
            ))

            # Update conversation metadata
            cursor.execute("""
                UPDATE conversations
                SET last_message_at = CURRENT_TIMESTAMP,
                    message_count = message_count + 1
                WHERE conversation_id = ?
            """, (conversation_id,))

            conn.commit()

        logger.debug(f"Added {role} message to conversation {conversation_id}")
        return message_id

    def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            conversation_id: Conversation ID
            limit: Maximum number of recent messages to return

        Returns:
            List of messages with metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT
                    message_id, role, content, timestamp,
                    query_type, chunk_ids, parent_chunk_ids
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query, (conversation_id,))
            rows = cursor.fetchall()

        messages = []
        for row in rows:
            message = {
                "message_id": row["message_id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "query_type": row["query_type"]
            }

            if row["chunk_ids"]:
                message["chunk_ids"] = json.loads(row["chunk_ids"])
            if row["parent_chunk_ids"]:
                message["parent_chunk_ids"] = json.loads(row["parent_chunk_ids"])

            messages.append(message)

        return messages

    def get_conversation_context(
        self,
        conversation_id: str,
        max_messages: int = 10
    ) -> str:
        """
        Get formatted conversation context for RAG query.

        Args:
            conversation_id: Conversation ID
            max_messages: Maximum recent messages to include

        Returns:
            Formatted conversation context string
        """
        messages = self.get_conversation_history(conversation_id, limit=max_messages)

        if not messages:
            return ""

        context_parts = ["CONVERSATION HISTORY:"]
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")

        return "\n".join(context_parts)

    def get_previous_chunks(
        self,
        conversation_id: str,
        max_messages: int = 3
    ) -> List[str]:
        """
        Get chunk IDs from recent messages for context.

        This helps maintain context when asking follow-up questions
        by prioritizing previously retrieved chunks.

        Args:
            conversation_id: Conversation ID
            max_messages: Number of recent messages to check

        Returns:
            List of parent chunk IDs from previous retrievals
        """
        messages = self.get_conversation_history(conversation_id, limit=max_messages)

        chunk_ids = []
        for msg in messages:
            if msg.get("parent_chunk_ids"):
                chunk_ids.extend(msg["parent_chunk_ids"])

        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_chunks.append(chunk_id)

        return unique_chunks

    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation statistics.

        Args:
            conversation_id: Conversation ID

        Returns:
            Dict with conversation stats
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    created_at, last_message_at, message_count
                FROM conversations
                WHERE conversation_id = ?
            """, (conversation_id,))

            row = cursor.fetchone()

            if not row:
                return {
                    "exists": False,
                    "message_count": 0
                }

            return {
                "exists": True,
                "conversation_id": conversation_id,
                "created_at": row["created_at"],
                "last_message_at": row["last_message_at"],
                "message_count": row["message_count"]
            }

    def should_start_new_conversation(
        self,
        conversation_id: str,
        max_messages: int = 50
    ) -> bool:
        """
        Check if conversation should be reset due to length limit.

        Args:
            conversation_id: Conversation ID
            max_messages: Maximum messages per conversation

        Returns:
            True if should start new conversation
        """
        stats = self.get_conversation_stats(conversation_id)
        return stats.get("message_count", 0) >= max_messages

    def cleanup_old_conversations(self, days: int = 30):
        """
        Delete conversations older than specified days.

        Args:
            days: Number of days to keep
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Delete old messages
            cursor.execute("""
                DELETE FROM messages
                WHERE conversation_id IN (
                    SELECT conversation_id
                    FROM conversations
                    WHERE last_message_at < datetime('now', '-' || ? || ' days')
                )
            """, (days,))

            # Delete old conversations
            cursor.execute("""
                DELETE FROM conversations
                WHERE last_message_at < datetime('now', '-' || ? || ' days')
            """, (days,))

            deleted = cursor.rowcount
            conn.commit()

        logger.info(f"Cleaned up {deleted} old conversations")


# Singleton instance
_conversation_manager_instance = None


def get_conversation_manager(**kwargs) -> ConversationManager:
    """
    Get or create singleton ConversationManager instance.

    Args:
        **kwargs: Arguments to pass to ConversationManager constructor

    Returns:
        ConversationManager instance
    """
    global _conversation_manager_instance

    if _conversation_manager_instance is None:
        _conversation_manager_instance = ConversationManager(**kwargs)

    return _conversation_manager_instance


if __name__ == "__main__":
    # Test the conversation manager
    logging.basicConfig(level=logging.INFO)

    manager = get_conversation_manager(db_path="/tmp/test_conversations.db")

    # Create conversation
    conv_id = manager.create_conversation()
    print(f"Created conversation: {conv_id}")

    # Add messages
    manager.add_message(conv_id, "user", "What is the PTO policy?")
    manager.add_message(
        conv_id,
        "assistant",
        "The PTO policy...",
        query_type="document",
        parent_chunk_ids=["chunk1", "chunk2"]
    )
    manager.add_message(conv_id, "user", "How do I request it?")

    # Get history
    history = manager.get_conversation_history(conv_id)
    print(f"\nConversation history ({len(history)} messages):")
    for msg in history:
        print(f"  {msg['role']}: {msg['content']}")

    # Get context
    context = manager.get_conversation_context(conv_id)
    print(f"\nFormatted context:\n{context}")

    # Get previous chunks
    chunks = manager.get_previous_chunks(conv_id)
    print(f"\nPrevious chunks: {chunks}")

    # Get stats
    stats = manager.get_conversation_stats(conv_id)
    print(f"\nConversation stats: {stats}")
