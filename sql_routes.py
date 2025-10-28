"""
FastAPI routes for SQL-based query systems

Provides endpoints for querying SQL databases using natural language.
"""

import logging
import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from sql_query_handler import get_sql_query_handler
from conversation_manager import get_conversation_manager
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "/data/airgapped_rag"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")

# Create router
sql_router = APIRouter(prefix="", tags=["SQL Queries"])


class SQLQueryRequest(BaseModel):
    """Request model for SQL queries."""
    prompt: str
    conversation_id: Optional[str] = None


class SQLQueryResponse(BaseModel):
    """Response model for SQL queries."""
    answer: str
    rows_returned: int
    total_rows_available: int
    overflow_warning: Optional[str] = None
    conversation_id: Optional[str] = None
    conversation_warning: Optional[str] = None
    query_type: str
    execution_time_ms: Optional[int] = None


# Initialize conversation manager (shared with document RAG)
conversation_manager = get_conversation_manager(
    db_path=str(DATA_DIR / "conversations.db")
)


@sql_router.post("/query-employee", response_model=SQLQueryResponse)
async def query_employee_directory(request: SQLQueryRequest):
    """
    Query the employee directory using natural language.

    This endpoint converts natural language questions to SQL queries,
    executes them against the employee directory database, and returns
    formatted results.

    Example queries:
    - "Find John Smith"
    - "List all employees in Engineering"
    - "Who was hired in 2024?"
    - "Show employees with 'Manager' in their title"
    """
    logger.info(f"Employee directory query: {request.prompt}")

    try:
        # Initialize SQL query handler for employee directory
        sql_handler = get_sql_query_handler(
            database_name="employee_directory",
            ollama_host=OLLAMA_HOST,
            model_name=LLM_MODEL
        )

        # Handle conversation ID
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = conversation_manager.create_conversation()
            logger.info(f"Created new conversation: {conversation_id}")

        # Check if conversation limit reached
        conv_limit_reached = conversation_manager.should_start_new_conversation(
            conversation_id,
            max_messages=50
        )

        if conv_limit_reached:
            logger.warning(f"Conversation {conversation_id} reached message limit")

        # Get conversation context for follow-up questions
        conversation_context = conversation_manager.get_conversation_context(
            conversation_id,
            max_messages=10
        )

        # Add user message to conversation
        conversation_manager.add_message(
            conversation_id,
            role="user",
            content=request.prompt,
            query_type="employee_directory"
        )

        # Step 1: Generate SQL from natural language
        try:
            sql_query, sql_metadata = await sql_handler.generate_sql(
                user_query=request.prompt,
                conversation_context=conversation_context if conversation_context else None
            )
        except ValueError as e:
            error_message = (
                "I'm having trouble understanding that query. Could you rephrase it? "
                "For example: 'Find employees in Engineering' or 'Show me John Smith's information'"
            )
            logger.error(f"SQL generation failed: {e}")

            # Store error in conversation
            conversation_manager.add_message(
                conversation_id,
                role="assistant",
                content=error_message,
                query_type="employee_directory"
            )

            return SQLQueryResponse(
                answer=error_message,
                rows_returned=0,
                total_rows_available=0,
                conversation_id=conversation_id,
                query_type="employee_directory"
            )

        # Step 2: Validate SQL for security
        validation = sql_handler.validate_sql(sql_query)
        if not validation['valid']:
            error_message = (
                "I couldn't process that request safely. Please try rephrasing your question "
                "or contact IT for assistance with complex queries."
            )
            logger.error(f"SQL validation failed: {validation['error']}")

            # Store error in conversation
            conversation_manager.add_message(
                conversation_id,
                role="assistant",
                content=error_message,
                query_type="employee_directory"
            )

            return SQLQueryResponse(
                answer=error_message,
                rows_returned=0,
                total_rows_available=0,
                conversation_id=conversation_id,
                query_type="employee_directory"
            )

        # Step 3: Execute SQL query
        try:
            results, exec_metadata = sql_handler.execute_sql(sql_query)
        except RuntimeError as e:
            error_message = (
                "I'm unable to query the employee directory right now. "
                "Please try again in a few moments or contact IT if the issue persists."
            )
            logger.error(f"SQL execution failed: {e}")

            # Store error in conversation
            conversation_manager.add_message(
                conversation_id,
                role="assistant",
                content=error_message,
                query_type="employee_directory"
            )

            return SQLQueryResponse(
                answer=error_message,
                rows_returned=0,
                total_rows_available=0,
                conversation_id=conversation_id,
                query_type="employee_directory"
            )

        # Step 4: Format results as natural language
        try:
            formatted_answer = await sql_handler.format_results(
                user_query=request.prompt,
                results=results,
                metadata=exec_metadata
            )
        except Exception as e:
            # Use fallback formatting
            logger.warning(f"LLM formatting failed, using fallback: {e}")
            formatted_answer = sql_handler._simple_format_results(
                user_query=request.prompt,
                results=results,
                metadata=exec_metadata
            )

        # Store assistant message in conversation
        # Include structured result data for better follow-up question handling
        conversation_content = formatted_answer

        # Append structured context (hidden from user, visible to LLM for follow-ups)
        if results and len(results) <= 3:
            # For small result sets, include key fields for follow-up context
            context_data = []
            for result in results:
                # Extract key identifying fields
                person_info = []

                # Try to extract name from various fields
                name = None
                if 'FirstName' in result and 'LastName' in result:
                    # Separate fields (standard queries)
                    first = result.get('FirstName', '')
                    last = result.get('LastName', '')
                    if first or last:
                        name = f"{first} {last}".strip()
                elif 'Employee' in result:
                    # Concatenated field from JOIN queries
                    name = result.get('Employee')
                elif 'EmployeeFirstName' in result and 'EmployeeLastName' in result:
                    # Aliased fields from JOIN queries
                    first = result.get('EmployeeFirstName', '')
                    last = result.get('EmployeeLastName', '')
                    if first or last:
                        name = f"{first} {last}".strip()

                if name:
                    person_info.append(f"Name: {name}")

                # Include EmpNo if available (more reliable than UserName)
                if 'EmpNo' in result:
                    person_info.append(f"EmpNo: {result.get('EmpNo', '')}")
                elif 'PersonnelId' in result:
                    person_info.append(f"ID: {result.get('PersonnelId', '')}")

                if person_info:
                    context_data.append(" | ".join(person_info))

            if context_data:
                conversation_content += "\n\n[Context: " + "; ".join(context_data) + "]"

        conversation_manager.add_message(
            conversation_id,
            role="assistant",
            content=conversation_content,
            query_type="employee_directory"
        )

        # Determine overflow warning
        overflow_warning = None
        if exec_metadata.get('truncated', False):
            max_rows = sql_handler.security_config['max_rows']
            overflow_warning = (
                f"This query matched more than {max_rows} results. "
                f"Only the first {max_rows} are shown. "
                f"Please contact IT if you need a complete dataset with all results."
            )

        return SQLQueryResponse(
            answer=formatted_answer,
            rows_returned=exec_metadata['rows_returned'],
            total_rows_available=exec_metadata['rows_returned'],  # We don't know actual total without COUNT
            overflow_warning=overflow_warning,
            conversation_id=conversation_id,
            conversation_warning="Conversation limit reached - please start a new conversation" if conv_limit_reached else None,
            query_type="employee_directory"
        )

    except Exception as e:
        logger.error(f"Unexpected error in employee directory query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred processing your query")


@sql_router.post("/query-purchasing", response_model=SQLQueryResponse)
async def query_purchasing_system(request: SQLQueryRequest):
    """
    Query the purchasing system using natural language.

    This endpoint is currently disabled. It will be enabled once the
    purchasing database schema is configured.

    Example queries (when enabled):
    - "Status of PO-12345"
    - "List all pending purchase orders"
    - "Show recent POs over $10,000"
    """
    return SQLQueryResponse(
        answer="The purchasing system query feature is not yet configured. Please contact IT for assistance.",
        rows_returned=0,
        total_rows_available=0,
        query_type="purchasing_system"
    )
