"""
SQL Query Handler for Text-to-SQL functionality

Handles natural language to SQL conversion, query execution,
and result formatting for MS SQL Server databases.
"""

import logging
import pyodbc
import yaml
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import ollama

logger = logging.getLogger(__name__)


class SQLQueryHandler:
    """
    Handles SQL query generation, validation, execution, and result formatting.

    Uses LLM for text-to-SQL conversion and result formatting.
    Includes security validations to prevent SQL injection and unsafe queries.
    """

    def __init__(
        self,
        database_name: str,
        config_path: str = "config/databases.yaml",
        ollama_host: str = "http://localhost:11434",
        model_name: str = "llama3:8b"
    ):
        """
        Initialize SQL query handler.

        Args:
            database_name: Name of database config (e.g., "employee_directory")
            config_path: Path to databases.yaml configuration file
            ollama_host: Ollama server URL
            model_name: LLM model name for text-to-SQL and formatting
        """
        self.database_name = database_name
        self.model_name = model_name
        self.ollama_client = ollama.Client(host=ollama_host)

        # Load configuration
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Database configuration not found: {config_path}")

        with open(config_file, 'r') as f:
            all_configs = yaml.safe_load(f)

        if database_name not in all_configs:
            raise ValueError(f"Database '{database_name}' not found in configuration")

        self.config = all_configs[database_name]

        if not self.config.get('enabled', False):
            raise ValueError(f"Database '{database_name}' is not enabled in configuration")

        # Extract configuration
        self.connection_config = self.config['connection']
        self.security_config = self.config['security']
        self.schema_config = self.config['schema']

        # Get password from environment variable
        password_env = self.connection_config.get('password_env')
        if password_env:
            self.password = os.getenv(password_env)
            if not self.password:
                logger.warning(f"Environment variable {password_env} not set, password will be empty")
                self.password = ""
        else:
            self.password = self.connection_config.get('password', "")

        logger.info(f"SQLQueryHandler initialized for {database_name}")
        logger.info(f"Max rows: {self.security_config['max_rows']}, Timeout: {self.security_config['query_timeout_seconds']}s")

    def _get_connection_string(self) -> str:
        """Build MS SQL Server connection string."""
        conn_cfg = self.connection_config

        if conn_cfg.get('use_windows_auth', False):
            # Windows Authentication
            conn_str = (
                f"DRIVER={{{conn_cfg['driver']}}};"
                f"SERVER={conn_cfg['server']},{conn_cfg['port']};"
                f"DATABASE={conn_cfg['database']};"
                f"Trusted_Connection=yes;"
            )
        else:
            # SQL Server Authentication
            conn_str = (
                f"DRIVER={{{conn_cfg['driver']}}};"
                f"SERVER={conn_cfg['server']},{conn_cfg['port']};"
                f"DATABASE={conn_cfg['database']};"
                f"UID={conn_cfg['user']};"
                f"PWD={self.password};"
            )

        return conn_str

    def _build_schema_context(self) -> str:
        """Build schema context for LLM prompt."""
        schema_parts = []

        for table_name, table_info in self.schema_config.items():
            schema_parts.append(f"\nTable/View: {table_name}")
            schema_parts.append(f"Description: {table_info.get('description', 'N/A')}")
            schema_parts.append("Columns:")

            for col in table_info.get('columns', []):
                sensitive = " (SENSITIVE - avoid unless necessary)" if col.get('sensitive', False) else ""
                schema_parts.append(f"  - {col['name']} ({col['type']}): {col.get('description', 'N/A')}{sensitive}")

        return "\n".join(schema_parts)

    def _build_example_queries(self) -> str:
        """Build example queries for LLM prompt."""
        examples = []

        for table_name, table_info in self.schema_config.items():
            for example in table_info.get('example_queries', []):
                examples.append(f"\nQuestion: {example['question']}")
                examples.append(f"SQL: {example['sql']}")

        return "\n".join(examples)

    async def generate_sql(
        self,
        user_query: str,
        conversation_context: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate SQL query from natural language using LLM.

        Args:
            user_query: User's natural language query
            conversation_context: Optional conversation history for follow-up questions

        Returns:
            Tuple of (sql_query, metadata)
        """
        schema_context = self._build_schema_context()
        examples = self._build_example_queries()
        max_rows = self.security_config['max_rows']

        # Build prompt for text-to-SQL
        prompt_parts = [
            "You are a SQL query generator for MS SQL Server. Convert natural language questions to SQL queries.",
            f"\nDATABASE SCHEMA:\n{schema_context}",
            "\nCRITICAL RULES:",
            "1. ONLY generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, ALTER, etc.)",
            f"2. ALWAYS add 'TOP {max_rows}' to limit results",
            "3. ALWAYS exclude terminated employees by adding 'WHERE IsTerminated = 0' unless specifically asked about terminated employees",
            "4. Use LIKE with wildcards (%) for text searches: WHERE FirstName LIKE '%John%'",
            "5. For name searches, search both FirstName AND LastName",
            "6. Avoid selecting sensitive columns (AnnualRate) unless the question specifically asks for it",
            "7. Use proper date formatting for MS SQL Server (e.g., YEAR(HireDate) = 2024)",
            "8. Return ONLY the SQL query on a SINGLE LINE - no explanations, no markdown, no quotes, no line breaks in the SQL",
            "9. Format: SELECT TOP 1000 columns FROM table WHERE conditions",
            "\nCOMMON QUERY PATTERNS:",
            "- 'contact info' or 'contact information' = FirstName, LastName, Email, WorkPhone, MailCode, BuildingCode, Room, OnOffSite",
            "- 'location' or 'where does X sit' = FirstName, LastName, BuildingCode, Room, OnOffSite",
            "- 'phone' = FirstName, LastName, WorkPhone",
            "- 'email' = FirstName, LastName, Email or CompanyEmail",
            "- 'who does X report to' or 'X's manager/supervisor' = Use LEFT JOIN with SupervisorNo = EmpNo",
            "- 'who reports to X' or 'X's direct reports' = Use INNER JOIN with employee.SupervisorNo = manager.EmpNo",
            "\nJOIN SYNTAX:",
            "- Use table aliases: FROM vwPersonnelAll e LEFT JOIN vwPersonnelAll s ON e.SupervisorNo = s.EmpNo",
            "- Concatenate names: e.FirstName + ' ' + e.LastName as Employee",
            "- Check IsTerminated = 0 for BOTH tables in joins",
            f"\nEXAMPLE QUERIES:\n{examples}"
        ]

        if conversation_context:
            prompt_parts.insert(1, f"\nCONVERSATION HISTORY:\n{conversation_context}")
            prompt_parts.append("\nFOLLOW-UP QUERY RULES:")
            prompt_parts.append("1. Use conversation context to understand references (he, she, his, her, etc.)")
            prompt_parts.append("2. If [Context: Name: ...] exists, use that EXACT name in FirstName/LastName WHERE clauses")
            prompt_parts.append("3. If [Context: EmpNo: ...] exists, you can use that in WHERE clause: WHERE e.EmpNo = 'value'")
            prompt_parts.append("4. NEVER use UserName field in WHERE clauses - ONLY use FirstName, LastName, or EmpNo")
            prompt_parts.append("5. Example: Context shows 'Name: Wang Hinrichs | EmpNo: 12345'")
            prompt_parts.append("   Good: WHERE (e.FirstName LIKE '%Wang%' AND e.LastName LIKE '%Hinrichs%')")
            prompt_parts.append("   Good: WHERE e.EmpNo = '12345'")
            prompt_parts.append("   BAD: WHERE e.UserName LIKE '%Wang%' - NEVER DO THIS!")

        prompt_parts.append(f"\nUSER QUESTION: {user_query}")
        prompt_parts.append("\nGENERATED SQL:")

        prompt = "\n".join(prompt_parts)

        logger.debug(f"Generating SQL for query: {user_query}")

        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent SQL generation
                    "num_predict": 200
                },
                keep_alive=-1
            )

            sql = response['response'].strip()

            # Clean up the response - remove markdown, quotes, explanations
            sql = sql.replace('```sql', '').replace('```', '')
            sql = sql.strip('"').strip("'").strip()

            # Handle multi-line SQL or SQL with explanations
            if '\n' in sql:
                lines = [line.strip() for line in sql.split('\n') if line.strip()]

                # Find where SELECT starts and capture everything until end or semicolon
                sql_lines = []
                capturing = False
                for line in lines:
                    if line.upper().startswith('SELECT'):
                        capturing = True

                    if capturing:
                        sql_lines.append(line)
                        # Stop if we hit a semicolon or explanation text
                        if ';' in line or line.lower().startswith(('note:', 'explanation:', 'this query')):
                            break

                # Join the SQL lines into a single line
                if sql_lines:
                    sql = ' '.join(sql_lines)
                    # Remove trailing semicolon if present
                    sql = sql.rstrip(';').strip()

            logger.info(f"Generated SQL: {sql}")

            return sql, {"model": self.model_name, "temperature": 0.1}

        except Exception as e:
            logger.error(f"Error generating SQL: {e}", exc_info=True)
            raise ValueError(f"Failed to generate SQL query: {str(e)}")

    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL query for security and safety.

        Args:
            sql: SQL query to validate

        Returns:
            Dict with 'valid' (bool) and 'error' (str) if invalid
        """
        sql_upper = sql.upper()

        # 1. Check for forbidden keywords (anything that modifies data)
        forbidden_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'SP_',
            'XP_', 'BACKUP', 'RESTORE', 'SHUTDOWN'
        ]

        for keyword in forbidden_keywords:
            if keyword in sql_upper:
                return {
                    "valid": False,
                    "error": f"Query contains forbidden keyword: {keyword}"
                }

        # 2. Must start with SELECT
        if not sql_upper.strip().startswith('SELECT'):
            return {
                "valid": False,
                "error": "Only SELECT queries are allowed"
            }

        # 3. Must have TOP clause for MS SQL Server
        if 'TOP' not in sql_upper:
            return {
                "valid": False,
                "error": f"Query must include TOP {self.security_config['max_rows']} clause"
            }

        # 4. Check only allowed tables/views are referenced
        allowed_tables = self.security_config.get('allowed_tables', []) + \
                        self.security_config.get('allowed_views', [])

        # Simple check - look for FROM clause
        if 'FROM' in sql_upper:
            from_part = sql_upper.split('FROM')[1].split('WHERE')[0].split('ORDER')[0].split('GROUP')[0]
            table_referenced = False
            for table in allowed_tables:
                if table.upper() in from_part:
                    table_referenced = True
                    break

            if not table_referenced:
                return {
                    "valid": False,
                    "error": f"Query must reference only allowed tables: {', '.join(allowed_tables)}"
                }

        return {"valid": True}

    def execute_sql(self, sql: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute SQL query and return results.

        Args:
            sql: Validated SQL query

        Returns:
            Tuple of (results list, metadata dict)
        """
        logger.info(f"Executing SQL: {sql}")

        try:
            conn_str = self._get_connection_string()
            timeout = self.security_config['query_timeout_seconds']

            # Connect to database
            conn = pyodbc.connect(conn_str, timeout=timeout)
            cursor = conn.cursor()

            # Set query timeout
            cursor.execute(f"SET LOCK_TIMEOUT {timeout * 1000}")  # Convert to milliseconds

            # Execute query
            cursor.execute(sql)

            # Fetch results
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()

            # Convert to list of dicts
            results = []
            for row in rows:
                result_dict = {}
                for i, column in enumerate(columns):
                    value = row[i]
                    # Convert datetime objects to strings
                    if hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    result_dict[column] = value
                results.append(result_dict)

            cursor.close()
            conn.close()

            metadata = {
                "rows_returned": len(results),
                "columns": columns,
                "truncated": len(results) >= self.security_config['max_rows']
            }

            logger.info(f"Query returned {len(results)} rows")

            return results, metadata

        except pyodbc.Error as e:
            logger.error(f"Database error executing SQL: {e}", exc_info=True)
            raise RuntimeError(f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Error executing SQL: {e}", exc_info=True)
            raise RuntimeError(f"Query execution failed: {str(e)}")

    async def format_results(
        self,
        user_query: str,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Format SQL results as natural language using LLM.

        Args:
            user_query: Original user question
            results: Query results
            metadata: Query metadata (row count, etc.)

        Returns:
            Natural language formatted answer
        """
        rows_returned = metadata['rows_returned']
        max_rows = self.security_config['max_rows']

        # Handle no results
        if rows_returned == 0:
            return (
                "I couldn't find any results matching your query. "
                "Please check your search terms and try again."
            )

        # Build prompt for result formatting
        prompt = f"""You are formatting database query results. Provide a direct, concise answer with HTML formatting.

USER QUESTION: {user_query}

QUERY RESULTS ({rows_returned} rows):
{json.dumps(results[:10], indent=2)}

CRITICAL FORMATTING RULES:
1. Answer directly - NO preambles like "Here is..." or "The answer is..."
2. NO closing statements like "Let me know..." or notes about result count
3. Use HTML formatting:
   - Line breaks: Use newlines (\\n) - will be converted to <br> automatically
   - Email addresses: Format as <a href="mailto:EMAIL">EMAIL</a>
   - Phone numbers: Format as <a href="tel:PHONE">PHONE</a>
   - Lists: Use bullet points with • or numbered lists
4. For names: ALWAYS use FirstName and LastName fields, NEVER use UserName field
5. For contact info: Present as a formatted list with labels
6. For dates: Format nicely (e.g., "January 15, 2020")
7. Skip null/empty fields
8. NEVER include sensitive fields (AnnualRate)

FORMATTING EXAMPLES:

Question: "What is John Smith's email?"
Answer: "John Smith's email is <a href=\\"mailto:john.smith@company.com\\">john.smith@company.com</a>."

Question: "What is John Smith's contact info?"
Answer: "John Smith's contact information:\\n\\nEmail: <a href=\\"mailto:john.smith@company.com\\">john.smith@company.com</a>\\nPhone: <a href=\\"tel:555-1234\\">555-1234</a>\\nBuilding: 101, Room: 205\\nMail Code: MC-1234"

Question: "List employees in Engineering"
Answer: "Engineering department employees:\\n\\n• John Smith - <a href=\\"mailto:john@company.com\\">john@company.com</a>\\n• Jane Doe - <a href=\\"mailto:jane@company.com\\">jane@company.com</a>\\n• Bob Johnson - <a href=\\"mailto:bob@company.com\\">bob@company.com</a>"

FORMATTED ANSWER:"""

        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 500
                },
                keep_alive=-1
            )

            answer = response['response'].strip()

            # Convert newlines to HTML line breaks for better display
            answer = answer.replace('\n', '<br>')

            # Add overflow warning if results were truncated
            if metadata.get('truncated', False) and rows_returned >= max_rows:
                answer += (
                    f"<br><br>Note: This query matched more than {max_rows} results. "
                    f"Only the first {max_rows} are shown. "
                    f"Please contact IT if you need a complete dataset with all results."
                )

            return answer

        except Exception as e:
            logger.error(f"Error formatting results: {e}", exc_info=True)
            # Fallback to simple formatting
            return self._simple_format_results(user_query, results, metadata)

    def _simple_format_results(
        self,
        user_query: str,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> str:
        """Fallback simple result formatting if LLM fails. Uses HTML formatting."""
        rows_returned = metadata['rows_returned']

        if rows_returned == 0:
            return "No results found."

        answer_parts = [f"Found {rows_returned} result(s):<br><br>"]

        for i, result in enumerate(results[:10], 1):
            # Build name from FirstName and LastName if available
            name = None
            if 'FirstName' in result and 'LastName' in result:
                first = result.get('FirstName', '')
                last = result.get('LastName', '')
                if first or last:
                    name = f"{first} {last}".strip()

            if name:
                answer_parts.append(f"<strong>{i}. {name}</strong><br>")
            else:
                answer_parts.append(f"<strong>{i}.</strong><br>")

            for key, value in result.items():
                # Skip null values, sensitive fields, and name fields (already shown)
                if value is None or key == 'AnnualRate' or key in ['FirstName', 'LastName']:
                    continue

                # Format emails as mailto links
                if key in ['Email', 'CompanyEmail', 'SupervisorEmail', 'ManagerEmail', 'EmployeeEmail']:
                    answer_parts.append(f"  {key}: <a href=\"mailto:{value}\">{value}</a><br>")
                # Format phone numbers as tel links
                elif key in ['WorkPhone', 'Phone', 'SupervisorPhone', 'ManagerPhone']:
                    answer_parts.append(f"  {key}: <a href=\"tel:{value}\">{value}</a><br>")
                else:
                    answer_parts.append(f"  {key}: {value}<br>")

            answer_parts.append("<br>")  # Spacing between results

        if rows_returned > 10:
            answer_parts.append(f"<br>... and {rows_returned - 10} more results")

        return "".join(answer_parts)


def get_sql_query_handler(database_name: str, **kwargs) -> SQLQueryHandler:
    """
    Factory function to get SQL query handler.

    Args:
        database_name: Name of database (e.g., 'employee_directory')
        **kwargs: Additional arguments for SQLQueryHandler

    Returns:
        SQLQueryHandler instance
    """
    return SQLQueryHandler(database_name=database_name, **kwargs)
