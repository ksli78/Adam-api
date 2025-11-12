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
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
import ollama

logger = logging.getLogger(__name__)


# Title expansion dictionary for smart matching
# Maps common title terms and abbreviations to their variants
TITLE_EXPANSIONS = {
    # Vice President variants
    "vp": ["VP", "Vice President", "V.P.", "Vice-President"],
    "vice president": ["VP", "Vice President", "V.P.", "Vice-President"],
    "vice-president": ["VP", "Vice President", "V.P.", "Vice-President"],

    # Senior Vice President
    "svp": ["SVP", "Senior VP", "Senior Vice President", "Sr. Vice President", "Sr VP"],
    "senior vp": ["SVP", "Senior VP", "Senior Vice President", "Sr. Vice President"],
    "senior vice president": ["SVP", "Senior VP", "Senior Vice President", "Sr. Vice President"],

    # Executive Vice President
    "evp": ["EVP", "Executive VP", "Executive Vice President", "Exec VP"],
    "executive vp": ["EVP", "Executive VP", "Executive Vice President"],
    "executive vice president": ["EVP", "Executive VP", "Executive Vice President"],

    # Director variants
    "director": ["Director", "Dir"],
    "dir": ["Director", "Dir"],

    # Manager variants
    "manager": ["Manager", "Mgr"],
    "mgr": ["Manager", "Mgr"],

    # Operations variants
    "operations": ["Operations", "Operation", "Ops"],
    "operation": ["Operations", "Operation", "Ops"],
    "ops": ["Operations", "Operation", "Ops"],

    # Technology variants
    "technology": ["Technology", "Tech"],
    "tech": ["Technology", "Tech"],

    # Information variants
    "information": ["Information", "Info"],
    "info": ["Information", "Info"],

    # Common department abbreviations
    "hr": ["HR", "Human Resources", "Human Resource"],
    "human resources": ["HR", "Human Resources", "Human Resource"],
    "human resource": ["HR", "Human Resources", "Human Resource"],

    "it": ["IT", "Information Technology", "Info Tech"],
    "information technology": ["IT", "Information Technology", "Info Tech"],

    # Enterprise variants
    "enterprise": ["Enterprise", "Ent"],
    "ent": ["Enterprise", "Ent"],
}


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

    def _normalize_query(self, query: str) -> str:
        """
        Normalize user query for better matching.

        - Replace dashes and slashes with spaces (but keep underscores)
        - Collapse multiple spaces into one
        - Lowercase for comparison

        Args:
            query: Raw user query

        Returns:
            Normalized query string
        """
        # Replace dashes and slashes with spaces (keep underscores)
        normalized = re.sub(r'[-/]+', ' ', query)

        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        # Trim and lowercase
        normalized = normalized.strip().lower()

        logger.debug(f"Normalized query: '{query}' → '{normalized}'")
        return normalized

    def _expand_keyword(self, keyword: str) -> List[str]:
        """
        Expand a keyword into all its variants.

        - Check TITLE_EXPANSIONS dictionary for known synonyms
        - Add simple plural/singular variants (remove/add 's')
        - Return unique list of expansions

        Args:
            keyword: Single keyword to expand

        Returns:
            List of keyword variants
        """
        keyword_lower = keyword.lower()
        variants = set()

        # Check expansion dictionary
        if keyword_lower in TITLE_EXPANSIONS:
            variants.update(TITLE_EXPANSIONS[keyword_lower])
            logger.debug(f"Found expansions for '{keyword}': {TITLE_EXPANSIONS[keyword_lower]}")
        else:
            # Not in dictionary, add the original
            variants.add(keyword)

        # Add simple plural/singular handling
        if keyword_lower.endswith('s') and len(keyword_lower) > 3:
            # Try singular (remove 's')
            singular = keyword_lower[:-1]
            variants.add(singular)
            variants.add(singular.capitalize())
        elif not keyword_lower.endswith('s'):
            # Try plural (add 's')
            plural = keyword_lower + 's'
            variants.add(plural)
            variants.add(plural.capitalize())

        # Always include the original casing
        variants.add(keyword)

        result = list(variants)
        logger.debug(f"Expanded '{keyword}' to: {result}")
        return result

    def _detect_title_search(self, query: str) -> bool:
        """
        Detect if query is asking for someone by their job title.

        Patterns:
        - "Who is the [TITLE]"
        - "Who is the [TITLE] of [AREA]"
        - "Find the [TITLE]"

        Args:
            query: User's query

        Returns:
            True if this is a title-based search
        """
        query_lower = query.lower()

        # Common patterns for title searches
        title_patterns = [
            r'who\s+is\s+the\s+',
            r'find\s+the\s+',
            r'show\s+me\s+the\s+',
            r'get\s+the\s+',
            r'what\s+is\s+the\s+name\s+of\s+the\s+',
        ]

        for pattern in title_patterns:
            if re.search(pattern, query_lower):
                logger.debug(f"Detected title search pattern: {pattern}")
                return True

        return False

    def _detect_phrases(self, normalized_query: str) -> List[str]:
        """
        Detect multi-word phrases in the query before tokenization.

        This is critical for handling queries like "vice president" which should
        map to "VP", not be split into separate "vice" and "president" tokens.

        Args:
            normalized_query: Normalized query string

        Returns:
            List of phrases/keywords (preserving multi-word phrases)
        """
        # Multi-word phrases to detect (order matters - longest first)
        MULTI_WORD_PHRASES = [
            "senior vice president",
            "executive vice president",
            "vice president",
            "senior vp",
            "executive vp",
            "human resources",
            "human resource",
            "information technology",
            "info tech",
        ]

        # Track which positions are part of a phrase
        query_lower = normalized_query.lower()
        phrase_positions = set()
        detected_phrases = []

        # Find all multi-word phrases
        for phrase in MULTI_WORD_PHRASES:
            start = 0
            while True:
                pos = query_lower.find(phrase, start)
                if pos == -1:
                    break

                # Check if this position is already part of another phrase
                phrase_range = range(pos, pos + len(phrase))
                if not any(p in phrase_positions for p in phrase_range):
                    # Mark positions as used
                    phrase_positions.update(phrase_range)
                    detected_phrases.append((pos, phrase))
                    logger.debug(f"Detected phrase '{phrase}' at position {pos}")

                start = pos + 1

        # If we found phrases, reconstruct the query with phrase markers
        if detected_phrases:
            # Sort by position
            detected_phrases.sort(key=lambda x: x[0])

            # Build final keyword list
            keywords = []
            last_end = 0

            for pos, phrase in detected_phrases:
                # Add any words before this phrase
                before_text = normalized_query[last_end:pos].strip()
                if before_text:
                    keywords.extend(before_text.split())

                # Add the phrase as a single keyword
                keywords.append(phrase)
                last_end = pos + len(phrase)

            # Add any remaining words after last phrase
            after_text = normalized_query[last_end:].strip()
            if after_text:
                keywords.extend(after_text.split())

            logger.debug(f"After phrase detection: {keywords}")
            return keywords
        else:
            # No phrases detected, just split normally
            return normalized_query.split()

    def _build_smart_title_sql(self, user_query: str, normalized_query: str) -> Optional[str]:
        """
        Build smart SQL for title searches with keyword expansion and tiered matching.

        Process:
        1. Detect multi-word phrases (e.g., "vice president")
        2. Extract title keywords from query
        3. Expand each keyword (synonyms + plurals)
        4. Generate SQL with OR groups for each keyword
        5. Require ALL keyword groups to match (strict)

        Args:
            user_query: Original user query
            normalized_query: Normalized query

        Returns:
            SQL query string or None if can't generate
        """
        # Detect multi-word phrases first (before tokenization)
        tokens = self._detect_phrases(normalized_query)

        # Remove common question words
        stop_words = ['who', 'is', 'the', 'of', 'for', 'a', 'an', 'in', 'at', 'to', 'find', 'show', 'me', 'get', 'what', 'name']
        keywords = [t for t in tokens if t not in stop_words and len(t) > 1]

        if not keywords:
            logger.warning("No meaningful keywords extracted from title search")
            return None

        logger.info(f"[SMART TITLE SEARCH] Extracted keywords: {keywords}")

        # Expand each keyword
        expanded_groups = []
        for keyword in keywords:
            variants = self._expand_keyword(keyword)
            if variants:
                expanded_groups.append(variants)

        if not expanded_groups:
            return None

        # Build WHERE clause with OR groups
        # Each group is OR'd (flexible), but ALL groups must match (accurate)
        or_clauses = []
        for group in expanded_groups:
            # Create OR conditions for each variant in the group
            variant_conditions = [f"BusinessTitle LIKE '%{variant}%'" for variant in group]
            or_clause = "(" + " OR ".join(variant_conditions) + ")"
            or_clauses.append(or_clause)

        # Combine with AND (all keyword groups must match)
        where_clause = " AND ".join(or_clauses)

        # Build complete SQL
        sql = f"SELECT TOP 1000 FirstName, LastName, BusinessTitle, HomeDept, Email, WorkPhone, BuildingCode, Room FROM vwPersonnelAll WHERE {where_clause} AND IsTerminated = 0"

        logger.info(f"[TIER 1] Generated smart SQL: {sql}")
        return sql

    async def generate_sql(
        self,
        user_query: str,
        conversation_context: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate SQL query from natural language.

        Uses smart title search for "Who is the [TITLE]" queries,
        falls back to LLM for other query types.

        Args:
            user_query: User's natural language query
            conversation_context: Optional conversation history for follow-up questions

        Returns:
            Tuple of (sql_query, metadata)
        """
        logger.info(f"Generating SQL for query: {user_query}")

        # Try smart title search first (for "Who is the [TITLE]" queries)
        if self._detect_title_search(user_query):
            logger.info("[SMART SEARCH] Detected title-based query")
            normalized_query = self._normalize_query(user_query)
            smart_sql = self._build_smart_title_sql(user_query, normalized_query)

            if smart_sql:
                logger.info(f"[SMART SEARCH] Using smart SQL generation: {smart_sql}")
                return smart_sql, {"method": "smart_title_search", "temperature": None}

            # If smart SQL generation failed, fall through to LLM
            logger.warning("[SMART SEARCH] Failed to generate smart SQL, falling back to LLM")

        # Fall back to LLM-based SQL generation
        logger.info("[LLM FALLBACK] Using LLM for SQL generation")
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
            "6. NEVER select ID fields (PersonnelId, EmpNo, SupervisorNo) unless specifically needed for joins or context tracking",
            "7. Avoid selecting sensitive columns (AnnualRate) unless the question specifically asks for it",
            "8. Use proper date formatting for MS SQL Server (e.g., YEAR(HireDate) = 2024)",
            "9. Return ONLY the SQL query on a SINGLE LINE - no explanations, no markdown, no quotes, no line breaks in the SQL",
            "10. Format: SELECT TOP 1000 columns FROM table WHERE conditions",
            "\nCOMMON QUERY PATTERNS:",
            "- 'Who is the [TITLE]' or 'Who is the [TITLE] of [DEPARTMENT/AREA]' = Search BusinessTitle field",
            "  * CRITICAL: This is asking for a person BY THEIR JOB TITLE, not by name!",
            "  * Parse the title words: 'director of enterprise operations' → search BusinessTitle for 'director', 'enterprise', 'operations'",
            "  * Use multiple LIKE conditions: WHERE BusinessTitle LIKE '%director%' AND BusinessTitle LIKE '%enterprise%' AND BusinessTitle LIKE '%operations%'",
            "  * Examples:",
            "    - 'Who is the VP of Operations' → WHERE BusinessTitle LIKE '%VP%' AND BusinessTitle LIKE '%Operations%'",
            "    - 'Who is the director of enterprise operations' → WHERE BusinessTitle LIKE '%director%' AND BusinessTitle LIKE '%enterprise%' AND BusinessTitle LIKE '%operations%'",
            "    - 'Who is the Chief Technology Officer' → WHERE BusinessTitle LIKE '%Chief%' AND BusinessTitle LIKE '%Technology%' AND BusinessTitle LIKE '%Officer%'",
            "  * Select: FirstName, LastName, BusinessTitle, HomeDept, Email, WorkPhone, BuildingCode, Room",
            "- 'department' or 'list people in/from department' or 'who works in/for DEPT-CODE' = Search HomeDept field (NOT Company!)",
            "  * Department codes like 'ENVR-001', 'ENGR-003' go in HomeDept field",
            "  * Example: WHERE HomeDept LIKE '%ENVR-001%'",
            "  * Select: FirstName, LastName, Email, WorkPhone, BusinessTitle, HomeDept",
            "- 'contact info' or 'contact information' = Use LEFT JOIN to include supervisor info, select: LastName, FirstName, MiddleName, HomeDept, WorkPhone, BuildingCode, Room, MailCode, Email, supervisor's LastName, FirstName, MiddleName",
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
            prompt_parts.append("1. PRIORITY: If the question explicitly mentions a person's name, use that name (ignore pronouns)")
            prompt_parts.append("   Example: 'When did Khaled get hired?' → Use 'Khaled' even if context has multiple people")
            prompt_parts.append("2. CRITICAL: Pronouns (he, she, his, her, him, his, hers, they, their) are NOT names!")
            prompt_parts.append("   NEVER search for pronouns as literal names in WHERE clauses")
            prompt_parts.append("   BAD: WHERE FirstName LIKE '%her%' ← Searching for name 'her' - WRONG!")
            prompt_parts.append("   GOOD: Use context to find the actual person's name")
            prompt_parts.append("3. For pronoun references, determine WHO from conversation context:")
            prompt_parts.append("   - Context shows: [Employee: Khaled Sliman | Manager: Colly Edgeworth]")
            prompt_parts.append("   - Previous answer was about Colly (the manager)")
            prompt_parts.append("   - 'her' or 'she' = Colly Edgeworth")
            prompt_parts.append("   - Use: WHERE (FirstName LIKE '%Colly%' AND LastName LIKE '%Edgeworth%')")
            prompt_parts.append("4. Determine WHO based on conversation flow:")
            prompt_parts.append("   - If previous Q was 'Who is X's boss?', then 'her department' = Boss's department")
            prompt_parts.append("   - If previous Q was 'Who reports to X?', then 'their info' = Employees' info")
            prompt_parts.append("5. Extract actual names from context, NEVER use pronouns in SQL")
            prompt_parts.append("6. NEVER use UserName field in WHERE clauses - ONLY use FirstName, LastName, or EmpNo")
            prompt_parts.append("\nEXAMPLES:")
            prompt_parts.append("Context: [Employee: Khaled Sliman | Manager: Colly Edgeworth]")
            prompt_parts.append("Previous Q: 'Who is Khaled's boss?' Answer: '...Colly Edgeworth...'")
            prompt_parts.append("Q: 'What is her department?' ← 'her' is a PRONOUN referring to Colly")
            prompt_parts.append("BAD SQL: WHERE FirstName LIKE '%her%' ← NO! Don't search for 'her' as a name!")
            prompt_parts.append("GOOD SQL: SELECT TOP 1000 FirstName, LastName, HomeDept FROM vwPersonnelAll WHERE (FirstName LIKE '%Colly%' AND LastName LIKE '%Edgeworth%') AND IsTerminated = 0")
            prompt_parts.append("")
            prompt_parts.append("Context: [Employee: Khaled Sliman | Manager: Colly Edgeworth]")
            prompt_parts.append("Q: 'When did Khaled get hired?' ← Khaled explicitly mentioned!")
            prompt_parts.append("→ Use Khaled (not Colly), include HireDate")
            prompt_parts.append("→ SQL: SELECT TOP 1000 FirstName, LastName, HireDate FROM vwPersonnelAll WHERE (FirstName LIKE '%Khaled%' AND LastName LIKE '%Sliman%') AND IsTerminated = 0")

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

    def _generate_html_table(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate HTML table directly in Python for large result sets.

        Args:
            results: Query results as list of dicts

        Returns:
            HTML table string
        """
        if not results:
            return ""

        # Get columns from first result, excluding ID fields
        all_columns = list(results[0].keys())
        columns = [col for col in all_columns if col.lower() not in ['personnelid', 'empno', 'supervisorno']]

        # Start table with styling
        html_parts = ['<table class="adam-ai-table" style="border-collapse: collapse; width: 100%;">']

        # Add header row
        html_parts.append('<tr>')
        for col in columns:
            # Convert camelCase/PascalCase to Title Case with spaces
            display_name = col
            # Add spaces before capitals
            display_name = re.sub(r'([A-Z])', r' \1', display_name).strip()
            # Handle common abbreviations
            display_name = display_name.replace('Email', 'Email').replace('Dept', 'Department')

            html_parts.append(f'<th style="background-color: #D6D4D4; padding: 4px; text-align: left;">{display_name}</th>')
        html_parts.append('</tr>')

        # Add data rows with zebra striping for >3 rows
        use_zebra = len(results) > 3

        for idx, row in enumerate(results):
            # Alternate row color for even rows (0-indexed, so idx 1, 3, 5 are "even" visually)
            row_style = ""
            if use_zebra and idx % 2 == 1:
                row_style = ' style="background-color: #F5F5F5;"'

            html_parts.append(f'<tr{row_style}>')

            for col in columns:
                value = row.get(col, '')

                # Skip null/None values
                if value is None or value == '':
                    value = ''
                else:
                    # Convert to string
                    value = str(value)

                    # Make emails clickable
                    if 'email' in col.lower() and '@' in value:
                        value = f'<a href="mailto:{value}">{value}</a>'
                    # Make phone numbers clickable
                    elif 'phone' in col.lower() and value.strip():
                        value = f'<a href="tel:{value}">{value}</a>'

                html_parts.append(f'<td style="padding: 4px; vertical-align: top; white-space: nowrap;">{value}</td>')

            html_parts.append('</tr>')

        html_parts.append('</table>')

        return ''.join(html_parts)

    async def format_results(
        self,
        user_query: str,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Format SQL results as natural language using LLM or direct HTML generation.

        For large result sets (>10 rows), generates HTML table directly in Python.
        For small results or contact info queries, uses LLM for natural formatting.

        Args:
            user_query: Original user question
            results: Query results
            metadata: Query metadata (row count, etc.)

        Returns:
            Natural language formatted answer or HTML table
        """
        rows_returned = metadata['rows_returned']
        max_rows = self.security_config['max_rows']

        # Handle no results
        if rows_returned == 0:
            return (
                "I couldn't find any results matching your query. "
                "Please check your search terms and try again."
            )

        # For large result sets (>10 rows), generate table directly in Python
        # This is much faster and more reliable than LLM generation
        if rows_returned > 10:
            logger.info(f"Using Python-based table generation for {rows_returned} rows")
            table_html = self._generate_html_table(results)

            # Add overflow warning if needed
            if metadata.get('truncated', False) and rows_returned >= max_rows:
                table_html += (
                    f"<br><br>Note: This query matched more than {max_rows} results. "
                    f"Only the first {max_rows} are shown. "
                    f"Please contact IT if you need a complete dataset with all results."
                )

            return table_html

        # Build prompt for result formatting
        prompt = f"""You are formatting database query results. Provide a direct, concise answer with HTML formatting.

USER QUESTION: {user_query}

QUERY RESULTS ({rows_returned} rows):
{json.dumps(results, indent=2)}

CRITICAL FORMATTING RULES:
1. Answer directly - NO preambles like "Here is..." or "The answer is..."
2. DO NOT repeat or echo the user's question in your answer
3. NO closing statements like "Let me know..." or notes about result count
4. Choose format based on result type:
   A. SINGLE RESULT or CONTACT INFO: Use label:value format with line breaks
      - Single line break (\n) between fields
      - ABSOLUTELY NO triple line breaks (\n\n\n)
      - Email addresses: Format as <a href="mailto:EMAIL">EMAIL</a>
      - Phone numbers: Format as <a href="tel:PHONE">PHONE</a>
   B. MULTIPLE RESULTS (lists): Use HTML TABLE format with:
      - NO <br> tags inside table elements (no <br> in <th>, <td>, or <tr>)
      - Header style: <th style="background-color: #D6D4D4; padding: 4px; text-align: left;">
      - For >3 rows: Alternate row colors using <tr style="background-color: #F5F5F5;"> for even rows
      - Cell style: <td style="padding: 4px; vertical-align: top;">
      - Make emails clickable: <a href="mailto:EMAIL">EMAIL</a>
      - Table style: <table style="border-collapse: collapse; width: 100%;">
5. For names: ALWAYS use FirstName and LastName fields, NEVER use UserName field
6. NEVER display ID fields (PersonnelId, EmpNo, SupervisorNo) in output - skip them entirely
7. For dates: Format nicely (e.g., "January 15, 2020")
8. Skip null/empty fields
9. NEVER include sensitive fields (AnnualRate)

FORMATTING EXAMPLES:

Question: "What is John Smith's email?"
GOOD: "John Smith's email is <a href=\\"mailto:john.smith@company.com\\">john.smith@company.com</a>."
BAD: "What is John Smith's email?\\n\\nJohn Smith's email is..." ← DON'T repeat question!

Question: "What department does Colly work for?"
GOOD: "Colly works for the ENGR-003 department."
BAD: "What department does Colly work for? Colly works for..." ← DON'T echo question!

Question: "What is John Smith's contact info?"
GOOD: "John Smith's contact information:\\n\\nName: John A Smith\\nOrg Code: ENGR-001\\nPhone: <a href=\\"tel:555-1234\\">555-1234</a>\\nBuilding: BLDG-1, Room: 404B\\nMail Code: MC-100\\nEmail: <a href=\\"mailto:john.smith@company.com\\">john.smith@company.com</a>\\nSupervisor: Jane K Johnson"
BAD: "John Smith's contact information:\\n\\n\\n\\nEmail..." ← Too many line breaks!

Question: "Who is Khaled's boss?"
GOOD: "Khaled Sliman's boss is Colly Edgeworth, Senior Project Manager.\\n\\nEmail: <a href=\\"mailto:colly@acme.com\\">colly@acme.com</a>\\nPhone: <a href=\\"tel:555-1234\\">555-1234</a>"
BAD: "Khaled Sliman's boss is:\\n\\n\\nColly Edgeworth, Senior Project Manager\\n\\n\\nContact information:\\n\\n\\nEmail..." ← Way too many blank lines!

Question: "Who is the VP of Operations?"
GOOD: "Jane Doe is the VP of Operations.\\n\\nDepartment: OPS-001\\nEmail: <a href=\\"mailto:jane.doe@company.com\\">jane.doe@company.com</a>\\nPhone: <a href=\\"tel:555-1234\\">555-1234</a>\\nBuilding: BLDG-2, Room: 201"
BAD: "The VP of Operations is Jane Doe..." ← Use natural "[Name] is the [Title]" format!

Question: "Who is the director of enterprise operations?"
GOOD: "John Smith is the Director of Enterprise Operations.\\n\\nDepartment: ENT-OPS\\nEmail: <a href=\\"mailto:john.smith@company.com\\">john.smith@company.com</a>\\nPhone: <a href=\\"tel:555-5678\\">555-5678</a>\\nBuilding: BLDG-1, Room: 305"
BAD: "Here is the information for the director..." ← Answer directly with name and title!

Question: "List employees in Engineering"
Answer: "<table style=\\"border-collapse: collapse; width: 100%;\\">
<tr>
<th style=\\"background-color: #D6D4D4; padding: 4px; text-align: left;\\">First Name</th>
<th style=\\"background-color: #D6D4D4; padding: 4px; text-align: left;\\">Last Name</th>
<th style=\\"background-color: #D6D4D4; padding: 4px; text-align: left;\\">Email</th>
<th style=\\"background-color: #D6D4D4; padding: 4px; text-align: left;\\">Title</th>
</tr>
<tr>
<td style=\\"padding: 4px; vertical-align: top;\\">John</td>
<td style=\\"padding: 4px; vertical-align: top;\\">Smith</td>
<td style=\\"padding: 4px; vertical-align: top;\\"><a href=\\"mailto:john@company.com\\">john@company.com</a></td>
<td style=\\"padding: 4px; vertical-align: top;\\">Engineer</td>
</tr>
<tr style=\\"background-color: #F5F5F5;\\">
<td style=\\"padding: 4px; vertical-align: top;\\">Jane</td>
<td style=\\"padding: 4px; vertical-align: top;\\">Doe</td>
<td style=\\"padding: 4px; vertical-align: top;\\"><a href=\\"mailto:jane@company.com\\">jane@company.com</a></td>
<td style=\\"padding: 4px; vertical-align: top;\\">Manager</td>
</tr>
<tr>
<td style=\\"padding: 4px; vertical-align: top;\\">Bob</td>
<td style=\\"padding: 4px; vertical-align: top;\\">Johnson</td>
<td style=\\"padding: 4px; vertical-align: top;\\"><a href=\\"mailto:bob@company.com\\">bob@company.com</a></td>
<td style=\\"padding: 4px; vertical-align: top;\\">Analyst</td>
</tr>
</table>"

FORMATTED ANSWER:"""

        # Calculate dynamic token limit based on number of rows
        # Estimate ~80 tokens per row for table formatting
        # Add base of 500 tokens for instructions and headers
        # Cap at 8000 tokens maximum
        num_predict = min(8000, 500 + (rows_returned * 80))

        logger.info(f"Formatting {rows_returned} rows with num_predict={num_predict}")

        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": num_predict
                },
                keep_alive=-1
            )

            answer = response['response'].strip()

            logger.info(f"LLM response length: {len(answer)} chars, starts with: {answer[:100]}")
            logger.info(f"Response contains table: {'<table' in answer.lower()}")

            # Count how many rows are in the generated table
            if '<table' in answer.lower():
                row_count = answer.lower().count('<tr>') - 1  # Subtract header row
                logger.info(f"Generated table has {row_count} data rows (expected {rows_returned})")

            # Check if answer contains HTML table
            if '<table' in answer.lower():
                # For tables: Remove newlines inside table elements but preserve table structure
                # Extract table and process separately
                def clean_table(match):
                    table_content = match.group(0)
                    # Remove newlines within table tags, but keep the table structure
                    table_content = table_content.replace('\n', '')
                    return table_content

                # Process tables separately
                answer = re.sub(r'<table[^>]*>.*?</table>', clean_table, answer, flags=re.DOTALL | re.IGNORECASE)

                # Convert remaining newlines (outside tables) to <br>
                parts = re.split(r'(<table[^>]*>.*?</table>)', answer, flags=re.DOTALL | re.IGNORECASE)
                for i in range(len(parts)):
                    if not parts[i].lower().startswith('<table'):
                        parts[i] = parts[i].replace('\n', '<br>')
                answer = ''.join(parts)
            else:
                # No table: Convert newlines to HTML line breaks for better display
                answer = answer.replace('\n', '<br>')

                # Aggressively collapse excessive consecutive line breaks
                # This is a safety net for LLM inconsistencies

                # First, normalize any whitespace between <br> tags
                answer = re.sub(r'<br>\s+<br>', '<br><br>', answer)

                # Then collapse 3+ consecutive <br> tags to exactly 2
                while '<br><br><br>' in answer:
                    answer = answer.replace('<br><br><br>', '<br><br>')

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

    async def format_results_stream(
        self,
        user_query: str,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Stream SQL results as natural language using LLM or direct HTML generation.

        - Single result (1 row): Stream LLM-generated natural language
        - Multiple results (2+ rows): Generate Python table, send complete (no streaming)

        Args:
            user_query: Original user question
            results: Query results
            metadata: Query metadata (row count, etc.)

        Yields:
            Individual tokens (for 1 result) or complete HTML (for 2+ results)
        """
        rows_returned = metadata['rows_returned']
        max_rows = self.security_config['max_rows']

        # Handle no results
        if rows_returned == 0:
            yield (
                "I couldn't find any results matching your query. "
                "Please check your search terms and try again."
            )
            return

        # For multiple results (2+), generate table directly in Python (no LLM, no streaming)
        # This avoids broken HTML from streaming partial tables
        if rows_returned >= 2:
            logger.info(f"Using Python-based table generation for {rows_returned} rows (complete, no streaming)")
            table_html = self._generate_html_table(results)

            # Add overflow warning if needed
            if metadata.get('truncated', False) and rows_returned >= max_rows:
                table_html += (
                    f"<br><br>Note: This query matched more than {max_rows} results. "
                    f"Only the first {max_rows} are shown. "
                    f"Please contact IT if you need a complete dataset with all results."
                )

            yield table_html
            return

        # For single result (1 row), use LLM streaming for natural language
        # COMBINED PROMPT: Ask for both answer AND follow-up questions in ONE call
        # Build context from results
        result = results[0]
        name = None
        if 'FirstName' in result and 'LastName' in result:
            first = result.get('FirstName', '')
            last = result.get('LastName', '')
            if first or last:
                name = f"{first} {last}".strip()

        title = result.get('BusinessTitle', '')
        dept = result.get('HomeDept', '')

        context_summary = f"Result: {name}"
        if title:
            context_summary += f", {title}"
        if dept:
            context_summary += f" in {dept}"

        prompt = f"""You are formatting database query results and generating follow-up questions.

USER QUESTION: {user_query}

QUERY RESULT (1 row):
{json.dumps(results[0], indent=2)}

TASK 1 - Format the answer:
- Answer directly - NO preambles like "Here is..." or "The answer is..."
- DO NOT repeat or echo the user's question
- Use label:value format with single line breaks (\n)
- Email addresses: Format as <a href="mailto:EMAIL">EMAIL</a>
- Phone numbers: Format as <a href="tel:PHONE">PHONE</a>
- Use FirstName and LastName fields (NEVER UserName)
- NEVER display ID fields (PersonnelId, EmpNo, SupervisorNo)
- Skip null/empty fields
- For "Who is the [TITLE]" questions, use format: "[Name] is the [Title]." followed by contact info

TASK 2 - Generate 2-3 follow-up questions:
- SHORT and SPECIFIC (5-10 words max)
- Directly related to the person/result returned
- Common types: contact info, location, reporting structure, team info

OUTPUT FORMAT (CRITICAL):
First provide the formatted answer, then on a new line put "###FOLLOWUPS###", then list the questions one per line.

Example:
Jane Doe is the VP of Operations.\n\nDepartment: OPS-001\nEmail: <a href="mailto:jane@company.com">jane@company.com</a>\nPhone: <a href="tel:555-1234">555-1234</a>
###FOLLOWUPS###
Who reports to Jane Doe?
What is Jane Doe's full contact information?
List all employees in OPS-001

Now generate the response:"""

        # Calculate dynamic token limit
        num_predict = min(1000, 500 + 150)  # Answer + 3 questions

        logger.info(f"Streaming single result with combined answer+followups generation")

        try:
            # Call Ollama with streaming enabled
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": num_predict
                },
                stream=True,
                keep_alive=-1
            )

            # Stream tokens as they're generated
            full_response = ""
            token_count = 0
            in_followups_section = False
            answer_part = ""

            logger.info("[SQL STREAM] Starting combined LLM generation (answer + followups)...")

            for chunk in response:
                # Access response attribute directly (chunk is GenerateResponse object, not dict)
                if hasattr(chunk, 'response') and chunk.response:
                    token = chunk.response
                    full_response += token
                    token_count += 1

                    # Check if we've hit the followups marker
                    if "###FOLLOWUPS###" in full_response and not in_followups_section:
                        in_followups_section = True
                        # Don't stream the marker or anything after it
                        # Extract just the answer part (before marker)
                        answer_part = full_response.split("###FOLLOWUPS###")[0].strip()
                        logger.info("[SQL STREAM] Detected followups marker, stopping answer stream")
                        continue

                    # Only stream tokens if we haven't hit the followups section yet
                    if not in_followups_section:
                        # Replace newlines with <br> for HTML display
                        display_token = token.replace('\n', '<br>')

                        # DEBUG: Log first 5 tokens and every 50th token to verify streaming
                        if token_count <= 5 or token_count % 50 == 0:
                            logger.info(f"[SQL STREAM] Token #{token_count}: {repr(token[:30])}")

                        # CRITICAL: Yield token and immediately yield control to event loop
                        yield display_token
                        await asyncio.sleep(0)  # Flush to client

            logger.info(f"[SQL STREAM COMPLETE] Generated {token_count} tokens, {len(full_response)} characters")

            # Parse the response to extract answer and followups
            if "###FOLLOWUPS###" in full_response:
                parts = full_response.split("###FOLLOWUPS###")
                final_answer = parts[0].strip()
                followups_text = parts[1].strip() if len(parts) > 1 else ""

                # Parse followup questions (one per line)
                followup_questions = [q.strip() for q in followups_text.split('\n') if q.strip()]
                # Remove any numbering
                followup_questions = [re.sub(r'^[\d\.\-\*\)]+\s*', '', q) for q in followup_questions]
                followup_questions = followup_questions[:3]  # Limit to 3

                logger.info(f"[SQL STREAM] Extracted {len(followup_questions)} follow-up questions from combined response")
            else:
                # If marker not found, use full response as answer, no followups
                logger.warning("[SQL STREAM] Followups marker not found in response")
                final_answer = full_response.strip()
                followup_questions = []

            # Store the followups as a pseudo-attribute for the stream generator to access
            # (We'll handle this in the calling code by checking the generator's final state)
            self._last_streamed_followups = followup_questions

            # Add overflow warning if results were truncated
            if metadata.get('truncated', False) and rows_returned >= max_rows:
                overflow_msg = (
                    f"<br><br>Note: This query matched more than {max_rows} results. "
                    f"Only the first {max_rows} are shown. "
                    f"Please contact IT if you need a complete dataset with all results."
                )
                yield overflow_msg
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"[SQL STREAM ERROR] Error formatting results: {e}", exc_info=True)
            # Fallback to simple formatting (non-streaming)
            yield self._simple_format_results(user_query, results, metadata)
            self._last_streamed_followups = []
            await asyncio.sleep(0)

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

    async def generate_followup_questions(
        self,
        user_query: str,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Generate contextual follow-up questions based on the query and results.

        Args:
            user_query: Original user question
            results: Query results
            metadata: Query metadata (row count, etc.)

        Returns:
            List of 2-3 suggested follow-up questions
        """
        rows_returned = metadata['rows_returned']

        # If no results, suggest alternative queries
        if rows_returned == 0:
            return [
                "Try searching for a different name or title",
                "List all employees in a specific department",
                "Search by location or building"
            ]

        # Build context from results for more relevant suggestions
        result_summary = []
        for i, result in enumerate(results[:3], 1):  # Only use first 3 results for context
            name = None
            if 'FirstName' in result and 'LastName' in result:
                first = result.get('FirstName', '')
                last = result.get('LastName', '')
                if first or last:
                    name = f"{first} {last}".strip()

            title = result.get('BusinessTitle', '')
            dept = result.get('HomeDept', '')

            summary = f"Result {i}:"
            if name:
                summary += f" {name}"
            if title:
                summary += f", {title}"
            if dept:
                summary += f" in {dept}"
            result_summary.append(summary)

        context_text = "\n".join(result_summary) if result_summary else "Employee information"

        prompt = f"""You are helping generate follow-up questions for an employee directory query system.

ORIGINAL QUESTION: {user_query}

RESULTS SUMMARY:
{context_text}

Generate 2-3 natural, conversational follow-up questions that a user might want to ask based on these results.

RULES:
1. Questions should be SHORT and SPECIFIC (5-10 words max)
2. Questions should be directly related to the people/results returned
3. Use natural language (no overly formal phrasing)
4. Common follow-up types:
   - Contact information: "What is [name]'s email?" or "What is [name]'s phone number?"
   - Location: "Where does [name] sit?" or "What building is [name] in?"
   - Reporting structure: "Who does [name] report to?" or "Who reports to [name]?"
   - Team information: "Who else is in [department]?" or "List [name]'s direct reports"
   - Role details: "What is [name]'s full title?" or "When did [name] start?"
5. If the result shows a title like "Director" or "Manager", suggest questions about their team or department
6. Return ONLY the questions, one per line, no numbering, no explanations

EXAMPLES:

Original: "Find John Smith"
Result: John Smith, Senior Engineer in ENGR-001
Follow-ups:
What is John Smith's email address?
Who does John Smith report to?
Who else is in ENGR-001?

Original: "Who is the VP of Operations?"
Result: Jane Doe, VP of Operations in OPS-001
Follow-ups:
Who reports to Jane Doe?
What is Jane Doe's contact information?
List all employees in OPS-001

Original: "List employees in Engineering"
Results: 15 employees
Follow-ups:
Show managers in Engineering
Who is the director of Engineering?
Filter by seniority level

Now generate follow-up questions:"""

        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.7,  # Higher temperature for creative question generation
                    "num_predict": 150
                },
                keep_alive=-1
            )

            # Parse response - one question per line
            questions_text = response['response'].strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]

            # Remove any numbering (1., 2., -, *)
            questions = [re.sub(r'^[\d\.\-\*\)]+\s*', '', q) for q in questions]

            # Limit to 3 questions
            questions = questions[:3]

            logger.info(f"Generated {len(questions)} follow-up questions: {questions}")
            return questions

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}", exc_info=True)
            # Return generic follow-ups as fallback
            return [
                "View contact information",
                "Find more employees in this department",
                "Check reporting structure"
            ]


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
