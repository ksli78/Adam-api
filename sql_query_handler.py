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
            "6. NEVER select ID fields (PersonnelId, EmpNo, SupervisorNo) unless specifically needed for joins or context tracking",
            "7. Avoid selecting sensitive columns (AnnualRate) unless the question specifically asks for it",
            "8. Use proper date formatting for MS SQL Server (e.g., YEAR(HireDate) = 2024)",
            "9. Return ONLY the SQL query on a SINGLE LINE - no explanations, no markdown, no quotes, no line breaks in the SQL",
            "10. Format: SELECT TOP 1000 columns FROM table WHERE conditions",
            "\nCOMMON QUERY PATTERNS:",
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

            # Convert markdown formatting to HTML (bold and lists) BEFORE table processing
            # This ensures consistent formatting across all responses
            answer = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', answer)

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
                        # Convert markdown lists to HTML lists
                        lines = parts[i].split('\n')
                        processed = []
                        in_list = False
                        for line in lines:
                            if line.strip().startswith('- '):
                                if not in_list:
                                    processed.append('<ul>')
                                    in_list = True
                                processed.append(f'<li>{line.strip()[2:]}</li>')
                            else:
                                if in_list:
                                    processed.append('</ul>')
                                    in_list = False
                                processed.append(line)
                        if in_list:
                            processed.append('</ul>')
                        parts[i] = '<br>'.join(processed)
                answer = ''.join(parts)
            else:
                # No table: Convert markdown lists and newlines to HTML
                lines = answer.split('\n')
                processed = []
                in_list = False
                for line in lines:
                    if line.strip().startswith('- '):
                        if not in_list:
                            processed.append('<ul>')
                            in_list = True
                        processed.append(f'<li>{line.strip()[2:]}</li>')
                    else:
                        if in_list:
                            processed.append('</ul>')
                            in_list = False
                        if line.strip() or not processed:
                            processed.append(line)
                        else:
                            processed.append('<br>')
                if in_list:
                    processed.append('</ul>')
                answer = '<br>'.join(processed)

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
