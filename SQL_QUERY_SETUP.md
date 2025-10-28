# SQL Query System Setup Guide

This guide explains how to set up and use the SQL query functionality in Adam.

## Overview

The SQL query system allows users to query MS SQL Server databases using natural language. The system:
- Converts natural language to SQL using LLM
- Validates queries for security
- Executes queries safely (read-only)
- Formats results as natural language
- Supports conversational follow-ups

## Prerequisites

### 1. Install Python Dependencies

```bash
pip install pyodbc>=4.0.39
```

### 2. Install ODBC Driver for SQL Server

**Windows:**
- Download and install [Microsoft ODBC Driver 17 for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)

**Linux (RHEL/CentOS):**
```bash
curl https://packages.microsoft.com/config/rhel/9/prod.repo > /etc/yum.repos.d/mssql-release.repo
yum remove unixODBC-utf16 unixODBC-utf16-devel
ACCEPT_EULA=Y yum install -y msodbcsql17
yum install -y unixODBC-devel
```

**Linux (Ubuntu/Debian):**
```bash
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql17
apt-get install -y unixodbc-dev
```

### 3. Create Read-Only Database User

**For Development (SQL Authentication):**
```sql
USE Corporate;
GO

-- Create login
CREATE LOGIN [svcasm-adamAi] WITH PASSWORD = 'YourSecurePassword123!';

-- Create user in Corporate database
CREATE USER [svcasm-adamAi] FOR LOGIN [svcasm-adamAi];

-- Grant SELECT permission ONLY on the view
GRANT SELECT ON dbo.vwPersonnelAll TO [svcasm-adamAi];

-- Verify permissions
SELECT * FROM sys.database_permissions WHERE grantee_principal_id = USER_ID('svcasm-adamAi');
```

**For Production (Windows Authentication):**
```sql
USE Corporate;
GO

-- Create user from AD account
CREATE USER [DOMAIN\svcasm-adamAi] FOR LOGIN [DOMAIN\svcasm-adamAi];

-- Grant SELECT permission ONLY on the view
GRANT SELECT ON dbo.vwPersonnelAll TO [DOMAIN\svcasm-adamAi];
```

### 4. Set Environment Variables

**Development (SQL Authentication):**
```bash
export EMPLOYEE_DB_PASSWORD='YourSecurePassword123!'
```

**Production (Windows Authentication):**
Edit `config/databases.yaml` and set:
```yaml
employee_directory:
  connection:
    use_windows_auth: true
```

## Configuration

### Database Configuration

The database configuration is in `config/databases.yaml`:

```yaml
employee_directory:
  enabled: true
  connection:
    server: "localhost"        # Change to your SQL Server hostname
    port: 1433
    database: "Corporate"
    user: "svcasm-adamAi"
    password_env: "EMPLOYEE_DB_PASSWORD"
    use_windows_auth: false    # Set to true for Windows Auth in production

  security:
    max_rows: 1000             # Maximum rows to return
    query_timeout_seconds: 30  # Query timeout
    allowed_tables: ["vwPersonnelAll"]
```

### Modify Connection Settings

If your SQL Server is on a different host or port, edit `config/databases.yaml`:

```yaml
employee_directory:
  connection:
    server: "sql-server.company.com"  # Your SQL Server
    port: 1433
    database: "Corporate"
```

## Testing

### 1. Start the API Server

```bash
python airgapped_rag_advanced.py
```

Or with uvicorn:
```bash
uvicorn airgapped_rag_advanced:app --host 0.0.0.0 --port 8000
```

### 2. Test the Employee Directory Endpoint

**Simple search:**
```bash
curl -X POST http://localhost:8000/query-employee \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Find John Smith"}'
```

**Department search:**
```bash
curl -X POST http://localhost:8000/query-employee \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "List all employees in Engineering"}'
```

**Hire date search:**
```bash
curl -X POST http://localhost:8000/query-employee \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Who was hired in 2024?"}'
```

**Follow-up question:**
```bash
# First query
curl -X POST http://localhost:8000/query-employee \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Find John Smith"}' | jq -r '.conversation_id' > conv_id.txt

# Follow-up (uses conversation context)
curl -X POST http://localhost:8000/query-employee \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\": \"What's his email?\", \"conversation_id\": \"$(cat conv_id.txt)\"}"
```

### 3. Check Logs

The system logs all SQL queries for auditing:

```bash
# Look for these log entries:
# - "Generated SQL: SELECT ..."
# - "Query returned X rows"
# - "SQL validation failed: ..." (if query is unsafe)
```

## API Response Format

```json
{
  "answer": "I found John Smith in the employee directory:\n\nName: John Smith\nEmail: john.smith@amentum.com\nDepartment: Engineering\nHire Date: 2020-01-15",
  "rows_returned": 1,
  "total_rows_available": 1,
  "overflow_warning": null,
  "conversation_id": "abc123-def456",
  "conversation_warning": null,
  "query_type": "employee_directory",
  "execution_time_ms": null
}
```

**Fields:**
- `answer`: Natural language formatted result
- `rows_returned`: Number of rows in result
- `overflow_warning`: Warning if >1000 rows (truncated)
- `conversation_id`: ID for follow-up questions
- `query_type`: Always "employee_directory" for this endpoint

## Security Features

### 1. Read-Only Queries
- Only SELECT statements allowed
- DROP, DELETE, UPDATE, INSERT, etc. are blocked

### 2. Row Limits
- Maximum 1000 rows returned
- Users notified if more data exists

### 3. Whitelisted Tables
- Only `vwPersonnelAll` can be queried
- Attempts to query other tables are blocked

### 4. Excluded by Default
- Terminated employees (IsTerminated = 1) excluded unless specifically requested
- Sensitive columns (AnnualRate) not included in results

### 5. Query Timeout
- 30 second timeout prevents long-running queries

### 6. SQL Injection Prevention
- Parameterized queries
- Keyword blacklist
- SQL validation before execution

## Common Query Examples

### Find Employee
- "Find John Smith"
- "Show me Jane Doe's information"
- "Search for employees named Michael"

### Department Queries
- "List all employees in Engineering"
- "Who works in the HR department?"
- "Show me Finance team members"

### Title/Role Queries
- "Find all managers"
- "List employees with 'Director' in their title"
- "Show me all engineers"

### Date-Based Queries
- "Who was hired in 2024?"
- "Show employees hired after January 1, 2023"
- "List recent hires"

### Location Queries
- "Who works in building 101?"
- "Show employees in room 205"
- "List on-site employees"

### Follow-Up Questions
After any query, you can ask follow-ups:
- "What's his email?"
- "Show me his phone number"
- "What department is she in?"

## Troubleshooting

### Error: "Unable to connect to the employee directory"

**Check:**
1. SQL Server is running and accessible
2. Firewall allows connections on port 1433
3. Database name is correct in `config/databases.yaml`
4. Credentials are correct

**Test connection manually:**
```bash
sqlcmd -S localhost -d Corporate -U svcasm-adamAi -P 'YourPassword'
```

### Error: "ODBC Driver not found"

Install the ODBC driver (see Prerequisites above).

**Check installed drivers:**
```bash
# Linux
odbcinst -q -d

# Windows
# Check "ODBC Data Sources" in Control Panel
```

### Error: "Password not set"

Set the environment variable:
```bash
export EMPLOYEE_DB_PASSWORD='YourPassword'
```

Verify:
```bash
echo $EMPLOYEE_DB_PASSWORD
```

### Error: "Query must include TOP clause"

The LLM failed to generate a proper query. This is usually fixed by:
1. Rephrasing the question
2. Being more specific
3. Checking LLM model is running (Ollama)

### No Results Found

**Check:**
1. Employee name spelling
2. Department name (may need wildcards)
3. Employee is not terminated (IsTerminated = 0)

## UI Integration

### Dropdown/Button Options

In your UI, provide these options:
- **"Policy & Procedures"** → POST /query (existing document RAG)
- **"Employee Directory"** → POST /query-employee (new)
- **"Purchasing System"** → POST /query-purchasing (placeholder)

### Example UI Code (JavaScript)

```javascript
async function queryAdam(prompt, systemType, conversationId = null) {
  const endpoints = {
    'policies': '/query',
    'employees': '/query-employee',
    'purchasing': '/query-purchasing'
  };

  const response = await fetch(endpoints[systemType], {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: prompt,
      conversation_id: conversationId
    })
  });

  return await response.json();
}

// Usage
const result = await queryAdam("Find John Smith", "employees");
console.log(result.answer);
```

## Production Deployment

### 1. Switch to Windows Authentication

Edit `config/databases.yaml`:
```yaml
employee_directory:
  connection:
    use_windows_auth: true
```

### 2. Remove Password Environment Variable

No longer need `EMPLOYEE_DB_PASSWORD` with Windows auth.

### 3. Verify Service Account Permissions

```sql
-- Verify the service account has correct permissions
SELECT
    dp.name AS principal_name,
    dp.type_desc,
    o.name AS object_name,
    p.permission_name,
    p.state_desc
FROM sys.database_permissions p
JOIN sys.database_principals dp ON p.grantee_principal_id = dp.principal_id
JOIN sys.objects o ON p.major_id = o.object_id
WHERE dp.name = 'DOMAIN\svcasm-adamAi';
```

### 4. Audit Logging

All SQL queries are logged. Monitor logs for:
- Suspicious query patterns
- Failed validation attempts
- High-frequency queries

### 5. Performance Monitoring

- Monitor query execution times
- Set up alerts for timeouts
- Review queries that return max rows (1000)

## Adding Additional Databases (Purchasing, etc.)

To add a new database:

1. **Get the schema** (CREATE TABLE statements)

2. **Update config/databases.yaml:**
```yaml
purchasing_system:
  enabled: true
  connection:
    server: "localhost"
    database: "PurchasingDB"
    # ... same structure as employee_directory
  schema:
    purchase_orders:
      # ... add columns and examples
```

3. **Update sql_routes.py:**
```python
@sql_router.post("/query-purchasing")
async def query_purchasing_system(request: SQLQueryRequest):
    sql_handler = get_sql_query_handler(
        database_name="purchasing_system",  # matches config key
        ollama_host=OLLAMA_HOST,
        model_name=LLM_MODEL
    )
    # ... same logic as employee directory
```

4. **Test the endpoint**

---

## Support

For issues or questions:
1. Check logs: `/data/airgapped_rag/logs/`
2. Review this documentation
3. Contact IT support
