# Improvements Needed for Better Retrieval

## Current Issues (Based on Test Results)

### Test Case:
- **Query**: "What is the company policy for telecommuting"
- **Expected**: EN-PO-0071 "Telecommuting Policy"
- **Got**: CR-PO-0301 "Management System Policy" (WRONG!)
- **Citations**: Boilerplate header text (not helpful)

## Root Causes

### 1. Topic Generation Too Generic
```
Current topic: "Management System Policy"
Problem: Matches ANY policy question
Needed: "Telecommuting Policy EN-PO-0071 - Amentum Space Mission Division"
```

### 2. Single Embedding Per Document
```
Current: One embedding for entire document topic
Problem: Not granular enough for specific queries
Needed: Multiple embeddings or hybrid search
```

### 3. Poor Metadata Extraction
```
Current: Only stores topic and source_url
Problem: Missing document number, title, sections
Needed: Extract and index document metadata
```

### 4. Bad Citation Extraction
```
Current: "This document contains proprietary information..."
Problem: Extracting header boilerplate
Needed: Extract relevant paragraphs from sections
```

## Proposed Solutions

### Solution 1: Enhanced Metadata Extraction

Extract from PDFs:
- Document number (EN-PO-0071)
- Document title (Telecommuting Policy)
- Section headings
- Key terms

Store in ChromaDB metadata for filtering.

### Solution 2: Improved Topic Generation

Instead of:
```
"Management System Policy"
```

Generate:
```
"Telecommuting Policy EN-PO-0071: Policy for remote work arrangements including hybrid teleworker, remote teleworker, eligibility criteria, approval process, equipment, security requirements"
```

### Solution 3: Hybrid Retrieval

Combine:
1. **Semantic search** (embeddings) - for conceptual matching
2. **Keyword search** (BM25) - for document numbers, specific terms
3. **Metadata filtering** - for document type, section

### Solution 4: Better Citation Extraction

Extract:
- Section 5.0 (main policy sections)
- Paragraphs containing key terms from query
- Not header/footer boilerplate

### Solution 5: Section-Level Indexing

Instead of one embedding per document:
- One embedding for document overview
- Additional embeddings for major sections
- Link sections back to parent document

## Implementation Options

### Option A: Quick Fix (Recommended for Now)

1. ✅ Improve topic generation prompt to be more specific
2. ✅ Extract document number/title from first page
3. ✅ Add document metadata to ChromaDB
4. ✅ Improve citation extraction (skip headers, get sections)
5. ✅ Add keyword matching for document numbers

**Time**: 1-2 hours
**Impact**: Significant improvement

### Option B: Hybrid Retrieval (Better Long-Term)

1. Add BM25 keyword search alongside semantic search
2. Combine scores with weighted fusion
3. Filter by metadata (document type, date)

**Time**: 3-4 hours
**Impact**: Best retrieval quality

### Option C: Section-Level Chunking (Most Complex)

1. Parse document into sections
2. Create embeddings for each section
3. Retrieve sections, then full documents
4. More complex but most accurate

**Time**: 6-8 hours
**Impact**: Best for very long documents

## Recommended Approach

Start with **Option A** (Quick Fix):

1. **Parse document metadata** from first page:
   ```python
   # Extract: "EN-PO-0071" and "Telecommuting Policy"
   doc_number = extract_doc_number(markdown)
   doc_title = extract_title(markdown)
   ```

2. **Enhance topic generation**:
   ```python
   prompt = f"""Extract key information from this document:
   - Document number
   - Main title
   - Key topics (3-5 key terms)

   Create a searchable description with format:
   "[Title] [Document Number]: [Key Topics]"
   """
   ```

3. **Store metadata in ChromaDB**:
   ```python
   metadata = {
       'source_url': source_url,
       'topic': topic,
       'doc_number': doc_number,  # NEW
       'doc_title': doc_title,    # NEW
       'filename': filename        # NEW
   }
   ```

4. **Improve search** with metadata filtering:
   ```python
   # If query contains document number, filter by it
   if doc_number_pattern.match(query):
       where_filter = {"doc_number": {"$eq": extracted_number}}
   ```

5. **Better citation extraction**:
   ```python
   # Skip first 500 chars (headers)
   # Find paragraphs containing query terms
   # Extract 2-3 relevant paragraphs
   ```

## Testing Plan

After improvements, test with:

```json
{
  "prompt": "What is the company policy for telecommuting",
  "top_k": 3
}
```

**Expected**:
- ✅ EN-PO-0071 "Telecommuting Policy" as #1 result
- ✅ Citations from Section 5.0 (actual policy content)
- ✅ Accurate answer about eligibility, approval, types

## Next Steps

1. Read user feedback on approach
2. Implement Option A (Quick Fix)
3. Test with user's 3 documents
4. If still not good enough, implement Option B

---

**Current Priority**: Quick Fix (Option A)
**Estimated Time**: 2 hours
**Expected Improvement**: 70-80% better retrieval accuracy
