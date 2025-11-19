#!/bin/bash
# Test Ollama health check endpoints

echo "Testing Ollama health check endpoints..."
echo ""

# Test GPU 0
echo "=== GPU 0 (port 11434) ==="
echo ""

echo "1. Testing /api/tags (current health check):"
curl -f http://localhost:11434/api/tags 2>&1 | head -10
echo ""
echo "Exit code: $?"
echo ""

echo "2. Testing root endpoint /:"
curl -f http://localhost:11434/ 2>&1 | head -10
echo ""
echo "Exit code: $?"
echo ""

echo "3. Testing /api/version:"
curl -f http://localhost:11434/api/version 2>&1 | head -10
echo ""
echo "Exit code: $?"
echo ""

# Test GPU 1
echo "=== GPU 1 (port 11435) ==="
echo ""

echo "1. Testing /api/tags:"
curl -f http://localhost:11435/api/tags 2>&1 | head -10
echo ""
echo "Exit code: $?"
echo ""

echo "2. Testing root endpoint /:"
curl -f http://localhost:11435/ 2>&1 | head -10
echo ""
echo "Exit code: $?"
echo ""
