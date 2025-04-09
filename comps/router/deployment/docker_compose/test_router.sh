#!/bin/bash
set -e

divider="====================================="
subdivider="-------------------------------------"

echo ""
echo "$divider"
echo "   🧪 Testing OPEA Router Microservice"
echo "$divider"

print_section() {
  echo ""
  echo "$subdivider"
  echo "[INFO] $1"
  echo "$subdivider"
}

# --------------------- Test Query 1 ---------------------
print_section "Test Query #1 (expecting weak inference route)"
echo "[Query]: What is the square root of 16?"
echo ""

RESPONSE1=$(curl -s --noproxy localhost,127.0.0.1 -X POST http://localhost:6000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the square root of 16?"}')

echo "[Response]:"
echo "$RESPONSE1"

sleep 1

# --------------------- Test Query 2 ---------------------
print_section "Test Query #2 (expecting strong inference route)"
echo "[Query]: Given a 100x100 grid where each cell is independently colored..."
echo ""

RESPONSE2=$(curl -s --noproxy localhost,127.0.0.1 -X POST http://localhost:6000/v1/route \
  -H "Content-Type: application/json" \
  -d '{"text": "Given a 100x100 grid where each cell is independently colored black or white such that for every cell the sum of black cells in its row, column, and both main diagonals is a distinct prime number, determine whether there exists a unique configuration of the grid that satisfies this condition and, if so, compute the total number of black cells in that configuration."}')

echo "[Response]:"
echo "$RESPONSE2"

echo ""
echo "$divider"
echo "✅ Test Completed."
echo "$divider"
