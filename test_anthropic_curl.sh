#!/bin/bash

# Test script for Anthropic API using curl
# Usage: ./test_anthropic_curl.sh

echo "Testing Anthropic API with Claude 3.5 Sonnet..."
echo "Please enter your Anthropic API key:"
read -s API_KEY

echo "Making curl request..."

curl -X POST https://api.anthropic.com/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-7-sonnet-20250219",
    "max_tokens": 100,
    "messages": [
      {
        "role": "user",
        "content": "Hello! Please respond with \"API connection successful\" to confirm the request is working."
      }
    ]
  }' | python3 -m json.tool

echo -e "\n\nIf you see a JSON response with 'API connection successful', your API key is working!"
