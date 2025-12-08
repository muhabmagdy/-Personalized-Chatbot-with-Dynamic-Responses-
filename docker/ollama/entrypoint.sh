#!/bin/bash

# Start the Ollama server in the background
echo "Starting Ollama server..."
ollama serve &
SERVER_PID=$!

# Wait for the Ollama server to be ready using the Ollama executable
echo "Waiting for Ollama API to be active..."
until ollama list 2>/dev/null; do
  sleep 1
done

echo "Ollama server is active. Starting model pull..."

# Pull the required model
ollama pull ollama/phi3:mini-instruct-q4_K_M

if [ $? -eq 0 ]; then
    echo "Phi-3 model pulled successfully."
else
    echo "ERROR: Failed to pull Phi-3 model."
    exit 1
fi

# Wait for the background ollama serve process to keep the container running
wait $SERVER_PID
