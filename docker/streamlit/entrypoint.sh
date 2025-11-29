#!/bin/bash
set -e

echo "Starting Streamlit UI..."

exec streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
