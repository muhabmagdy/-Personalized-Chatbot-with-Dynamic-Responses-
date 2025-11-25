#!/bin/bash
set -e

echo "Running database migrations..."
cd /app/models/db_schemes/myrag/
alembic upgrade head
cd /app

echo "Starting FastAPI server..."
# Use exec "$@" to run CMD from Dockerfile (uvicorn)
exec "$@"