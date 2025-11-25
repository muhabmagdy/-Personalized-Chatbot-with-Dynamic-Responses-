FROM ghcr.io/astral-sh/uv:0.6.14-python3.13-bookworm

WORKDIR /app

# Install additional system dependencies for lxml and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libavif-dev pkg-config \
    libjpeg-dev \
    gcc unzip zip \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt .

RUN uv pip install -r requirements.txt --system

COPY src/ .

# Create directory structure for Alembic
RUN mkdir -p /app/models/db_schemes/myrag/

COPY docker/myrag/alembic.ini /app/models/db_schemes/myrag/alembic.ini

## ==> Can't run alembic commands here because the DB service (pgveector) is not up yet
## ==> entrypoint.sh will contain commands to run after build the Docker image (not during build)

COPY docker/myrag/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Using Entrypoint to run migrations before starting the app
# RUN not used here because we need to run migrations when container starts
#RUN will run during build time
ENTRYPOINT ["/entrypoint.sh"]

# Command to run the application
# --reload is not used in production, it used only in development for hot-reloading after code changes
# You can adjust the number of workers based on your server's CPU cores
# worker count is set to 4 as a reasonable default
# worker usage improves handling multiple requests concurrently
# --reload will disable multiple workers, so not used here
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]