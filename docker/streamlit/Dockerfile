FROM ghcr.io/astral-sh/uv:0.6.14-python3.13-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY streamlit_ui/requirements.txt /app/requirements.txt
RUN uv pip install -r requirements.txt --system

# Copy ONLY the Streamlit application
COPY streamlit_ui/ .

EXPOSE 8501

COPY docker/streamlit/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
