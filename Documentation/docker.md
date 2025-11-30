## 💻 Project Documentation: Containerization and Orchestration

This document details the configuration for containerizing and orchestrating the project services using **Docker** and **Docker Compose**.

### 1\. Folder Structure 🗃️

The project utilizes the following structure for deployment files:

```
#project/
├── src/                  # Source code for the backend (e.g., FastAPI application)
├── streamlit_ui/         # Source code for the frontend (Streamlit application)
└── docker/               # Containerization and Orchestration files
    ├── docker-compose.yml
    ├── myrag/            # Files for the FastAPI/RAG service container
    │   ├── Dockerfile
    │   ├── entrypoint.sh
    │   └── alembic.ini
    ├── streamlit/        # Files for the Streamlit UI service container
    │   ├── Dockerfile
    │   └── entrypoint.sh
    ├── prometheus/       # Configuration for the Prometheus monitoring service
    │   └── prometheus.yml
    ├── nginx/            # Configuration for the Nginx reverse proxy
    │   └── default.conf  # (Configuration file is empty, but volume mapping exists)
    └── env/              # Environment variables for all services
        ├── .env.app
        ├── .env.grafana
        ├── .env.postgres
        ├── .env.postgres-exporter
        └── .env.streamlit.app
```

-----

### 2\. Docker Compose File (`docker/docker-compose.yml`) ⚙️

This file orchestrates a complete MLOps/RAG stack consisting of an application backend, a frontend, a database, a reverse proxy, and a comprehensive monitoring system.

#### Services Overview

| Service Name | Role | Ports | Dependencies | Key Features |
| :--- | :--- | :--- | :--- | :--- |
| **`streamlit`** | **Frontend UI** (Streamlit) | $8501:8501$ | `fastapi` | Serves the user interface. Depends on FastAPI for data/RAG operations. Uses a volume map for live code updates. |
| **`fastapi`** | **Backend API** (RAG/Core Logic) | $8000:8000$ | `pgvector` | Serves the core ML/RAG logic. Uses a **healthcheck** on `pgvector`. Runs Alembic migrations via `entrypoint.sh`. |
| **`nginx`** | **Reverse Proxy** | $80:80$ | `fastapi` | Routes external traffic, typically to the frontend (Streamlit) or backend (FastAPI). |
| **`pgvector`** | **Vector/SQL DB** (PostgreSQL) | $5432:5432$ | - | Stores RAG metadata and vector embeddings. Includes a **healthcheck** to ensure database readiness. |
| **`prometheus`** | **Monitoring** | $9090:9090$ | - | Time-series database for collecting metrics from FastAPI, Postgres, and the host system. |
| **`grafana`** | **Dashboard** | $3000:3000$ | `prometheus` | Visualizes the metrics collected by Prometheus. |
| **`node-exporter`** | **Host Metrics** | $9100:9100$ | - | Exposes system-level metrics (CPU, RAM, Disk) for the host machine. |
| **`postgres-exporter`** | **DB Metrics** | $9187:9187$ | `pgvector` | Exposes PostgreSQL-specific performance metrics. |

#### Networks and Volumes

  * **Network:** All services connect to a single **`backend`** bridge network, allowing them to communicate using their service names (e.g., `streamlit` talks to `fastapi`).
  * **Volumes:**
      * **Named Volumes:** `fastapi_data`, `pgvector`, `prometheus_data`, `grafana_data` are used for **persisting data** outside the container lifecycle (e.g., database files, monitoring history).
      * **Bind Mounts:** Used for development, such as mapping `./streamlit_ui` to `/app` in the `streamlit` container.

-----

### 3\. FastAPI/RAG Service Files (`docker/myrag/`) 📦

This directory contains the necessary files to build and run the main backend service, responsible for the RAG logic.

#### A. Dockerfile (`docker/myrag/Dockerfile`)

  * **Base Image:** Uses `ghcr.io/astral-sh/uv:0.6.14-python3.13-bookworm`, a lightweight base image from **uv**, an efficient package installer and resolver.
  * **Dependencies:** Installs essential system packages (`build-essential`, `libxml2-dev`, etc.) required for various Python libraries (e.g., `lxml`).
  * **Dependencies Installation:** Copies `src/requirements.txt` and uses `uv pip install -r requirements.txt --system` for fast, reproducible Python dependency installation.
  * **Alembic Setup:** Copies `alembic.ini` and creates the necessary directory structure (`/app/models/db_schemes/myrag/`) for database migration tools.
  * **Execution Flow:** The **`ENTRYPOINT`** is set to `/entrypoint.sh` to run setup tasks **before** the application starts. The **`CMD`** specifies the final application command: `uvicorn main:app ...` with **1 worker** for stable serving.

#### B. Entrypoint Script (`docker/myrag/entrypoint.sh`)

This script ensures that the database is ready and correctly migrated before the application attempts to connect.

1.  **Sets `set -e`:** Ensures the script exits immediately if any command fails.
2.  **Runs Migrations:**
      * Navigates to the Alembic configuration directory (`/app/models/db_schemes/myrag/`).
      * Executes **`alembic upgrade head`** to apply all pending database migrations.
3.  **Starts Server:** Uses **`exec "$@"`** to replace the shell process with the `CMD` command from the Dockerfile (Uvicorn), which is the final running application.

#### C. Alembic Configuration (`docker/myrag/alembic.ini`)

This configuration file tells Alembic (the database migration tool) how to connect to the database.

  * **`script_location`:** Points to the directory where migration scripts reside.
  * **`prepend_sys_path`:** Ensures the current working directory is included in Python's path.
  * **`sqlalchemy.url`:** This is the critical line. It is configured to connect to the PostgreSQL service named **`pgvector`** within the Docker network:
    ```ini
    sqlalchemy.url = postgresql://postgres:postgres_passwrod@pgvector:5432/myrag
    ```
    This shows a clear connection from the FastAPI service to the database service using the **service name** as the hostname.

-----

### 4\. Streamlit UI Service Files (`docker/streamlit/`) 🎨

This directory contains the files to build and run the Streamlit frontend.

#### A. Dockerfile (`docker/streamlit/Dockerfile`)

  * **Base Image:** Also uses the `uv` base image for efficiency.
  * **Dependencies:** Installs necessary system packages and then uses `uv pip install` to install Python dependencies from `streamlit_ui/requirements.txt`.
  * **Execution Flow:** Sets the **`ENTRYPOINT`** to `/entrypoint.sh` to control the application startup. Exposes port $8501$.

#### B. Entrypoint Script (`docker/streamlit/entrypoint.sh`)

This script runs the Streamlit application with appropriate production configuration options.

1.  **Starts Streamlit:** Executes **`streamlit run app.py`**.
2.  **Configuration:** Uses command-line arguments to set the server address to **`0.0.0.0`** (to be accessible from outside the container) and disables non-essential browser features like CORS, XSRF protection, and usage stats for a cleaner production environment.

-----

### 5\. Monitoring Configuration (`docker/prometheus/prometheus.yml`) 📊

This file configures Prometheus to scrape metrics from the deployed services.

  * **`global`:** Sets the default scrape interval to **15 seconds**.
  * **`scrape_configs`:** Defines targets for metrics collection:
      * **`fastapi`:** Scrapes metrics from the FastAPI service at `fastapi:8000`. Note the use of a non-standard `metrics_path: '/TrhBVe_m5gg2002_E5VVqS'`, which is a good security practice to obscure the metrics endpoint.
      * **`node-exporter`:** Scrapes system metrics from `node-exporter:9100`.
      * **`prometheus`:** Scrapes Prometheus's own internal metrics (self-monitoring).
      * **`postgres`:** Scrapes PostgreSQL metrics from the `postgres-exporter` service at `postgres-exporter:9187`.

-----

### 6\. Environment Variables (`docker/env/`) 🔑

The `env` folder contains separate `.env` files for each service, promoting the **separation of concerns** and centralizing configuration secrets/variables.

  * **`.env.app`:** For the **FastAPI** service (e.g., database connection credentials, API keys).
  * **`.env.grafana`:** For the **Grafana** service (e.g., admin password, data source settings).
  * **`.env.postgres`:** For the **`pgvector`** database (e.g., `POSTGRES_USER`, `POSTGRES_PASSWORD`).
  * **`.env.postgres-exporter`:** For the **Postgres Exporter** (e.g., the specific connection string it uses to reach `pgvector`).
  * **`.env.streamlit.app`:** For the **Streamlit** UI (e.g., the URL of the FastAPI backend).

This setup is highly professional and follows best practices for a containerized MLOps stack, including health checks, database migrations, and comprehensive monitoring.