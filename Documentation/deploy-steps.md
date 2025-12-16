## 🚀 Deployment Steps: From Code to Cloud

The deployment process transitions the application from a local repository to a running, monitored service on **Amazon Lightsail**, leveraging **GitHub Actions** for automation.

### 1. 🌐 Initial Server & Security Setup (Lightsail)

This phase establishes the base environment and secures remote access.

* **Secure Connection:** Establish an initial secure **SSH connection** to the Lightsail instance (`ubuntu@Public_IP`).
* **Update & Monitor:** Update system packages and install a monitoring utility like **`htop`** to check server resources.
* **Firewall Configuration:** Configure the AWS Lightsail firewall to **open the application port** (e.g., `3000`) to allow external access.

---

### 2. 🔑 User & Key Configuration for Automation (CI/CD)

A critical security practice: creating a dedicated non-root user for automated deployments.

* **Dedicated User:** Create a non-root user (**`github_user`**) and grant it `sudo` privileges.
* **GitHub Actions Key:**
    * Generate an **SSH Key Pair** for **`github_user`**.
    * Append the **Public Key** to the server's `authorized_keys`.
    * Securely retrieve the **Private Key** and store it as a **GitHub Secret** (`SSH_MAIN_PRIVATE_KEY`).
* **Deploy Key:** Generate a separate **Deploy Key** (Ed25519) and add the **Public Key** to the **GitHub Repository Deploy Keys** for read-only access, allowing the server to clone the code.

---

### 3. 📦 Application Cloning & Docker Installation

Setting up the code and its runtime environment on the server.

* **Clone Repository:** As **`github_user`**, clone the application repository from GitHub using the secure SSH URL into a designated workspace.
* **Configuration:** Use **VS Code Insiders CLI Tunnel** to securely and remotely edit configuration files (like **`.env`**) on the server.
* **Install Docker:** Install the **Docker Engine** and **Docker Compose plugin** using the official APT repository method.

---

### 4. ⚙️ Service Management (Systemd)

Ensuring the application is resilient and starts automatically.

* **Create Service File:** Define a **`systemd` service** (**`myrag.service`**) in `/etc/systemd/system`.
    * This file specifies how Docker Compose should be run, ensuring the application starts under the **`github_user`** and requires the Docker service.
* **Enable Service:** Use `systemctl` to **load**, **start**, and **enable** the service to ensure the application automatically launches on server reboot.
* **Initial Start:** Execute the **`docker compose up --build -d`** command via the service or manually to build and start the containers in detached mode.

---

### 5. 🤖 Continuous Deployment (GitHub Actions)

Finalizing the automation pipeline for production updates.

* **`NOPASSWD` Config:** Configure `sudoers` to allow **`github_user`** to run **only** the `systemctl restart myrag.service` command without a password. This is essential for non-interactive automation.
* **GitHub Secrets:** Store the server's **Public IP** (`SSH_MAIN_HOST_IP`) and the **Private SSH Key** (`SSH_MAIN_PRIVATE_KEY`) as GitHub Secrets.
* **Automated Workflow:** The GitHub Action workflow uses these secrets to securely connect to the server, pull the latest code, and restart the `systemd` service, completing the CI/CD pipeline. 

***

This summary provides a clean, 5-step flow for your presentation, focusing on the *why* (security, automation, resilience) behind your technical choices.