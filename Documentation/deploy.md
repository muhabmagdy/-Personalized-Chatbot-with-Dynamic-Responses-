# 🚀 Deployment Guide for Personalized Chatbot on AWS Lightsail

This document outlines the steps for setting up a production environment on an **AWS Lightsail** instance (Ubuntu) to deploy the application, configure secure access, and automate updates using **GitHub Actions** and **Docker Compose**.

## 1\. Initial Server Connection and Setup

### 1.1 Secure SSH Connection

The first step is establishing a secure connection to your server using the private key associated with your Lightsail instance.

| Command | Description |
| :--- | :--- |
| `ssh -i rag-app-aws-key.pem ubuntu@Public_IP` | Connects to the server using the `ubuntu` default user and your private key (`rag-app-aws-key.pem`). **`Public_IP`** is your Lightsail instance's public IP address. |
| **Action** | Select **`yes`** when prompted to confirm the connection. |

### 1.2 System Update and Monitoring Tool

Update the server's package lists and install a utility for real-time system monitoring.

| Command | Description |
| :--- | :--- |
| `sudo apt update` | Updates the local package index. |
| `sudo apt install htop` | Installs **`htop`**, an interactive process viewer. |
| `htop` | Runs `htop` to check server resources (CPU, Memory). Press **F10** to quit. |

-----

## 2\. GitHub Actions User and Key Configuration

For security and automation, a dedicated non-root user (`github_user`) is created for GitHub Actions to manage deployments. This user will use key-based authentication.

### 2.1 Create Deployment User

Create a new user with `sudo` permissions for the GitHub Actions runner.

| Command | Description |
| :--- | :--- |
| `sudo adduser github_user` | Creates the new user. Follow the prompts for password and details. |
| `sudo usermod -aG sudo github_user` | Adds the new user to the **`sudo`** group, granting administrative privileges. |
| `sudo su - github_user` | **Switch** the current session to the new **`github_user`** (essential for the next steps). |

### 2.2 Generate SSH Key for `github_user`

Generate an SSH key pair for the `github_user`. The **private key** will be stored as a GitHub Secret, and the **public key** will be placed in the server's authorized keys.

| Command | Description |
| :--- | :--- |
| `ssh-keygen -t rsa -b 4096 -C "githubb_user_key" -f ~/.ssh/github_user_key -N ""` | Generates the RSA key pair. **`github_user_key`** is the private key, and **`github_user_key.pub`** is the public key. [cite\_start]The `-N ""` option creates a key without a passphrase. [cite: 2] |
| `cd ~/.ssh/` | Navigate to the SSH configuration directory. |
| `ls -al` | Verify the two key files were created. |
| `chmod 700 ~/.ssh/` | Sets permissions for the directory (owner has full access). |
| `chmod 644 ~/.ssh/github_user_key.pub` | Sets permissions for the public key. |
| `chmod 600 ~/.ssh/github_user_key` | Sets permissions for the private key (only owner can read/write). |
| `cat github_user_key.pub >> authorized_keys` | Appends the new public key to the **`authorized_keys`** file, allowing login with the corresponding private key. |
| `cat authorized_keys` | Displays the content to confirm the public key was added. |

### 2.3 Retrieve Private Key and Cleanup

The **private key** must be secured on your local machine and **removed from the server**.

1.  **Switch back** to the `ubuntu` user to handle file permissions and copying.

    | Command | Description |
    | :--- | :--- |
    | `cd /home/github_user/.ssh` | [cite\_start]Navigate to the key location. [cite: 3] |
    | `sudo cp github_user_key /home/ubuntu/` | Copy the private key to the `ubuntu` user's home directory. |
    | `sudo chown ubuntu:ubuntu /home/ubuntu/github_user_key` | Change the key's owner to `ubuntu` for retrieval. |
    | `sudo su - ubuntu` | Switch to the `ubuntu` user. |
    | `cd ~` | Go to the home directory. |

2.  **On your Local Machine (new Git Bash window):** Copy the private key from the server to your local machine.

    | Command | Description |
    | :--- | :--- |
    | `scp -i rag-app-aws-key.pem ubuntu@Public_IP:/home/ubuntu/github_user_key .` | [cite\_start]Securely copies the private key to your current local directory. [cite: 4] |

3.  **Test the new key (Optional):** Ensure the key works before deletion.

    | Command | Description |
    | :--- | :--- |
    | `ssh -i github_user_key github_user@Public_IP` | Attempts login as `github_user` using the new private key. |

4.  **On the Server (as `ubuntu` user):** **Delete the private key** from the server as it should only reside on the client (your local machine and GitHub Secrets).

    | Command | Description |
    | :--- | :--- |
    | `sudo rm /home/ubuntu/github_user_key` | Removes the private key from the `ubuntu` user's home directory. |
    | `sudo rm /home/github_user/.ssh/github_user_key` | *If you are still the `github_user`, this would remove the file.* (Ensure the file is deleted from all server locations). |

-----

## 3\. GitHub Repository Access Configuration

Since your repository is private, you need a separate deploy key to allow the server to securely clone and pull code from GitHub.

### 3.1 Generate Deploy Key and Configure SSH

| Command | Description |
| :--- | :--- |
| `sudo su - github_user` | Switch back to the `github_user` if you aren't already. |
| `ssh-keygen -t ed25519 -C "github_deplo_key" -f ~/.ssh/github_deploy_key` | Generates the Ed25519 deploy key pair (modern and secure). Press **Enter** twice for no passphrase. |
| `sudo apt install nano` | [cite\_start]Installs the `nano` text editor. [cite: 4] |
| `nano ~/.ssh/config` | [cite\_start]Opens the SSH configuration file for editing. [cite: 5] |

**Content to add to `~/.ssh/config`:**

```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/github_deploy_key
  IdentitiesOnly yes
```

  * **Save:** Press **`Ctrl + X`**, then **`Y`** (for yes), then **`Enter`**.

### 3.2 Add Public Deploy Key to GitHub

1.  **On the Server:** Display the public key content.

    | Command | Description |
    | :--- | :--- |
    | `cd ~/.ssh/` | Navigate to the directory. |
    | `cat github_deploy_key.pub` | Copy the entire output (the public key). |

2.  **On GitHub:**

      * Go to your **Repository Settings** -\> **Deploy Keys**.
      * Click **Add deploy key**.
      * **Name:** `github_deploy_key`
      * **Key:** Paste the public key copied above.
      * **Allow write access:** **Uncheck** (The server only needs pull/read access).

3.  **Test Connection:**

    | Command | Description |
    | :--- | :--- |
    | `ssh -T git@github.com` | [cite\_start]Tests the SSH connection with GitHub using the deploy key. [cite: 6] |
    | **Action** | Type **`yes`** when prompted. A successful connection will show a welcome message. |

-----

## 4\. Application Cloning and Docker Installation

### 4.1 Clone Repository

Set up a workspace and clone your application using the secure SSH URL.

| Command | Description |
| :--- | :--- |
| `cd ~` | Go to the home directory. |
| `mkdir workspace/` | Create a directory for your projects. |
| `cd workspace/` | Enter the new directory. |
| `git clone git@github.com:muhabmagdy/-Personalized-Chatbot-with-Dynamic-Responses-.git` | Clone your repository using the SSH URL. |
| `cd -Personalized-Chatbot-with-Dynamic-Responses-` | Navigate into the cloned repository directory. |
| `cd src/` | Navigate to the `src` directory. |
| `cp .env.example .env` | Create the essential **`.env`** file from the example. (You will edit this later with VS Code Insiders). |

### 4.2 Install Docker Engine

Install Docker Engine, CLI, and Compose plugin using the official `apt` repository method.

1.  [cite\_start]**Set up Docker's APT Repository** [cite: 7]

    ```bash
    sudo apt update
    sudo apt install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
    Types: deb
    URIs: https://download.docker.com/linux/ubuntu
    Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
    Components: stable
    Signed-By: /etc/apt/keyrings/docker.asc
    EOF

    sudo apt update
    ```

2.  [cite\_start]**Install Docker Packages** [cite: 8]

    ```bash
    sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```

      * Type **`y`** for yes when prompted.

3.  **Verify Installation**

    ```bash
    sudo docker run hello-world
    ```

-----

## 5\. Remote Development with VS Code Insiders

Use the VS Code Insiders CLI to remotely edit files on your server via a secure web tunnel.

### 5.1 Install VS Code Insiders CLI

| Command | Description |
| :--- | :--- |
| `cd ~` | Go to home directory. |
| `wget https://vscode.download.prss.microsoft.com/dbazure/download/insider/b1018f0c37cb5dc62bba1665f6d2821fff098646/vscode_cli_alpine_x64_cli.tar.gz` | Download the VS Code Insiders CLI package. |
| `tar -xvf vscode_cli_alpine_x64_cli.tar.gz` | Extract the package, which includes the `code-insiders` binary. |
| `chmod +x code-insiders` | [cite\_start]Make the binary executable. [cite: 10] |

### 5.2 Create and Manage a Background Session

Use `screen` to run the VS Code tunnel process in the background, allowing you to disconnect your SSH session without stopping it.

| Command | Description |
| :--- | :--- |
| `sudo apt install screen` | Installs the `screen` utility. |
| `screen -S vscode` | [cite\_start]Starts a new `screen` session named **`vscode`**. [cite: 10] |
| `./code-insiders tunnel` | [cite\_start]Starts the remote tunnel. [cite: 10] |

**Follow the on-screen prompts:**

1.  [cite\_start]Log in to the provided **GitHub URL** (`https://github.com/login/device`) and enter the specific code. [cite: 11]
2.  Authorize your account.
3.  Name your machine (e.g., `ip-172-26-14-45`).
4.  VS Code provides a secure URL (e.g., `https://insiders.vscode.dev/tunnel/...`).

**`screen` Session Management:**

  * **Detach** (send to background): Press **`Ctrl + A`**, release, then press **`D`**.
  * **Resume** (reconnect): `screen -r vscode`
  * **Exit** (stop tunnel): `screen -r vscode` then **`Ctrl + C`**

### 5.3 Edit Files and Initial Docker Start

Use the VS Code Insiders URL in your browser to edit configuration files like `.env`, then start your application.

  * **Edit Files:** Use the VS Code web editor to update `.env`, `alembic.ini`, etc.

  * [cite\_start]**Configure Firewall:** Ensure **Port `3000`** (or your application's port) is opened in the **AWS Lightsail Networking/Firewall** settings. [cite: 12]

  * **Start Docker:**

    ```bash
    cd /home/github_user/workspace/-Personalized-Chatbot-with-Dynamic-Responses-/
    docker compose up --build -d
    ```

      * [cite\_start]The application should now be accessible at `Public IPV4 address:3000`. [cite: 12]

-----

## 6\. Systemd Service Configuration

A **systemd service** ensures your Docker Compose application starts automatically on server boot/restart and can be managed reliably.

### 6.1 Create the Service File

| Command | Description |
| :--- | :--- |
| `cd /etc/systemd/system` | Navigate to the systemd configuration directory. |
| `sudo nano myrag.service` | Create and open the service file. |

**Content for `myrag.service`:**

```ini
[Unit]
Description=MyRAG Docker Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=forking
RemainAfterExit=yes
User=github_user
Group=docker
WorkingDirectory=/home/github_user/workspace/-Personalized-Chatbot-with-Dynamic-Responses-/
ExecStartPre=/bin/bash -c '/usr/bin/docker compose down || true'
ExecStartPre=/bin/sleep 5
ExecStart=/usr/bin/docker compose up --build -d
ExecStop=/usr/bin/docker compose down
ExecReload=/usr/bin/docker compose restart
TimeoutStartSec=300
TimeoutStopSec=120
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

  * **Note:** I've adjusted `WorkingDirectory` to the repository root, as `docker compose` is typically run there.
  * [cite\_start]**Save:** Press **`Ctrl + X`**, then **`Y`**, then **`Enter`**. [cite: 13]

### 6.2 Manage the Service

Use `systemctl` to load, start, and enable the service.

| Command | Description |
| :--- | :--- |
| `sudo systemctl daemon-reload` | Reloads the systemd manager configuration to recognize the new file. |
| `sudo systemctl start myrag.service` | Starts the application service. |
| `sudo systemctl status myrag.service` | Checks the running status and logs. |
| `sudo systemctl enable myrag.service` | Configures the service to start automatically on server boot. |
| `sudo journalctl -u myrag.service -f` | Views real-time logs for the service. |

-----

## 7\. Configure GitHub Actions for Deployment

[cite\_start]GitHub Actions will automate code updates and service restarts upon changes to specific branches. [cite: 14]

### 7.1 Enable `NOPASSWD` for Service Restart

Since GitHub Actions runs non-interactively, the `github_user` must be able to restart the service without being prompted for a `sudo` password.

| Command | Description |
| :--- | :--- |
| `sudo visudo` | [cite\_start]Opens the `sudoers` configuration file for editing. [cite: 15] |

**Add the following line at the end of the file:**

```
github_user ALL=(ALL) NOPASSWD: /bin/systemctl restart myrag.service
```

  * [cite\_start]This grants the `github_user` permission to run **only** the specified command with `sudo` privileges without a password. [cite: 15]

### 7.2 Configure GitHub Secrets

The GitHub Action workflow will require access credentials to the server. [cite\_start]These must be stored securely as **Repository Secrets**. [cite: 16]

1.  **On GitHub:**

      * Go to **Repository Settings** -\> **Secrets and variables** -\> **Actions** -\> **New repository secret**.

2.  **Secrets to Create:**

| Secret Name | Value | Description |
| :--- | :--- | :--- |
| `SSH_MAIN_HOST_IP` | Your Lightsail Public IP (e.g., `192.0.2.1`). | [cite\_start]The public IP of the server. [cite: 16] |
| `SSH_MAIN_PRIVATE_KEY` | The **entire content** of `github_user_key`. | [cite\_start]The private key generated for `github_user` in **Step 2.2**. [cite: 16] |

You can now reference these secrets within your GitHub Actions workflow file (`.yml`) to deploy your application automatically.
