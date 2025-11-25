# 💻 Setup Guide: WSL, Ubuntu, Miniconda, and Project Deployment

This document outlines the steps required to set up a Linux environment on Windows using WSL 2, install necessary dependencies (Miniconda), and deploy the project services.

---

## 1. Windows Subsystem for Linux (WSL) Installation

First, ensure your Windows version meets the requirements.

### Check Windows Version

1. Open the **Start Menu**.
2. Search for and open **System Information**.
3. Check the **OS Name** and **Version**:

   * **Windows 11** is supported.
   * **Windows 10** must be version **20.04** or higher.

### Install WSL and Ubuntu

Run **Windows PowerShell** as **Administrator**, then execute:

```
wsl --install
wsl --set-default-version 2
```

Verify WSL version:

```
wsl --version
```

Install Ubuntu:

```
wsl --install Ubuntu
```

---

## 2. Ubuntu Terminal Setup

1. Launch **Ubuntu** from Start Menu.
2. Create **username** and **password**.
3. Update packages:

```
sudo apt update
```

---

## 3. Working with WSL and Windows Files

Access Windows directory:

```
cd /mnt/d/development/rag
```

---

## 4. Using Visual Studio Code with WSL

```
code .
```

---

## 5. Install Miniconda in WSL

```
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Apply configuration:

```
bash ~/.profile
```

---

## 6. Create and Activate Conda Environment

```
conda create -n rag-app python=3.13
conda activate rag-app
conda info --envs
```

---

## 7. Optional: Improve Terminal Appearance

```
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

---

## 8. Install Python Dependencies

```
pip install -r requirements.txt
```

---

## 9. Setup Environment Variables

```
cp .env.example .env
```

---

## 10. Run Alembic Migrations

```
alembic upgrade head
```

---

## 11. Run Docker Compose Services

```
cd docker
cp .env.example .env
sudo docker compose up -d
```

---

## 12. Run FastAPI Server

```
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```
