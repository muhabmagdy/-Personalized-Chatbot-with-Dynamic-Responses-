# GitHub Actions Workflow: Deploy Main Branch to Server

## Overview
This document describes the automated deployment workflow that triggers when code is pushed to the `main` branch. The workflow deploys the application to a production server and performs health checks to ensure successful deployment.

## Workflow File
**Location**: `.github/workflows/deploy-main.yml`

## Trigger Conditions
- **Trigger**: `push` events
- **Branches**: `main` branch only
- **Frequency**: Runs every time code is pushed to the main branch

## Jobs

### `deploy` Job
**Runner Environment**: `ubuntu-latest`

#### Steps Breakdown

1. **Checkout Code**
   - **Action**: `actions/checkout@v4.2.2`
   - **Purpose**: Fetches the repository code to the GitHub Actions runner

2. **Deploy via SSH**
   - **Action**: `appleboy/ssh-action@v1.2.2`
   - **Purpose**: Establishes SSH connection to the production server and executes deployment commands

## Deployment Process

### SSH Configuration
- **Host**: `${{ secrets.SSH_MAIN_HOST_IP }}`
- **Username**: `github_user`
- **Authentication**: SSH Private Key from `${{ secrets.SSH_MAIN_PRIVATE_KEY }}`

### Server-Side Execution Script

```bash
# Navigate to project directory
cd /home/github_user/workspace/-Personalized-Chatbot-with-Dynamic-Responses-

# Update to latest main branch code
git checkout main
git pull

# Restart application service
sudo systemctl restart myrag.service

# Allow service startup time
sleep 20

# Health check - verify port 80 is active
for i in {1..6}; do
  if ss -tuln | grep -q ':80'; then
    echo "✅ Port 80 is now active."
    break
  else
    echo "⏳ Port 80 not ready yet. Retrying in 5 seconds..."
    sleep 5
  fi
done

# Final health check validation
if ! ss -tuln | grep -q ':80'; then
  echo "❌ Service failed to start on port 80"
  exit 1
fi
```

## Health Check Strategy
- **Wait Time**: Initial 20-second wait after service restart
- **Retry Logic**: 6 attempts with 5-second intervals
- **Validation**: Checks if port 80 is listening using `ss -tuln` command
- **Failure Condition**: Exits with error code 1 if port 80 never becomes active

## Required Secrets
| Secret Name | Description |
|-------------|-------------|
| `SSH_MAIN_HOST_IP` | IP address of the production server |
| `SSH_MAIN_PRIVATE_KEY` | SSH private key for server authentication |

## Failure Scenarios
1. SSH connection failure to the server
2. Git pull conflicts or errors
3. Systemd service restart failure
4. Application failure to bind to port 80 within 50 seconds (20s + 6×5s)

## Success Criteria
- Successful SSH connection
- Clean git pull from main branch
- Successful systemd service restart
- Application listening on port 80 within the timeout period

## Monitoring
- Workflow status visible in GitHub Actions tab
- Deployment logs available in GitHub Actions console
- Application health verified through port availability check

This automated deployment pipeline ensures consistent and reliable deployments to the production environment while maintaining visibility into the deployment status and application health.