# PowerShell script to create a deployment package for GCP
# Run from the vegah directory: .\deploy-package.ps1

Write-Host "Creating deployment package..." -ForegroundColor Green

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$packageName = "vegah-deploy-$timestamp.tar.gz"

# Create temporary directory for clean packaging
$tempDir = ".\deploy-temp"
if (Test-Path $tempDir) {
    Remove-Item -Recurse -Force $tempDir
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

Write-Host "Copying files to temporary directory..." -ForegroundColor Cyan

# Copy backend
Copy-Item -Recurse ".\backend" "$tempDir\backend"

# Copy essential files
Copy-Item "requirements.txt" "$tempDir\"
Copy-Item "check_metadata.py" "$tempDir\" -ErrorAction SilentlyContinue
Copy-Item "download_model.py" "$tempDir\" -ErrorAction SilentlyContinue
Copy-Item ".env.example" "$tempDir\" -ErrorAction SilentlyContinue

# Copy data directories (preserve structure but exclude large files)
Write-Host "Copying Chroma database..." -ForegroundColor Cyan
Copy-Item -Recurse ".\chroma_db" "$tempDir\chroma_db" -ErrorAction SilentlyContinue

Write-Host "Copying documents..." -ForegroundColor Cyan
Copy-Item -Recurse ".\documents" "$tempDir\documents" -ErrorAction SilentlyContinue

# Clean up Python cache
Write-Host "Cleaning Python cache files..." -ForegroundColor Cyan
Get-ChildItem -Path $tempDir -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path $tempDir -Recurse -Filter "*.pyc" | Remove-Item -Force
Get-ChildItem -Path $tempDir -Recurse -Filter "*.pyo" | Remove-Item -Force

# Create deployment scripts
Write-Host "Creating deployment scripts..." -ForegroundColor Cyan

# Create remote deployment script
@"
#!/bin/bash
# Remote deployment script - run this on the GCP instance

set -e

DEPLOY_DIR="/opt/vegah"
BACKUP_DIR="/opt/vegah-backups"
SERVICE_NAME="vegah"

echo "==================================="
echo "Vegah Backend Deployment"
echo "==================================="

# Create backup
echo "Creating backup of current deployment..."
BACKUP_PATH="`$BACKUP_DIR/vegah-backup-`$(date +%Y%m%d-%H%M%S)"
sudo mkdir -p `$BACKUP_DIR
if [ -d "`$DEPLOY_DIR" ]; then
    sudo cp -r `$DEPLOY_DIR `$BACKUP_PATH
    echo "Backup created at: `$BACKUP_PATH"
fi

# Stop service
echo "Stopping service..."
sudo systemctl stop `$SERVICE_NAME || echo "Service not running"

# Extract new code
echo "Extracting deployment package..."
sudo mkdir -p `$DEPLOY_DIR
sudo tar -xzf vegah-deploy-*.tar.gz -C `$DEPLOY_DIR
cd `$DEPLOY_DIR

# Set up virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set permissions
echo "Setting permissions..."
sudo chown -R `$(whoami):www-data `$DEPLOY_DIR
chmod -R 755 `$DEPLOY_DIR
chmod -R 775 `$DEPLOY_DIR/chroma_db 2>/dev/null || true
chmod -R 775 `$DEPLOY_DIR/documents 2>/dev/null || true

# Restart service
echo "Starting service..."
sudo systemctl daemon-reload
sudo systemctl start `$SERVICE_NAME
sudo systemctl enable `$SERVICE_NAME

# Check status
echo ""
echo "Deployment complete! Service status:"
sudo systemctl status `$SERVICE_NAME --no-pager

echo ""
echo "To view logs: sudo journalctl -u `$SERVICE_NAME -f"
echo "To check health: curl http://localhost:8000/health"
"@ | Out-File -FilePath "$tempDir\deploy.sh" -Encoding utf8

# Create systemd service file
@"
[Unit]
Description=Vegah Agentic RAG Backend
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
Group=www-data
WorkingDirectory=/opt/vegah
Environment="PATH=/opt/vegah/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/opt/vegah/.env

ExecStart=/opt/vegah/.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1

Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vegah

[Install]
WantedBy=multi-user.target
"@ | Out-File -FilePath "$tempDir\vegah.service" -Encoding utf8

# Create .env template
@"
# Vegah Backend Environment Variables
# Copy this to /opt/vegah/.env and fill in your values

# NVIDIA NIM API
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL_NAME=meta/llama-3.3-70b-instruct

# DeepSeek API (if using)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Application Settings
VEGAH_ENV=production
LOG_LEVEL=INFO

# Paths (relative to /opt/vegah)
CHROMA_PATH=/opt/vegah/chroma_db
COLLECTION_NAME=vegah_documents
UPLOAD_DIR=/opt/vegah/documents

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=ms-marco-MiniLM-L-6-v2
USE_RERANKER=true

# LLM Settings
AGENT_TEMPERATURE=0.1
REASONING_TEMPERATURE=0.3
ANSWER_TEMPERATURE=0.7
MAX_AGENT_ITERATIONS=5
LLM_MAX_TOKENS=2048

# Server Settings (optional)
# CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
"@ | Out-File -FilePath "$tempDir\.env.example" -Encoding utf8

# Create archive
Write-Host "Creating tar.gz archive..." -ForegroundColor Cyan
cd $tempDir
tar -czf "..\$packageName" .
cd ..

# Cleanup temp directory
Remove-Item -Recurse -Force $tempDir

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Deployment package created!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Package: $packageName" -ForegroundColor Yellow
Write-Host "Size: $((Get-Item $packageName).Length / 1MB) MB" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Upload to GCP instance:" -ForegroundColor White
Write-Host "   gcloud compute scp $packageName USERNAME@INSTANCE_NAME:~/ --zone=ZONE" -ForegroundColor Gray
Write-Host "   OR" -ForegroundColor White
Write-Host "   scp $packageName USERNAME@INSTANCE_IP:~/" -ForegroundColor Gray
Write-Host ""
Write-Host "2. SSH into instance:" -ForegroundColor White
Write-Host "   gcloud compute ssh USERNAME@INSTANCE_NAME --zone=ZONE" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Run deployment:" -ForegroundColor White
Write-Host "   tar -xzf $packageName" -ForegroundColor Gray
Write-Host "   chmod +x deploy.sh" -ForegroundColor Gray
Write-Host "   ./deploy.sh" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Update service file (first time only):" -ForegroundColor White
Write-Host "   sudo nano vegah.service  # Update YOUR_USERNAME" -ForegroundColor Gray
Write-Host "   sudo cp vegah.service /etc/systemd/system/" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Configure environment:" -ForegroundColor White
Write-Host "   sudo nano /opt/vegah/.env  # Add your API keys" -ForegroundColor Gray
Write-Host ""
