#!/bin/bash

# Airflow Initialization Script for Secure AI/ML Operations
# This script sets up the Airflow environment and initializes the database

set -e

echo "🚀 Starting Airflow initialization for Secure AI/ML Operations..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info &>/dev/null; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker is running ✅"

# Check if docker-compose is available
if ! command -v docker-compose &>/dev/null; then
    print_error "docker-compose is not installed. Please install docker-compose and try again."
    exit 1
fi

print_status "docker-compose is available ✅"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p ./logs
mkdir -p ./plugins
mkdir -p ./config
mkdir -p ./dags
mkdir -p ./data

# Set proper permissions for Airflow
print_status "Setting proper permissions..."
sudo chown -R 50000:0 ./logs
sudo chown -R 50000:0 ./plugins
sudo chown -R 50000:0 ./config
sudo chown -R 50000:0 ./dags

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file..."
    cat > .env << EOF
# Airflow Configuration
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
AIRFLOW__CORE__FERNET_KEY=
AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true
AIRFLOW__CORE__LOAD_EXAMPLES=false
AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth
AIRFLOW__WEBSERVER__EXPOSE_CONFIG=true

# AWS Configuration
AWS_DEFAULT_REGION=eu-west-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# PostgreSQL Configuration
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow

# Redis Configuration (for Celery if needed)
REDIS_HOST=redis
REDIS_PORT=6379

# Security
AIRFLOW__WEBSERVER__SECRET_KEY=secure_aiml_ops_secret_key_2024
EOF
    print_warning "Created .env file. Please update AWS credentials before starting Airflow."
else
    print_status ".env file already exists ✅"
fi

# Generate Fernet key if not set
if ! grep -q "AIRFLOW__CORE__FERNET_KEY=.*[A-Za-z0-9]" .env; then
    print_status "Generating Fernet key..."
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
    sed -i "s/AIRFLOW__CORE__FERNET_KEY=/AIRFLOW__CORE__FERNET_KEY=${FERNET_KEY}/" .env
    print_status "Fernet key generated and added to .env ✅"
fi

# Initialize Airflow database
print_status "Initializing Airflow database..."
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
print_status "Waiting for PostgreSQL to be ready..."
until docker-compose exec postgres pg_isready -U airflow; do
    sleep 1
done

print_status "PostgreSQL is ready ✅"

# Initialize Airflow
print_status "Running Airflow database migration..."
docker-compose run --rm airflow-webserver airflow db init

# Create admin user
print_status "Creating Airflow admin user..."
docker-compose run --rm airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@secure-aiml-ops.com \
    --password admin

# Start all Airflow services
print_status "Starting Airflow services..."
docker-compose up -d

# Wait for services to be ready
print_status "Waiting for Airflow webserver to be ready..."
sleep 30

# Check if webserver is accessible
if curl -f http://localhost:8080/health &>/dev/null; then
    print_status "Airflow webserver is ready ✅"
else
    print_warning "Airflow webserver might not be ready yet. Please check logs."
fi

# Display service status
print_status "Checking service status..."
docker-compose ps

echo ""
print_status "🎉 Airflow initialization completed!"
echo ""
echo -e "${BLUE}📊 Access Airflow Web UI at: http://localhost:8080${NC}"
echo -e "${BLUE}👤 Username: admin${NC}"
echo -e "${BLUE}🔑 Password: admin${NC}"
echo ""
print_status "To view logs: docker-compose logs -f"
print_status "To stop services: docker-compose down"
print_status "To restart services: docker-compose restart"
echo ""

# Test AWS connectivity if credentials are set
if grep -q "AWS_ACCESS_KEY_ID=.*[A-Za-z0-9]" .env; then
    print_status "Testing AWS connectivity..."
    if docker-compose run --rm airflow-webserver python -c "
import boto3
try:
    s3 = boto3.client('s3', region_name='eu-west-1')
    s3.list_buckets()
    print('✅ AWS connectivity test passed')
except Exception as e:
    print(f'❌ AWS connectivity test failed: {e}')
"; then
        print_status "AWS connectivity test completed"
    fi
else
    print_warning "AWS credentials not configured. Please update .env file with your AWS credentials."
fi

print_status "Setup complete! 🚀"