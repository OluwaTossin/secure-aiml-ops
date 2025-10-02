#!/bin/bash

# Secure AI/ML Operations Setup Script
# This script initializes the development environment

set -e

echo "🚀 Setting up Secure AI/ML Operations Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+"
        exit 1
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_warning "AWS CLI not found. Installing..."
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install
        rm -rf aws awscliv2.zip
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop"
        exit 1
    fi
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        print_warning "Terraform not found. Installing..."
        wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
        sudo apt update && sudo apt install terraform
    fi
    
    print_success "Prerequisites check completed"
}

# Setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Setup AWS Copilot
setup_copilot() {
    print_status "Setting up AWS Copilot..."
    
    if ! command -v copilot &> /dev/null; then
        curl -Lo copilot https://github.com/aws/copilot-cli/releases/latest/download/copilot-linux
        chmod +x copilot && sudo mv copilot /usr/local/bin/copilot
        print_success "AWS Copilot CLI installed"
    else
        print_success "AWS Copilot CLI already installed"
    fi
}

# Create .env file from template
setup_env_file() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_warning "Please update .env file with your AWS account details"
        print_warning "Required: AWS_ACCOUNT_ID, AIRFLOW_ADMIN_PASSWORD"
    fi
    
    print_success "Environment file created"
}

# Initialize Git repository (if not already initialized)
setup_git() {
    if [ ! -d ".git" ]; then
        print_status "Initializing Git repository..."
        git init
        git add .
        git commit -m "Initial commit: Secure AI/ML Operations project setup"
        print_success "Git repository initialized"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    
    mkdir -p logs
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p models
    mkdir -p outputs
    
    print_success "Project directories created"
}

# Main setup function
main() {
    print_status "Starting Secure AI/ML Operations setup..."
    
    check_prerequisites
    setup_python_env
    setup_copilot
    setup_env_file
    create_directories
    setup_git
    
    print_success "🎉 Setup completed successfully!"
    print_status "Next steps:"
    echo "1. Update .env file with your AWS account details"
    echo "2. Configure AWS CLI: aws configure"
    echo "3. Run 'source venv/bin/activate' to activate virtual environment"
    echo "4. Start with Phase 1: Infrastructure Setup"
}

# Run main function
main "$@"