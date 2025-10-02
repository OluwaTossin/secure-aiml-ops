#!/bin/bash

# Secure AI/ML Operations - Infrastructure Deployment Script
# This script deploys the AWS infrastructure using Terraform

set -e

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
    print_status "Checking prerequisites for infrastructure deployment..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install AWS CLI first."
        exit 1
    fi
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is not installed. Please install Terraform first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Create terraform.tfvars from example if it doesn't exist
setup_terraform_vars() {
    print_status "Setting up Terraform variables..."
    
    if [ ! -f "terraform.tfvars" ]; then
        cp terraform.tfvars.example terraform.tfvars
        print_warning "Created terraform.tfvars from example. Please review and update as needed."
        print_warning "Press Enter to continue or Ctrl+C to exit..."
        read
    fi
    
    print_success "Terraform variables configured"
}

# Initialize Terraform
init_terraform() {
    print_status "Initializing Terraform..."
    
    terraform init
    
    if [ $? -eq 0 ]; then
        print_success "Terraform initialized successfully"
    else
        print_error "Terraform initialization failed"
        exit 1
    fi
}

# Plan Terraform deployment
plan_terraform() {
    print_status "Planning Terraform deployment..."
    
    terraform plan -var-file="terraform.tfvars" -out=tfplan
    
    if [ $? -eq 0 ]; then
        print_success "Terraform plan completed successfully"
        print_warning "Review the plan above. Press Enter to continue with deployment or Ctrl+C to exit..."
        read
    else
        print_error "Terraform planning failed"
        exit 1
    fi
}

# Apply Terraform deployment
apply_terraform() {
    print_status "Applying Terraform deployment..."
    
    terraform apply tfplan
    
    if [ $? -eq 0 ]; then
        print_success "Infrastructure deployed successfully!"
    else
        print_error "Terraform deployment failed"
        exit 1
    fi
}

# Save Terraform outputs
save_outputs() {
    print_status "Saving Terraform outputs..."
    
    terraform output -json > ../outputs.json
    
    if [ $? -eq 0 ]; then
        print_success "Outputs saved to outputs.json"
    else
        print_warning "Failed to save outputs"
    fi
}

# Show post-deployment information
show_post_deployment_info() {
    print_success "🎉 Infrastructure deployment completed!"
    echo ""
    print_status "Next steps:"
    echo "1. ECR Repository: $(terraform output -raw ecr_repository_url)"
    echo "2. VPC ID: $(terraform output -raw vpc_id)"
    echo "3. Configure AWS Copilot with the deployed infrastructure"
    echo "4. Proceed to Phase 2: Apache Airflow Pipeline"
    echo ""
    print_status "To destroy the infrastructure later, run:"
    echo "terraform destroy -var-file=\"terraform.tfvars\""
}

# Main deployment function
main() {
    print_status "🚀 Starting AWS Infrastructure Deployment for Secure AI/ML Operations"
    echo ""
    
    # Change to infrastructure directory
    cd infrastructure
    
    check_prerequisites
    setup_terraform_vars
    init_terraform
    plan_terraform
    apply_terraform
    save_outputs
    show_post_deployment_info
    
    print_success "Infrastructure deployment script completed!"
}

# Handle script arguments
case "${1:-}" in
    "plan")
        cd infrastructure
        check_prerequisites
        setup_terraform_vars
        init_terraform
        terraform plan -var-file="terraform.tfvars"
        ;;
    "apply")
        cd infrastructure
        main
        ;;
    "destroy")
        cd infrastructure
        print_warning "This will destroy ALL infrastructure resources!"
        print_warning "Are you sure? Type 'yes' to continue:"
        read confirmation
        if [ "$confirmation" = "yes" ]; then
            terraform destroy -var-file="terraform.tfvars"
        else
            print_status "Destruction cancelled"
        fi
        ;;
    "output")
        cd infrastructure
        terraform output
        ;;
    *)
        echo "Usage: $0 {plan|apply|destroy|output}"
        echo ""
        echo "Commands:"
        echo "  plan     - Show what will be deployed"
        echo "  apply    - Deploy the infrastructure"
        echo "  destroy  - Destroy the infrastructure"
        echo "  output   - Show deployment outputs"
        echo ""
        echo "For full deployment, run: $0 apply"
        exit 1
        ;;
esac