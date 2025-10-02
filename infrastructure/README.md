# Infrastructure Deployment Guide

This directory contains Terraform Infrastructure as Code (IaC) for the Secure AI/ML Operations project.

## 🏗️ Infrastructure Components

### Core Infrastructure
- **VPC** with public and private subnets in eu-west-1
- **Security Groups** for web, internal, database, and ALB traffic
- **NAT Gateway** for private subnet internet access
- **Internet Gateway** for public subnet access

### Container & Compute
- **ECR Repository** with vulnerability scanning enabled
- **IAM Roles** for Airflow, Bedrock, ECR, and ECS tasks
- **Instance Profiles** for EC2 services

### Storage & Data
- **S3 Bucket** with encryption and lifecycle policies
- **CloudWatch Log Groups** for centralized logging

### Security & Monitoring
- **GuardDuty** threat detection (optional)
- **CloudWatch Alarms** for error monitoring
- **SNS Topics** for alerting
- **IAM Policies** with least-privilege access

## 🚀 Quick Deployment

### Prerequisites
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# Install Terraform
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform

# Configure AWS credentials
aws configure
```

### Deployment Commands
```bash
# Option 1: Use deployment script (Recommended)
./scripts/deploy-infrastructure.sh apply

# Option 2: Manual Terraform commands
cd infrastructure/
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -var-file="terraform.tfvars"
```

## 📋 Configuration

### Required Variables
Copy `terraform.tfvars.example` to `terraform.tfvars` and update:

```hcl
# AWS Configuration
aws_region = "eu-west-1"

# Project Configuration
project_name = "secure-aiml-ops"
environment  = "development"

# VPC Configuration
vpc_cidr              = "10.0.0.0/16"
public_subnet_cidr    = "10.0.1.0/24"
private_subnet_cidr   = "10.0.2.0/24"

# Security Configuration
enable_encryption = true
enable_guardduty  = true
```

## 🔍 Verification

After deployment, verify infrastructure:

```bash
# Check VPC
aws ec2 describe-vpcs --filters "Name=tag:Project,Values=secure-aiml-ops"

# Check ECR Repository
aws ecr describe-repositories --repository-names secure-aiml-ops

# Check IAM Roles
aws iam list-roles --path-prefix /secure-aiml-ops/

# View Terraform outputs
terraform output
```

## 📊 Resource Overview

| Resource Type | Count | Purpose |
|---------------|-------|---------|
| VPC | 1 | Network isolation |
| Subnets | 2 | Public/Private separation |
| Security Groups | 4 | Traffic control |
| IAM Roles | 4 | Service permissions |
| ECR Repository | 1 | Container storage |
| S3 Bucket | 1 | Data storage |
| CloudWatch Groups | 4 | Centralized logging |

## 💰 Cost Estimation

**Free Tier Usage:**
- VPC, Subnets, Security Groups: Free
- NAT Gateway: ~$32/month (main cost)
- ECR: 500MB free, then $0.10/GB/month
- CloudWatch Logs: 5GB free, then $0.50/GB
- GuardDuty: 30-day free trial

**Total Estimated Cost:** ~$35-40/month

## 🔒 Security Features

- **Network Isolation:** Public/private subnet architecture
- **Encryption:** S3 server-side encryption enabled
- **Access Control:** IAM roles with least-privilege policies
- **Monitoring:** GuardDuty threat detection
- **Compliance:** SSL-only S3 access policies

## 🧹 Cleanup

To destroy all infrastructure:

```bash
# Using script
./scripts/deploy-infrastructure.sh destroy

# Manual cleanup
cd infrastructure/
terraform destroy -var-file="terraform.tfvars"
```

## 🔧 Troubleshooting

### Common Issues

1. **AWS Credentials Error**
   ```bash
   aws configure
   aws sts get-caller-identity
   ```

2. **Terraform State Lock**
   ```bash
   terraform force-unlock <LOCK_ID>
   ```

3. **Resource Naming Conflicts**
   - Ensure unique project_name in terraform.tfvars
   - Check for existing resources in AWS console

4. **Permission Denied**
   - Verify AWS user has sufficient permissions
   - Check IAM policies for deployment user

## 📝 Next Steps

After successful deployment:

1. ✅ Infrastructure ready
2. 🔄 Proceed to Phase 2: Apache Airflow Pipeline
3. 🐳 Configure containerized applications
4. 🤖 Set up AWS Bedrock integration
5. 📊 Implement monitoring and optimization

## 📚 References

- [AWS VPC Documentation](https://docs.aws.amazon.com/vpc/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest)
- [AWS Free Tier](https://aws.amazon.com/free/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)