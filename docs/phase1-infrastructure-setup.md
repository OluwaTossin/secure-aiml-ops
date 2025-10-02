# Phase 1: Infrastructure Setup

This phase focuses on setting up the core AWS infrastructure components required for the AI/ML pipeline.

## Overview

The infrastructure setup includes:
- AWS VPC with public and private subnets
- IAM roles and security policies
- AWS ECR for container registry
- AWS Copilot for application deployment

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS VPC                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   Public Subnet     │    │     Private Subnet          │ │
│  │   10.0.1.0/24      │    │     10.0.2.0/24             │ │
│  │                     │    │                             │ │
│  │  ┌───────────────┐  │    │  ┌───────────────────────┐  │ │
│  │  │ Streamlit App │  │    │  │    Apache Airflow     │  │ │
│  │  │ (Load Balancer)│  │    │  │    (Private EC2)      │  │ │
│  │  └───────────────┘  │    │  └───────────────────────┘  │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                               │
                    ┌─────────────────────┐
                    │     AWS ECR         │
                    │  Container Registry │
                    └─────────────────────┘
```

## Prerequisites

- AWS CLI configured with appropriate permissions
- Terraform installed
- AWS account with Free Tier eligible services

## Step 1: VPC and Networking

### VPC Configuration
- **CIDR Block**: 10.0.0.0/16
- **Public Subnet**: 10.0.1.0/24 (for web-facing services)
- **Private Subnet**: 10.0.2.0/24 (for backend services)

### Security Groups
1. **Web Security Group**: HTTP/HTTPS traffic from internet
2. **Internal Security Group**: Communication between services
3. **Database Security Group**: Restricted database access

## Step 2: IAM Roles and Policies

### Required IAM Roles

#### 1. Airflow Execution Role
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability"
            ],
            "Resource": "*"
        }
    ]
}
```

#### 2. Bedrock Access Role
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:ListModels"
            ],
            "Resource": "*"
        }
    ]
}
```

#### 3. ECR Access Role
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:PutImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload"
            ],
            "Resource": "*"
        }
    ]
}
```

## Step 3: AWS ECR Setup

### Create ECR Repository
```bash
aws ecr create-repository \
    --repository-name secure-aiml-ops \
    --region eu-west-1
```

### Configure Image Scanning
```bash
aws ecr put-image-scanning-configuration \
    --repository-name secure-aiml-ops \
    --image-scanning-configuration scanOnPush=true
```

## Step 4: AWS Copilot Deployment

### Initialize Copilot Application
```bash
copilot app init secure-aiml-ops
cd secure-aiml-ops
```

### Create Backend Service Environment
```bash
copilot env init --name production
copilot env deploy --name production
```

## Deployment Commands

1. **Initialize Terraform**
   ```bash
   cd infrastructure/
   terraform init
   ```

2. **Plan Infrastructure**
   ```bash
   terraform plan -var-file="terraform.tfvars"
   ```

3. **Deploy Infrastructure**
   ```bash
   terraform apply -var-file="terraform.tfvars"
   ```

4. **Verify Deployment**
   ```bash
   aws ec2 describe-vpcs --filters "Name=tag:Project,Values=secure-aiml-ops"
   aws ecr describe-repositories --repository-names secure-aiml-ops
   ```

## Security Considerations

### Network Security
- Private subnets for sensitive workloads
- NAT Gateway for outbound internet access
- Security groups with least-privilege access

### IAM Best Practices
- Principle of least privilege
- Role-based access control
- Regular permission auditing

### Encryption
- EBS volume encryption
- S3 bucket encryption
- ECR image encryption

## Cost Optimization

### Free Tier Usage
- t2.micro instances for development
- 5GB ECR storage included
- VPC and subnets are free

### Monitoring Costs
```bash
aws ce get-cost-and-usage \
    --time-period Start=2025-01-01,End=2025-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost
```

## Troubleshooting

### Common Issues

1. **VPC Creation Fails**
   - Check AWS region availability
   - Verify CIDR block doesn't conflict

2. **IAM Permission Denied**
   - Ensure AWS CLI has admin permissions
   - Check policy syntax and resource ARNs

3. **ECR Push Fails**
   - Authenticate Docker to ECR
   - Check repository name and region

### Verification Steps

1. **Test VPC Connectivity**
   ```bash
   aws ec2 describe-vpc-endpoints
   ```

2. **Verify IAM Roles**
   ```bash
   aws iam list-roles --path-prefix /secure-aiml-ops/
   ```

3. **Check ECR Repository**
   ```bash
   aws ecr describe-repositories
   ```

## Next Steps

After completing Phase 1:
1. ✅ VPC and networking configured
2. ✅ IAM roles and policies created
3. ✅ ECR repository ready
4. ✅ Copilot environment deployed

**Ready for Phase 2**: Apache Airflow Pipeline Setup

## Resources

- [AWS VPC Documentation](https://docs.aws.amazon.com/vpc/)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [AWS ECR User Guide](https://docs.aws.amazon.com/AmazonECR/latest/userguide/)
- [AWS Copilot Documentation](https://aws.github.io/copilot-cli/)