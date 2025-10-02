# Scalable AI/ML Infrastructure Deployment on AWS

## Project Overview

This project demonstrates the design and deployment of a cost-effective, scalable, and secure AI/ML infrastructure using AWS services. The goal is to simulate a real-world AI/ML cloud engineering workflow that addresses the business challenge of processing unstructured data efficiently.

## Business Challenge

**Scenario**: A fintech company's support team receives thousands of long customer service tickets monthly, leading to:
- High average handling time (AHT) for resolving cases
- Inconsistent triage of tickets, increasing errors and escalations
- Inefficient knowledge access with important details buried in conversations

**Solution**: Deploy a pre-trained Hugging Face model for summarization integrated with AWS Bedrock for conversational AI to reduce AHT, improve accuracy, and enforce strong DevSecOps practices.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AWS VPC       │    │   Apache        │    │   Streamlit     │
│ Public/Private  │────│   Airflow       │────│   Application   │
│   Subnets       │    │   Pipeline      │    │   (Frontend)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   AWS ECR       │              │
         └──────────────│   Container     │──────────────┘
                        │   Registry      │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   AWS Bedrock   │
                        │   LLM Service   │
                        └─────────────────┘
```

## Tech Stack

- **AWS Services**: VPC, ECR, Copilot, Bedrock, CloudWatch, GuardDuty
- **ML/AI**: Apache Airflow, Streamlit, Hugging Face models
- **Security**: IAM roles, RBAC, encryption, least-privilege access
- **Containerization**: Docker + ECR + Copilot orchestration

## Project Phases

### Phase 1: Infrastructure Setup
- ✅ AWS VPC with public/private subnets
- ✅ IAM roles and security policies
- ✅ AWS ECR setup
- ✅ AWS Copilot deployment

### Phase 2: AI/ML Pipeline with Apache Airflow
- ✅ Airflow installation and configuration
- ✅ DAG creation for data workflows
- ✅ Model training automation

### Phase 3: Containerized Model Deployment
- ✅ Streamlit application development
- ✅ Docker containerization
- ✅ ECR integration and deployment

### Phase 4: LLM Integration with AWS Bedrock
- ✅ AWS Bedrock configuration
- ✅ Chatbot integration
- ✅ Performance optimization

### Phase 5: Security and Optimization
- ✅ Security best practices
- ✅ Monitoring and logging
- ✅ Cost optimization

## Getting Started

1. **Prerequisites**
   - AWS CLI configured with appropriate permissions
   - Docker installed
   - Python 3.8+ environment
   - AWS Copilot CLI

2. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd secure-aiml-ops
   ./scripts/setup.sh
   ```

3. **Deploy Infrastructure**
   ```bash
   cd infrastructure/
   terraform init
   terraform plan
   terraform apply
   ```

## Project Structure

```
secure-aiml-ops/
├── docs/                      # Documentation
├── infrastructure/            # AWS infrastructure as code
├── airflow/                   # Apache Airflow DAGs and configs
├── streamlit-app/            # Streamlit application
├── docker/                   # Docker configurations
├── scripts/                  # Deployment and utility scripts
├── security/                 # Security configurations
├── monitoring/               # CloudWatch and monitoring
└── tests/                    # Testing files
```

## Cost Management

This project is designed to run on AWS Free Tier where possible:
- Use t2.micro instances
- Leverage free CloudWatch logs
- Optimize container resource allocation
- Implement auto-shutdown for non-production resources

## Security Features

- VPC with public/private subnet isolation
- IAM least-privilege access policies
- ECR image vulnerability scanning
- AWS GuardDuty threat detection
- Encrypted data at rest and in transit

## Contributing

Please read our [Contributing Guidelines](docs/CONTRIBUTING.md) and [Code of Conduct](docs/CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.