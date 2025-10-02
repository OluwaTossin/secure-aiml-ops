# Apache Airflow for Secure AI/ML Operations

This directory contains the Apache Airflow setup for orchestrating AI/ML workflows in the Secure AI/ML Operations project. Airflow provides robust workflow management, scheduling, and monitoring capabilities for our machine learning pipelines.

## 🏗️ Architecture Overview

```
airflow/
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Custom Airflow image
├── requirements.txt            # Python dependencies
├── config/
│   └── airflow.cfg            # Airflow configuration
├── dags/                      # DAG definitions
│   ├── data_ingestion_dag.py
│   ├── data_preprocessing_dag.py
│   ├── model_training_dag.py
│   ├── model_deployment_dag.py
│   └── monitoring_alerting_dag.py
├── plugins/                   # Custom plugins
│   ├── operators.py          # Custom operators
│   └── utils.py              # Utility functions
├── scripts/                   # Management scripts
│   ├── init_airflow.sh       # Initialization script
│   └── manage_airflow.sh     # Management script
└── logs/                     # Airflow logs
```

## � Quick Start

### Prerequisites

- Docker and Docker Compose installed
- AWS credentials configured
- 8GB+ RAM recommended
- 10GB+ free disk space

### 1. Initialize Airflow

```bash
cd airflow
chmod +x scripts/init_airflow.sh
./scripts/init_airflow.sh
```

This script will:
- Set up necessary directories and permissions
- Generate security keys
- Initialize the database
- Create an admin user
- Start all services

### 2. Access Airflow Web UI

- **URL**: http://localhost:8080
- **Username**: `admin`
- **Password**: `admin`
### 3. Configure AWS Credentials

Update the `.env` file with your AWS credentials:

```bash
# Edit .env file
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
```

Then restart services:
```bash
./scripts/manage_airflow.sh restart
```

## � Available DAGs

### 1. Data Ingestion DAG (`data_ingestion_dag.py`)
- **Purpose**: Ingest and validate raw data from multiple sources
- **Schedule**: Daily at 2:00 AM UTC
- **Features**:
  - Customer ticket simulation
  - Financial transaction generation
  - External API data collection
  - Data validation and quality checks
  - S3 storage with partitioning

### 2. Data Preprocessing DAG (`data_preprocessing_dag.py`)
- **Purpose**: Clean, transform, and prepare data for ML training
- **Schedule**: Triggered after data ingestion
- **Features**:
  - Text preprocessing with NLTK
  - Feature engineering
  - Data standardization
  - Train/validation dataset creation
  - Data quality validation

### 3. Model Training DAG (`model_training_dag.py`)
- **Purpose**: Train and evaluate ML models
- **Schedule**: Triggered after preprocessing
- **Features**:
  - Text summarization model (T5)
  - Anomaly detection model (Isolation Forest)
  - Model evaluation and validation
  - Model artifact storage
  - Performance reporting

### 4. Model Deployment DAG (`model_deployment_dag.py`)
- **Purpose**: Deploy trained models to staging/production
- **Schedule**: Manual trigger
- **Features**:
  - Model artifact validation
  - Docker image creation
  - ECR deployment
  - Staging environment deployment
  - Integration testing

### 5. Monitoring & Alerting DAG (`monitoring_alerting_dag.py`)
- **Purpose**: Monitor system health and model performance
- **Schedule**: Every hour
- **Features**:
  - Model performance monitoring
  - Infrastructure health checks
  - Data quality monitoring
  - Alert generation
  - Automated reporting

## �️ Management Commands

Use the management script for common operations:

```bash
# Start services
./scripts/manage_airflow.sh start

# Stop services
./scripts/manage_airflow.sh stop

# Restart services
./scripts/manage_airflow.sh restart

# Check service status
./scripts/manage_airflow.sh status

# View logs
./scripts/manage_airflow.sh logs

# List available DAGs
./scripts/manage_airflow.sh list-dags

# Trigger a DAG
./scripts/manage_airflow.sh trigger data_ingestion_pipeline

# Test a DAG
./scripts/manage_airflow.sh test-dag data_ingestion_pipeline

# System health check
./scripts/manage_airflow.sh health

# Cleanup old files
./scripts/manage_airflow.sh cleanup

# Backup database
./scripts/manage_airflow.sh backup

# Open Airflow shell
./scripts/manage_airflow.sh shell
```

## 🔧 Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Core Airflow settings
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__FERNET_KEY=<generated_key>
AIRFLOW__CORE__LOAD_EXAMPLES=false

# Database configuration
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow

# AWS configuration
AWS_DEFAULT_REGION=eu-west-1
AWS_ACCESS_KEY_ID=<your_key>
AWS_SECRET_ACCESS_KEY=<your_secret>

# Security
AIRFLOW__WEBSERVER__SECRET_KEY=<secret_key>
```

## 📊 Monitoring and Observability

### Built-in Monitoring

1. **Web UI**: Real-time DAG monitoring at http://localhost:8080
2. **Task Logs**: Detailed execution logs for each task
3. **Metrics**: Built-in metrics and performance indicators
4. **Health Checks**: Automated health monitoring

### Custom Monitoring

The monitoring DAG provides:
- Model performance tracking
- Infrastructure health monitoring
- Data quality assessments
- Automated alerting
- Daily reports

## � Troubleshooting

### Common Issues

#### 1. Services Not Starting
```bash
# Check Docker status
docker info

# Check logs
./scripts/manage_airflow.sh logs

# Restart services
./scripts/manage_airflow.sh restart
```

#### 2. DAG Import Errors
```bash
# Check DAG import errors
./scripts/manage_airflow.sh shell
airflow dags list-import-errors

# Test DAG syntax
python -m py_compile dags/your_dag.py
```

#### 3. Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose ps postgres

# Restart database
docker-compose restart postgres
```

## 🔐 Security

- Default: Basic authentication with admin/admin
- Production: Configure OAuth, LDAP, or RBAC
- Use AWS IAM roles for service authentication
- Encrypt sensitive variables using Fernet

## � Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [AWS Provider Documentation](https://airflow.apache.org/docs/apache-airflow-providers-amazon/)
- [Docker Compose Reference](https://docs.docker.com/compose/)

---

**Note**: This setup is optimized for development and small-scale production environments. For large-scale production deployments, consider using managed services like AWS MWAA.

## 🐛 Troubleshooting

Common issues and solutions:
- **DAG Import Errors**: Check Python syntax and imports
- **Task Failures**: Review logs in Airflow UI
- **Connection Issues**: Verify AWS credentials and network
- **Performance**: Scale workers and adjust resources