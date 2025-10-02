# Streamlit Application for Secure AI/ML Operations

This directory contains the Streamlit web application that provides an interactive interface for our AI/ML models and data analytics dashboard.

## 🏗️ Application Architecture

```
streamlit/
├── app.py                     # Main Streamlit application
├── pages/                     # Multi-page application structure
│   ├── 1_📊_Dashboard.py     # Analytics dashboard
│   ├── 2_🤖_Text_Summarization.py
│   ├── 3_🔍_Anomaly_Detection.py
│   └── 4_⚙️_Model_Management.py
├── components/                # Reusable UI components
│   ├── __init__.py
│   ├── charts.py             # Chart components
│   ├── forms.py              # Form components
│   └── layouts.py            # Layout components
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── aws_client.py         # AWS service interactions
│   ├── model_client.py       # Model inference client
│   └── data_processor.py     # Data processing utilities
├── config/                    # Configuration files
│   ├── __init__.py
│   └── settings.py           # Application settings
├── static/                    # Static assets
│   ├── css/
│   ├── images/
│   └── js/
├── Dockerfile                 # Container definition
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Local development setup
└── README.md                  # This file
```

## 🎯 Features

### 📊 **Analytics Dashboard**
- Real-time model performance metrics
- Data quality monitoring
- Infrastructure health status
- Business KPI visualization

### 🤖 **Text Summarization Interface**
- Interactive text input/output
- Multiple summarization models
- Real-time processing
- Export functionality

### 🔍 **Anomaly Detection Tool**
- Financial transaction monitoring
- Interactive data visualization
- Alert management
- Historical analysis

### ⚙️ **Model Management**
- Model version control
- Performance comparison
- Deployment status
- Configuration management

## 🚀 Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   cd streamlit
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials and settings
   ```

3. **Run Application**
   ```bash
   streamlit run app.py
   ```

4. **Access Application**
   - URL: http://localhost:8501

### Docker Deployment

1. **Build Container**
   ```bash
   docker build -t secure-aiml-ops-streamlit .
   ```

2. **Run Container**
   ```bash
   docker run -p 8501:8501 \
     -e AWS_ACCESS_KEY_ID=your_key \
     -e AWS_SECRET_ACCESS_KEY=your_secret \
     secure-aiml-ops-streamlit
   ```

3. **Docker Compose (Development)**
   ```bash
   docker-compose up -d
   ```

## 🔧 Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=eu-west-1

# S3 Configuration
S3_BUCKET_NAME=secure-aiml-ops-data
S3_MODELS_PREFIX=models

# ECR Configuration
ECR_REPOSITORY=455921291596.dkr.ecr.eu-west-1.amazonaws.com/secure-aiml-ops

# Application Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Model Configuration
MODEL_ENDPOINT_URL=http://localhost:8000
MODEL_API_KEY=your_api_key
```

## 🎨 UI Components

### Layout Structure
- **Sidebar**: Navigation and filters
- **Main Area**: Primary content and interactions
- **Metrics Bar**: Key performance indicators
- **Charts Area**: Data visualizations

### Theme and Styling
- Modern, clean interface design
- Responsive layout for different screen sizes
- Dark/light theme support
- Consistent color scheme and typography

## 🔌 Integrations

### AWS Services
- **S3**: Model artifacts and data storage
- **ECR**: Container image management
- **CloudWatch**: Metrics and logging
- **IAM**: Authentication and authorization

### ML Model Integration
- **Text Summarization**: T5-based summarization model
- **Anomaly Detection**: Isolation Forest model
- **Model Serving**: REST API endpoints
- **Batch Processing**: Background job integration

### Data Sources
- **Airflow**: Pipeline status and metrics
- **PostgreSQL**: Application database
- **S3**: Raw and processed data
- **Real-time APIs**: Live data feeds

## 📊 Monitoring

### Application Metrics
- User engagement analytics
- Performance monitoring
- Error tracking
- Resource utilization

### Model Performance
- Inference latency
- Accuracy metrics
- Throughput monitoring
- Model drift detection

## 🔐 Security

### Authentication
- AWS IAM integration
- Session management
- Role-based access control
- API key authentication

### Data Protection
- Encrypted data transmission
- Secure credential storage
- Input validation and sanitization
- Audit logging

## 🚢 Deployment

### AWS Copilot Deployment
```bash
# Initialize Copilot application
copilot app init secure-aiml-ops-streamlit

# Deploy to staging
copilot env init --name staging
copilot svc init --name streamlit-app
copilot svc deploy --name streamlit-app --env staging

# Deploy to production
copilot env init --name production
copilot svc deploy --name streamlit-app --env production
```

### ECR Integration
```bash
# Build and push to ECR
docker build -t secure-aiml-ops-streamlit .
docker tag secure-aiml-ops-streamlit:latest 455921291596.dkr.ecr.eu-west-1.amazonaws.com/secure-aiml-ops:streamlit-latest
docker push 455921291596.dkr.ecr.eu-west-1.amazonaws.com/secure-aiml-ops:streamlit-latest
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/unit/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### UI Tests
```bash
python -m pytest tests/ui/
```

## 📈 Performance

### Optimization Techniques
- Streamlit caching for expensive operations
- Lazy loading of large datasets
- Efficient chart rendering
- Background data preprocessing

### Scalability
- Horizontal scaling with load balancer
- Database connection pooling
- CDN integration for static assets
- Microservice architecture support

## 🔧 Development

### Code Quality
- Type hints throughout codebase
- Comprehensive error handling
- Logging and debugging support
- Code formatting with Black and isort

### Development Workflow
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code quality checks
black .
isort .
flake8 .
mypy .

# Run tests
pytest

# Start development server
streamlit run app.py --server.runOnSave true
```

## 📚 Documentation

- **API Documentation**: Auto-generated from docstrings
- **User Guide**: Step-by-step usage instructions
- **Developer Guide**: Technical implementation details
- **Deployment Guide**: Production deployment procedures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow coding standards
4. Add tests for new functionality
5. Submit a pull request

---

**Next Steps**: This Streamlit application will serve as the primary user interface for our secure AI/ML operations platform, providing an intuitive way to interact with our models and monitor system performance.