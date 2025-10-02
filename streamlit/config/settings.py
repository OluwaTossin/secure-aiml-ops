"""
Application Configuration Settings
================================

Centralized configuration management for the Streamlit application.
Handles environment variables, API endpoints, and application settings.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, List
import streamlit as st

@dataclass
class AWSConfig:
    """AWS service configuration"""
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    region: str = "eu-west-1"
    s3_bucket: str = "secure-aiml-ops-data"
    ecr_repository: str = "455921291596.dkr.ecr.eu-west-1.amazonaws.com/secure-aiml-ops"
    
    def __post_init__(self):
        # Load from environment variables
        self.access_key_id = os.getenv("AWS_ACCESS_KEY_ID", self.access_key_id)
        self.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", self.secret_access_key)
        self.region = os.getenv("AWS_DEFAULT_REGION", self.region)
        self.s3_bucket = os.getenv("S3_BUCKET_NAME", self.s3_bucket)
        self.ecr_repository = os.getenv("ECR_REPOSITORY", self.ecr_repository)

@dataclass
class ModelConfig:
    """Model service configuration"""
    api_base_url: str = "https://api.secure-aiml.com"
    api_version: str = "v1"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    
    # Model-specific endpoints
    summarization_endpoint: str = "/summarize"
    anomaly_detection_endpoint: str = "/detect"
    sentiment_analysis_endpoint: str = "/sentiment"
    
    def __post_init__(self):
        self.api_key = os.getenv("MODEL_API_KEY", self.api_key)
        self.api_base_url = os.getenv("MODEL_ENDPOINT_URL", self.api_base_url)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "secure_aiml_ops"
    username: str = "postgres"
    password: Optional[str] = None
    
    def __post_init__(self):
        self.host = os.getenv("DB_HOST", self.host)
        self.port = int(os.getenv("DB_PORT", self.port))
        self.database = os.getenv("DB_NAME", self.database)
        self.username = os.getenv("DB_USER", self.username)
        self.password = os.getenv("DB_PASSWORD", self.password)
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class RedisConfig:
    """Redis cache configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    def __post_init__(self):
        self.host = os.getenv("REDIS_HOST", self.host)
        self.port = int(os.getenv("REDIS_PORT", self.port))
        self.password = os.getenv("REDIS_PASSWORD", self.password)

@dataclass
class SecurityConfig:
    """Security and authentication configuration"""
    secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_key_header: str = "X-API-Key"
    
    # CORS settings
    cors_origins: List[str] = None
    cors_credentials: bool = True
    
    def __post_init__(self):
        self.secret_key = os.getenv("SECRET_KEY", self.secret_key)
        if self.cors_origins is None:
            origins = os.getenv("CORS_ORIGINS", "*")
            self.cors_origins = [o.strip() for o in origins.split(",")]

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 8000
    
    # Prometheus settings
    prometheus_enabled: bool = True
    prometheus_endpoint: str = "/metrics"
    
    # Sentry settings
    sentry_dsn: Optional[str] = None
    sentry_environment: str = "production"
    
    def __post_init__(self):
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.sentry_dsn = os.getenv("SENTRY_DSN", self.sentry_dsn)
        self.sentry_environment = os.getenv("SENTRY_ENVIRONMENT", self.sentry_environment)

@dataclass
class StreamlitConfig:
    """Streamlit-specific configuration"""
    server_port: int = 8501
    server_address: str = "0.0.0.0"
    browser_gather_usage_stats: bool = False
    
    # UI settings
    theme_primary_color: str = "#1f77b4"
    theme_background_color: str = "#ffffff"
    theme_secondary_background_color: str = "#f0f2f6"
    theme_text_color: str = "#262730"
    
    def __post_init__(self):
        self.server_port = int(os.getenv("STREAMLIT_SERVER_PORT", self.server_port))
        self.server_address = os.getenv("STREAMLIT_SERVER_ADDRESS", self.server_address)
        self.browser_gather_usage_stats = os.getenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false").lower() == "true"

class AppConfig:
    """Main application configuration class"""
    
    def __init__(self):
        self.aws = AWSConfig()
        self.model = ModelConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.streamlit = StreamlitConfig()
        
        # Application metadata
        self.app_name = "Secure AI/ML Operations"
        self.app_version = "1.0.0"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
    
    def get_model_endpoint(self, model_type: str) -> str:
        """Get full endpoint URL for a specific model"""
        endpoint_map = {
            "summarization": self.model.summarization_endpoint,
            "anomaly_detection": self.model.anomaly_detection_endpoint,
            "sentiment_analysis": self.model.sentiment_analysis_endpoint
        }
        
        endpoint = endpoint_map.get(model_type, "")
        return f"{self.model.api_base_url}/{self.model.api_version}{endpoint}"
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return any errors"""
        errors = {}
        
        # Check required AWS settings for production
        if self.is_production:
            aws_errors = []
            if not self.aws.access_key_id:
                aws_errors.append("AWS_ACCESS_KEY_ID is required in production")
            if not self.aws.secret_access_key:
                aws_errors.append("AWS_SECRET_ACCESS_KEY is required in production")
            if aws_errors:
                errors["aws"] = aws_errors
        
        # Check model API configuration
        if not self.model.api_key and self.is_production:
            errors["model"] = ["MODEL_API_KEY is required in production"]
        
        # Check database configuration
        if not self.database.password and self.is_production:
            errors["database"] = ["DB_PASSWORD is required in production"]
        
        # Check security configuration
        if self.security.secret_key == "your-secret-key-change-in-production" and self.is_production:
            errors["security"] = ["SECRET_KEY must be changed in production"]
        
        return errors
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary (excluding sensitive data)"""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
            "debug": self.debug,
            "aws": {
                "region": self.aws.region,
                "s3_bucket": self.aws.s3_bucket,
                "ecr_repository": self.aws.ecr_repository
            },
            "model": {
                "api_base_url": self.model.api_base_url,
                "api_version": self.model.api_version,
                "timeout": self.model.timeout,
                "max_retries": self.model.max_retries
            },
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db
            },
            "streamlit": {
                "server_port": self.streamlit.server_port,
                "server_address": self.streamlit.server_address,
                "theme_primary_color": self.streamlit.theme_primary_color
            }
        }

# Global configuration instance
config = AppConfig()

# Streamlit configuration helper functions
def configure_streamlit_page():
    """Configure Streamlit page with application settings"""
    st.set_page_config(
        page_title=config.app_name,
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def get_theme_config() -> Dict[str, str]:
    """Get Streamlit theme configuration"""
    return {
        "primaryColor": config.streamlit.theme_primary_color,
        "backgroundColor": config.streamlit.theme_background_color,
        "secondaryBackgroundColor": config.streamlit.theme_secondary_background_color,
        "textColor": config.streamlit.theme_text_color
    }

def display_config_info():
    """Display configuration information in Streamlit sidebar"""
    with st.sidebar:
        with st.expander("ℹ️ Configuration Info"):
            st.write(f"**Environment:** {config.environment}")
            st.write(f"**Version:** {config.app_version}")
            st.write(f"**AWS Region:** {config.aws.region}")
            st.write(f"**Debug Mode:** {config.debug}")
            
            # Validate configuration
            errors = config.validate_config()
            if errors:
                st.error("⚠️ Configuration Issues:")
                for service, error_list in errors.items():
                    for error in error_list:
                        st.error(f"• {error}")
            else:
                st.success("✅ Configuration Valid")

# Environment-specific settings
class EnvironmentSettings:
    """Environment-specific configuration settings"""
    
    DEVELOPMENT = {
        "debug": True,
        "log_level": "DEBUG",
        "enable_metrics": False,
        "cors_origins": ["*"]
    }
    
    STAGING = {
        "debug": False,
        "log_level": "INFO",
        "enable_metrics": True,
        "cors_origins": ["https://staging.secure-aiml.com"]
    }
    
    PRODUCTION = {
        "debug": False,
        "log_level": "WARNING",
        "enable_metrics": True,
        "cors_origins": ["https://secure-aiml.com"]
    }
    
    @classmethod
    def get_settings(cls, environment: str) -> Dict:
        """Get settings for specific environment"""
        return getattr(cls, environment.upper(), cls.DEVELOPMENT)