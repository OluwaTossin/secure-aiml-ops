"""
Custom utility functions for Airflow DAGs in Secure AI/ML Operations

This module provides reusable utility functions for:
- S3 operations and data management
- Model artifacts handling
- Notification and alerting
- Data validation and quality checks
- AWS service integrations
"""

import json
import logging
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.hooks.ec2 import EC2Hook
from airflow.providers.amazon.aws.hooks.ecr import EcrHook

logger = logging.getLogger(__name__)

# Constants
DEFAULT_S3_BUCKET = "secure-aiml-ops-data"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ECR_REPO = "455921291596.dkr.ecr.eu-west-1.amazonaws.com/secure-aiml-ops"


class S3DataManager:
    """Utility class for S3 data operations."""
    
    def __init__(self, bucket_name: str = DEFAULT_S3_BUCKET):
        self.bucket_name = bucket_name
        self.s3_hook = S3Hook(aws_conn_id='aws_default')
    
    def upload_dataframe(self, df: pd.DataFrame, key: str, file_format: str = 'parquet') -> str:
        """
        Upload pandas DataFrame to S3 in specified format.
        
        Args:
            df: DataFrame to upload
            key: S3 key path
            file_format: Format to save ('parquet', 'csv', 'json')
            
        Returns:
            S3 key where data was saved
        """
        try:
            if file_format.lower() == 'parquet':
                # Save as parquet (more efficient for large datasets)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp_file:
                    df.to_parquet(tmp_file.name, index=False)
                    self.s3_hook.load_file(
                        filename=tmp_file.name,
                        key=key,
                        bucket_name=self.bucket_name,
                        replace=True
                    )
            
            elif file_format.lower() == 'csv':
                csv_data = df.to_csv(index=False)
                self.s3_hook.load_string(
                    string_data=csv_data,
                    key=key,
                    bucket_name=self.bucket_name,
                    replace=True
                )
            
            elif file_format.lower() == 'json':
                json_data = df.to_json(orient='records', indent=2)
                self.s3_hook.load_string(
                    string_data=json_data,
                    key=key,
                    bucket_name=self.bucket_name,
                    replace=True
                )
            
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"DataFrame uploaded to s3://{self.bucket_name}/{key}")
            return key
            
        except Exception as e:
            logger.error(f"Error uploading DataFrame to S3: {str(e)}")
            raise
    
    def read_dataframe(self, key: str, file_format: str = 'parquet') -> pd.DataFrame:
        """
        Read DataFrame from S3.
        
        Args:
            key: S3 key path
            file_format: Format to read ('parquet', 'csv', 'json')
            
        Returns:
            pandas DataFrame
        """
        try:
            if file_format.lower() == 'parquet':
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp_file:
                    self.s3_hook.download_file(
                        key=key,
                        bucket_name=self.bucket_name,
                        local_path=tmp_file.name
                    )
                    df = pd.read_parquet(tmp_file.name)
            
            elif file_format.lower() == 'csv':
                obj = self.s3_hook.get_key(key=key, bucket_name=self.bucket_name)
                csv_data = obj.get()['Body'].read().decode('utf-8')
                from io import StringIO
                df = pd.read_csv(StringIO(csv_data))
            
            elif file_format.lower() == 'json':
                obj = self.s3_hook.get_key(key=key, bucket_name=self.bucket_name)
                json_data = obj.get()['Body'].read().decode('utf-8')
                df = pd.read_json(json_data, orient='records')
            
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"DataFrame read from s3://{self.bucket_name}/{key}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading DataFrame from S3: {str(e)}")
            raise
    
    def check_data_freshness(self, key: str, max_age_hours: int = 24) -> bool:
        """
        Check if data in S3 is fresh (within specified hours).
        
        Args:
            key: S3 key path
            max_age_hours: Maximum age in hours
            
        Returns:
            True if data is fresh, False otherwise
        """
        try:
            if not self.s3_hook.check_for_key(key=key, bucket_name=self.bucket_name):
                return False
            
            obj = self.s3_hook.get_key(key=key, bucket_name=self.bucket_name)
            last_modified = obj.last_modified
            
            age = datetime.now(last_modified.tzinfo) - last_modified
            age_hours = age.total_seconds() / 3600
            
            is_fresh = age_hours <= max_age_hours
            logger.info(f"Data at {key} is {age_hours:.2f} hours old (fresh: {is_fresh})")
            
            return is_fresh
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {str(e)}")
            return False


class ModelArtifactManager:
    """Utility class for managing ML model artifacts."""
    
    def __init__(self, bucket_name: str = DEFAULT_S3_BUCKET):
        self.bucket_name = bucket_name
        self.s3_hook = S3Hook(aws_conn_id='aws_default')
    
    def save_model_metadata(self, model_path: str, metadata: Dict[str, Any]) -> str:
        """
        Save model metadata to S3.
        
        Args:
            model_path: Path where model is stored
            metadata: Dictionary containing model metadata
            
        Returns:
            S3 key where metadata was saved
        """
        try:
            # Add standard metadata fields
            metadata.update({
                'saved_at': datetime.now().isoformat(),
                'model_path': model_path,
                'metadata_version': '1.0'
            })
            
            # Construct metadata key
            metadata_key = f"{model_path}/metadata.json"
            
            # Save metadata
            self.s3_hook.load_string(
                string_data=json.dumps(metadata, indent=2, default=str),
                key=metadata_key,
                bucket_name=self.bucket_name,
                replace=True
            )
            
            logger.info(f"Model metadata saved to s3://{self.bucket_name}/{metadata_key}")
            return metadata_key
            
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")
            raise
    
    def load_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Load model metadata from S3.
        
        Args:
            model_path: Path where model is stored
            
        Returns:
            Dictionary containing model metadata
        """
        try:
            metadata_key = f"{model_path}/metadata.json"
            
            obj = self.s3_hook.get_key(key=metadata_key, bucket_name=self.bucket_name)
            metadata = json.loads(obj.get()['Body'].read().decode('utf-8'))
            
            logger.info(f"Model metadata loaded from s3://{self.bucket_name}/{metadata_key}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading model metadata: {str(e)}")
            raise
    
    def register_model_version(self, model_name: str, version: str, metadata: Dict[str, Any]) -> str:
        """
        Register a new model version in the model registry.
        
        Args:
            model_name: Name of the model
            version: Version identifier
            metadata: Model metadata
            
        Returns:
            Registry key where version was registered
        """
        try:
            registry_key = "models/registry/model_registry.json"
            
            # Load existing registry or create new one
            try:
                obj = self.s3_hook.get_key(key=registry_key, bucket_name=self.bucket_name)
                registry = json.loads(obj.get()['Body'].read().decode('utf-8'))
            except:
                registry = {
                    'created_at': datetime.now().isoformat(),
                    'models': {}
                }
            
            # Add model entry if not exists
            if model_name not in registry['models']:
                registry['models'][model_name] = {
                    'created_at': datetime.now().isoformat(),
                    'versions': {}
                }
            
            # Register new version
            registry['models'][model_name]['versions'][version] = {
                'registered_at': datetime.now().isoformat(),
                'metadata': metadata,
                'status': 'registered'
            }
            
            # Update registry
            registry['last_updated'] = datetime.now().isoformat()
            
            # Save updated registry
            self.s3_hook.load_string(
                string_data=json.dumps(registry, indent=2, default=str),
                key=registry_key,
                bucket_name=self.bucket_name,
                replace=True
            )
            
            logger.info(f"Model {model_name} version {version} registered in registry")
            return registry_key
            
        except Exception as e:
            logger.error(f"Error registering model version: {str(e)}")
            raise


class NotificationManager:
    """Utility class for managing notifications and alerts."""
    
    @staticmethod
    def send_slack_notification(webhook_url: str, message: str, channel: str = "#alerts") -> bool:
        """
        Send notification to Slack.
        
        Args:
            webhook_url: Slack webhook URL
            message: Message to send
            channel: Slack channel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import requests
            
            payload = {
                'channel': channel,
                'text': message,
                'username': 'Airflow Bot',
                'icon_emoji': ':robot_face:'
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent to {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return False
    
    @staticmethod
    def send_email_alert(subject: str, body: str, to_emails: List[str]) -> bool:
        """
        Send email alert using Airflow's email functionality.
        
        Args:
            subject: Email subject
            body: Email body
            to_emails: List of recipient emails
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from airflow.utils.email import send_email
            
            send_email(
                to=to_emails,
                subject=subject,
                html_content=body
            )
            
            logger.info(f"Email alert sent to {', '.join(to_emails)}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False


class DataQualityChecker:
    """Utility class for data quality validation."""
    
    @staticmethod
    def check_dataframe_quality(df: pd.DataFrame, checks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform data quality checks on a DataFrame.
        
        Args:
            df: DataFrame to check
            checks: Dictionary specifying checks to perform
            
        Returns:
            Dictionary containing check results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'checks': {}
        }
        
        try:
            # Check for null values
            if checks.get('null_check', True):
                null_counts = df.isnull().sum()
                null_percentages = (null_counts / len(df)) * 100
                
                results['checks']['null_check'] = {
                    'status': 'passed' if null_percentages.max() < checks.get('max_null_percent', 10) else 'failed',
                    'null_counts': null_counts.to_dict(),
                    'null_percentages': null_percentages.to_dict(),
                    'max_null_percentage': float(null_percentages.max())
                }
            
            # Check for duplicates
            if checks.get('duplicate_check', True):
                duplicate_count = df.duplicated().sum()
                duplicate_percentage = (duplicate_count / len(df)) * 100
                
                results['checks']['duplicate_check'] = {
                    'status': 'passed' if duplicate_percentage < checks.get('max_duplicate_percent', 5) else 'failed',
                    'duplicate_count': int(duplicate_count),
                    'duplicate_percentage': float(duplicate_percentage)
                }
            
            # Check column schemas
            if checks.get('schema_check', True) and 'expected_columns' in checks:
                expected_columns = set(checks['expected_columns'])
                actual_columns = set(df.columns)
                
                missing_columns = expected_columns - actual_columns
                extra_columns = actual_columns - expected_columns
                
                results['checks']['schema_check'] = {
                    'status': 'passed' if len(missing_columns) == 0 and len(extra_columns) == 0 else 'failed',
                    'missing_columns': list(missing_columns),
                    'extra_columns': list(extra_columns),
                    'column_match': len(missing_columns) == 0 and len(extra_columns) == 0
                }
            
            # Check data ranges
            if checks.get('range_check', False) and 'column_ranges' in checks:
                range_results = {}
                for column, (min_val, max_val) in checks['column_ranges'].items():
                    if column in df.columns:
                        col_min = df[column].min()
                        col_max = df[column].max()
                        
                        range_results[column] = {
                            'status': 'passed' if min_val <= col_min and col_max <= max_val else 'failed',
                            'actual_min': float(col_min) if pd.notnull(col_min) else None,
                            'actual_max': float(col_max) if pd.notnull(col_max) else None,
                            'expected_min': min_val,
                            'expected_max': max_val
                        }
                
                results['checks']['range_check'] = range_results
            
            # Overall quality score
            passed_checks = sum(1 for check in results['checks'].values() 
                              if isinstance(check, dict) and check.get('status') == 'passed')
            total_checks = len(results['checks'])
            
            results['quality_score'] = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
            results['overall_status'] = 'passed' if results['quality_score'] >= checks.get('min_quality_score', 80) else 'failed'
            
            logger.info(f"Data quality check completed with score: {results['quality_score']:.1f}%")
            return results
            
        except Exception as e:
            logger.error(f"Error performing data quality checks: {str(e)}")
            results['error'] = str(e)
            results['overall_status'] = 'error'
            return results


class AWSResourceManager:
    """Utility class for AWS resource management."""
    
    def __init__(self, region: str = DEFAULT_REGION):
        self.region = region
    
    def get_ec2_instances_by_tag(self, tag_key: str, tag_value: str) -> List[Dict[str, Any]]:
        """
        Get EC2 instances by tag.
        
        Args:
            tag_key: Tag key to filter by
            tag_value: Tag value to filter by
            
        Returns:
            List of instance information dictionaries
        """
        try:
            ec2_hook = EC2Hook(region_name=self.region)
            client = ec2_hook.get_conn()
            
            response = client.describe_instances(
                Filters=[
                    {'Name': f'tag:{tag_key}', 'Values': [tag_value]},
                    {'Name': 'instance-state-name', 'Values': ['running', 'stopped']}
                ]
            )
            
            instances = []
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instances.append({
                        'instance_id': instance['InstanceId'],
                        'instance_type': instance['InstanceType'],
                        'state': instance['State']['Name'],
                        'launch_time': instance['LaunchTime'],
                        'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                    })
            
            logger.info(f"Found {len(instances)} EC2 instances with tag {tag_key}={tag_value}")
            return instances
            
        except Exception as e:
            logger.error(f"Error getting EC2 instances: {str(e)}")
            return []
    
    def get_ecr_image_tags(self, repository_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get ECR image tags for a repository.
        
        Args:
            repository_name: Name of the ECR repository
            max_results: Maximum number of results to return
            
        Returns:
            List of image information dictionaries
        """
        try:
            ecr_hook = EcrHook(region_name=self.region)
            client = ecr_hook.get_conn()
            
            response = client.describe_images(
                repositoryName=repository_name,
                maxResults=max_results
            )
            
            images = []
            for image in response['imageDetails']:
                images.append({
                    'image_digest': image['imageDigest'],
                    'image_tags': image.get('imageTags', []),
                    'image_size_bytes': image.get('imageSizeInBytes', 0),
                    'image_pushed_at': image.get('imagePushedAt'),
                    'registry_id': image.get('registryId')
                })
            
            logger.info(f"Found {len(images)} images in ECR repository {repository_name}")
            return images
            
        except Exception as e:
            logger.error(f"Error getting ECR images: {str(e)}")
            return []


# Utility functions for common operations

def create_execution_date_partition(execution_date: datetime, base_path: str) -> str:
    """
    Create a partitioned path based on execution date.
    
    Args:
        execution_date: The execution date
        base_path: Base path for partitioning
        
    Returns:
        Partitioned path string
    """
    year = execution_date.strftime('%Y')
    month = execution_date.strftime('%m')
    day = execution_date.strftime('%d')
    
    return f"{base_path}/year={year}/month={month}/day={day}"


def generate_unique_task_id(base_name: str, timestamp: Optional[datetime] = None) -> str:
    """
    Generate a unique task ID with timestamp.
    
    Args:
        base_name: Base name for the task
        timestamp: Optional timestamp (uses current time if not provided)
        
    Returns:
        Unique task ID string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp_str}"


def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize object to JSON, handling datetime and other non-serializable types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string
    """
    def default_serializer(o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    
    return json.dumps(obj, default=default_serializer, indent=2)


# Export main classes and functions
__all__ = [
    'S3DataManager',
    'ModelArtifactManager', 
    'NotificationManager',
    'DataQualityChecker',
    'AWSResourceManager',
    'create_execution_date_partition',
    'generate_unique_task_id',
    'safe_json_serialize'
]