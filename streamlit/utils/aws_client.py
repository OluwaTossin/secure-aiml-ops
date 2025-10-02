"""
AWS Service Client
================

Unified AWS service client for S3, ECR, CloudWatch, and other AWS services.
Provides simplified interfaces for common operations.
"""

import boto3
import botocore
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

class AWSClient:
    """Unified AWS service client"""
    
    def __init__(self, 
                 access_key_id: Optional[str] = None,
                 secret_access_key: Optional[str] = None,
                 region: str = "eu-west-1"):
        """
        Initialize AWS client
        
        Args:
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            region: AWS region
        """
        self.region = region
        self.session = None
        self._clients = {}
        
        try:
            # Create boto3 session
            if access_key_id and secret_access_key:
                self.session = boto3.Session(
                    aws_access_key_id=access_key_id,
                    aws_secret_access_key=secret_access_key,
                    region_name=region
                )
            else:
                # Use default credentials (environment, IAM role, etc.)
                self.session = boto3.Session(region_name=region)
                
            # Test credentials
            sts_client = self.session.client('sts')
            self.account_id = sts_client.get_caller_identity()['Account']
            
            logger.info(f"AWS client initialized for account {self.account_id} in region {region}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            self.session = None
            self.account_id = None
        except Exception as e:
            logger.error(f"Failed to initialize AWS client: {e}")
            self.session = None
            self.account_id = None
    
    def get_client(self, service_name: str):
        """Get or create AWS service client"""
        if not self.session:
            raise Exception("AWS session not initialized")
            
        if service_name not in self._clients:
            self._clients[service_name] = self.session.client(service_name)
        
        return self._clients[service_name]
    
    def is_connected(self) -> bool:
        """Check if AWS connection is established"""
        return self.session is not None and self.account_id is not None

class S3Manager:
    """S3 operations manager"""
    
    def __init__(self, aws_client: AWSClient):
        self.aws_client = aws_client
        self.s3_client = None
        
        if aws_client.is_connected():
            try:
                self.s3_client = aws_client.get_client('s3')
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
    
    def list_buckets(self) -> List[Dict[str, Any]]:
        """List all S3 buckets"""
        if not self.s3_client:
            return []
        
        try:
            response = self.s3_client.list_buckets()
            return response.get('Buckets', [])
        except ClientError as e:
            logger.error(f"Failed to list buckets: {e}")
            return []
    
    def list_objects(self, bucket_name: str, prefix: str = "") -> List[Dict[str, Any]]:
        """List objects in S3 bucket"""
        if not self.s3_client:
            return []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            return response.get('Contents', [])
        except ClientError as e:
            logger.error(f"Failed to list objects in bucket {bucket_name}: {e}")
            return []
    
    def get_object_metadata(self, bucket_name: str, key: str) -> Optional[Dict[str, Any]]:
        """Get object metadata"""
        if not self.s3_client:
            return None
        
        try:
            response = self.s3_client.head_object(Bucket=bucket_name, Key=key)
            return {
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {})
            }
        except ClientError as e:
            logger.error(f"Failed to get metadata for {key}: {e}")
            return None
    
    def download_object(self, bucket_name: str, key: str, local_path: str) -> bool:
        """Download object from S3"""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.download_file(bucket_name, key, local_path)
            return True
        except ClientError as e:
            logger.error(f"Failed to download {key}: {e}")
            return False
    
    def upload_object(self, bucket_name: str, key: str, local_path: str) -> bool:
        """Upload object to S3"""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.upload_file(local_path, bucket_name, key)
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {key}: {e}")
            return False

class ECRManager:
    """ECR operations manager"""
    
    def __init__(self, aws_client: AWSClient):
        self.aws_client = aws_client
        self.ecr_client = None
        
        if aws_client.is_connected():
            try:
                self.ecr_client = aws_client.get_client('ecr')
            except Exception as e:
                logger.error(f"Failed to initialize ECR client: {e}")
    
    def list_repositories(self) -> List[Dict[str, Any]]:
        """List ECR repositories"""
        if not self.ecr_client:
            return []
        
        try:
            response = self.ecr_client.describe_repositories()
            return response.get('repositories', [])
        except ClientError as e:
            logger.error(f"Failed to list repositories: {e}")
            return []
    
    def list_images(self, repository_name: str) -> List[Dict[str, Any]]:
        """List images in ECR repository"""
        if not self.ecr_client:
            return []
        
        try:
            response = self.ecr_client.list_images(repositoryName=repository_name)
            return response.get('imageIds', [])
        except ClientError as e:
            logger.error(f"Failed to list images in {repository_name}: {e}")
            return []
    
    def get_image_details(self, repository_name: str) -> List[Dict[str, Any]]:
        """Get detailed image information"""
        if not self.ecr_client:
            return []
        
        try:
            response = self.ecr_client.describe_images(repositoryName=repository_name)
            return response.get('imageDetails', [])
        except ClientError as e:
            logger.error(f"Failed to get image details for {repository_name}: {e}")
            return []

class CloudWatchManager:
    """CloudWatch operations manager"""
    
    def __init__(self, aws_client: AWSClient):
        self.aws_client = aws_client
        self.cloudwatch_client = None
        
        if aws_client.is_connected():
            try:
                self.cloudwatch_client = aws_client.get_client('cloudwatch')
            except Exception as e:
                logger.error(f"Failed to initialize CloudWatch client: {e}")
    
    def get_metric_data(self, 
                       metric_name: str, 
                       namespace: str,
                       dimensions: List[Dict[str, str]] = None,
                       start_time: datetime = None,
                       end_time: datetime = None,
                       statistic: str = 'Average',
                       period: int = 300) -> List[Dict[str, Any]]:
        """Get CloudWatch metric data"""
        if not self.cloudwatch_client:
            return []
        
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.utcnow()
        
        try:
            response = self.cloudwatch_client.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions or [],
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=[statistic]
            )
            return response.get('Datapoints', [])
        except ClientError as e:
            logger.error(f"Failed to get metric data for {metric_name}: {e}")
            return []
    
    def list_metrics(self, namespace: str = None) -> List[Dict[str, Any]]:
        """List available metrics"""
        if not self.cloudwatch_client:
            return []
        
        try:
            params = {}
            if namespace:
                params['Namespace'] = namespace
                
            response = self.cloudwatch_client.list_metrics(**params)
            return response.get('Metrics', [])
        except ClientError as e:
            logger.error(f"Failed to list metrics: {e}")
            return []

class IAMManager:
    """IAM operations manager"""
    
    def __init__(self, aws_client: AWSClient):
        self.aws_client = aws_client
        self.iam_client = None
        
        if aws_client.is_connected():
            try:
                self.iam_client = aws_client.get_client('iam')
            except Exception as e:
                logger.error(f"Failed to initialize IAM client: {e}")
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """List IAM roles"""
        if not self.iam_client:
            return []
        
        try:
            response = self.iam_client.list_roles()
            return response.get('Roles', [])
        except ClientError as e:
            logger.error(f"Failed to list roles: {e}")
            return []
    
    def get_role(self, role_name: str) -> Optional[Dict[str, Any]]:
        """Get IAM role details"""
        if not self.iam_client:
            return None
        
        try:
            response = self.iam_client.get_role(RoleName=role_name)
            return response.get('Role')
        except ClientError as e:
            logger.error(f"Failed to get role {role_name}: {e}")
            return None

class AWSResourceMonitor:
    """AWS resource monitoring and dashboard"""
    
    def __init__(self, aws_client: AWSClient):
        self.aws_client = aws_client
        self.s3_manager = S3Manager(aws_client)
        self.ecr_manager = ECRManager(aws_client)
        self.cloudwatch_manager = CloudWatchManager(aws_client)
        self.iam_manager = IAMManager(aws_client)
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get AWS account resource summary"""
        summary = {
            'account_id': self.aws_client.account_id,
            'region': self.aws_client.region,
            'connected': self.aws_client.is_connected(),
            'buckets': [],
            'repositories': [],
            'roles': []
        }
        
        if self.aws_client.is_connected():
            summary['buckets'] = self.s3_manager.list_buckets()
            summary['repositories'] = self.ecr_manager.list_repositories()
            summary['roles'] = self.iam_manager.list_roles()
        
        return summary
    
    def display_resource_dashboard(self):
        """Display AWS resource dashboard in Streamlit"""
        st.markdown("### 🌊 AWS Resources Overview")
        
        if not self.aws_client.is_connected():
            st.error("⚠️ AWS connection not established. Please check your credentials.")
            return
        
        # Account info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Account ID", self.aws_client.account_id or "Unknown")
        
        with col2:
            st.metric("Region", self.aws_client.region)
        
        with col3:
            status = "🟢 Connected" if self.aws_client.is_connected() else "🔴 Disconnected"
            st.metric("Status", status)
        
        # Resource tabs
        tab1, tab2, tab3 = st.tabs(["📦 S3 Buckets", "🐳 ECR Repositories", "📊 CloudWatch Metrics"])
        
        with tab1:
            st.markdown("#### S3 Buckets")
            
            buckets = self.s3_manager.list_buckets()
            if buckets:
                bucket_data = []
                for bucket in buckets:
                    bucket_data.append({
                        'Name': bucket['Name'],
                        'Creation Date': bucket['CreationDate'].strftime('%Y-%m-%d %H:%M:%S'),
                        'Region': self.aws_client.region
                    })
                
                df = pd.DataFrame(bucket_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Bucket details
                selected_bucket = st.selectbox("Select bucket for details", [b['Name'] for b in buckets])
                
                if selected_bucket:
                    objects = self.s3_manager.list_objects(selected_bucket)
                    st.write(f"**Objects in {selected_bucket}:** {len(objects)}")
                    
                    if objects:
                        object_data = []
                        for obj in objects[:10]:  # Show first 10 objects
                            object_data.append({
                                'Key': obj['Key'],
                                'Size': f"{obj['Size']:,} bytes",
                                'Last Modified': obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                            })
                        
                        st.dataframe(pd.DataFrame(object_data), use_container_width=True, hide_index=True)
            else:
                st.info("No S3 buckets found.")
        
        with tab2:
            st.markdown("#### ECR Repositories")
            
            repositories = self.ecr_manager.list_repositories()
            if repositories:
                repo_data = []
                for repo in repositories:
                    repo_data.append({
                        'Repository Name': repo['repositoryName'],
                        'URI': repo['repositoryUri'],
                        'Created': repo['createdAt'].strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                df = pd.DataFrame(repo_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Repository images
                selected_repo = st.selectbox("Select repository for images", [r['repositoryName'] for r in repositories])
                
                if selected_repo:
                    images = self.ecr_manager.get_image_details(selected_repo)
                    st.write(f"**Images in {selected_repo}:** {len(images)}")
                    
                    if images:
                        image_data = []
                        for img in images[:10]:  # Show first 10 images
                            tags = img.get('imageTags', ['<untagged>'])
                            image_data.append({
                                'Tags': ', '.join(tags),
                                'Size': f"{img.get('imageSizeInBytes', 0):,} bytes",
                                'Pushed': img['imagePushedAt'].strftime('%Y-%m-%d %H:%M:%S')
                            })
                        
                        st.dataframe(pd.DataFrame(image_data), use_container_width=True, hide_index=True)
            else:
                st.info("No ECR repositories found.")
        
        with tab3:
            st.markdown("#### CloudWatch Metrics")
            
            # Sample metrics
            metrics = self.cloudwatch_manager.list_metrics('AWS/EC2')
            
            if metrics:
                st.write(f"Available metrics: {len(metrics)}")
                
                # Show sample metric data
                metric_names = list(set(m['MetricName'] for m in metrics[:20]))
                selected_metric = st.selectbox("Select metric", metric_names)
                
                if selected_metric:
                    # Get metric data for the last hour
                    data = self.cloudwatch_manager.get_metric_data(
                        metric_name=selected_metric,
                        namespace='AWS/EC2'
                    )
                    
                    if data:
                        df = pd.DataFrame(data)
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                        df = df.sort_values('Timestamp')
                        
                        st.line_chart(df.set_index('Timestamp')['Average'])
                    else:
                        st.info("No data available for this metric.")
            else:
                st.info("No CloudWatch metrics found.")

# Factory function to create AWS client from config
def create_aws_client_from_config(config) -> AWSClient:
    """Create AWS client from application configuration"""
    return AWSClient(
        access_key_id=config.aws.access_key_id,
        secret_access_key=config.aws.secret_access_key,
        region=config.aws.region
    )