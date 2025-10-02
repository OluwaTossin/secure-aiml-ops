"""
Custom Airflow operators for Secure AI/ML Operations

This module provides custom operators for:
- ML model training and evaluation
- Data validation and quality checks
- AWS service integrations
- Model deployment operations
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from airflow.models.baseoperator import BaseOperator
from airflow.utils.context import Context
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.hooks.ec2 import EC2Hook

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class S3DataValidationOperator(BaseOperator):
    """
    Custom operator for validating data in S3 before processing.
    """
    
    template_fields = ['s3_key', 'bucket_name']
    
    def __init__(
        self,
        s3_key: str,
        bucket_name: str,
        validation_rules: Dict[str, Any],
        aws_conn_id: str = 'aws_default',
        fail_on_validation_error: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.s3_key = s3_key
        self.bucket_name = bucket_name
        self.validation_rules = validation_rules
        self.aws_conn_id = aws_conn_id
        self.fail_on_validation_error = fail_on_validation_error
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Execute data validation."""
        s3_hook = S3Hook(aws_conn_id=self.aws_conn_id)
        
        try:
            # Check if file exists
            if not s3_hook.check_for_key(key=self.s3_key, bucket_name=self.bucket_name):
                raise FileNotFoundError(f"File not found: s3://{self.bucket_name}/{self.s3_key}")
            
            # Get file metadata
            obj = s3_hook.get_key(key=self.s3_key, bucket_name=self.bucket_name)
            file_size = obj.content_length
            last_modified = obj.last_modified
            
            validation_results = {
                'file_path': f"s3://{self.bucket_name}/{self.s3_key}",
                'file_size_bytes': file_size,
                'last_modified': last_modified.isoformat(),
                'validation_timestamp': datetime.now().isoformat(),
                'checks': {}
            }
            
            # File size validation
            if 'min_size_bytes' in self.validation_rules:
                min_size = self.validation_rules['min_size_bytes']
                size_check_passed = file_size >= min_size
                validation_results['checks']['file_size'] = {
                    'passed': size_check_passed,
                    'actual_size': file_size,
                    'min_required': min_size
                }
                
                if not size_check_passed and self.fail_on_validation_error:
                    raise ValueError(f"File size {file_size} bytes is below minimum {min_size} bytes")
            
            # File age validation
            if 'max_age_hours' in self.validation_rules:
                max_age = self.validation_rules['max_age_hours']
                age_hours = (datetime.now(last_modified.tzinfo) - last_modified).total_seconds() / 3600
                age_check_passed = age_hours <= max_age
                
                validation_results['checks']['file_age'] = {
                    'passed': age_check_passed,
                    'age_hours': age_hours,
                    'max_allowed_hours': max_age
                }
                
                if not age_check_passed and self.fail_on_validation_error:
                    raise ValueError(f"File age {age_hours:.2f} hours exceeds maximum {max_age} hours")
            
            # Content validation (if CSV/JSON)
            if self.validation_rules.get('validate_content', False):
                try:
                    if self.s3_key.endswith('.csv'):
                        # Read and validate CSV structure
                        obj_content = obj.get()['Body'].read().decode('utf-8')
                        from io import StringIO
                        df = pd.read_csv(StringIO(obj_content), nrows=5)  # Sample first 5 rows
                        
                        validation_results['checks']['content'] = {
                            'passed': True,
                            'format': 'csv',
                            'columns': list(df.columns),
                            'sample_rows': len(df)
                        }
                        
                        # Validate expected columns
                        if 'expected_columns' in self.validation_rules:
                            expected_cols = set(self.validation_rules['expected_columns'])
                            actual_cols = set(df.columns)
                            columns_match = expected_cols.issubset(actual_cols)
                            
                            validation_results['checks']['content']['columns_match'] = columns_match
                            validation_results['checks']['content']['missing_columns'] = list(expected_cols - actual_cols)
                            
                            if not columns_match and self.fail_on_validation_error:
                                raise ValueError(f"Missing columns: {list(expected_cols - actual_cols)}")
                    
                    elif self.s3_key.endswith('.json'):
                        # Validate JSON structure
                        obj_content = obj.get()['Body'].read().decode('utf-8')
                        json_data = json.loads(obj_content)
                        
                        validation_results['checks']['content'] = {
                            'passed': True,
                            'format': 'json',
                            'valid_json': True,
                            'type': type(json_data).__name__
                        }
                        
                        if isinstance(json_data, list) and json_data:
                            validation_results['checks']['content']['sample_keys'] = list(json_data[0].keys()) if isinstance(json_data[0], dict) else []
                    
                except Exception as e:
                    validation_results['checks']['content'] = {
                        'passed': False,
                        'error': str(e)
                    }
                    
                    if self.fail_on_validation_error:
                        raise ValueError(f"Content validation failed: {e}")
            
            # Overall validation status
            all_passed = all(check.get('passed', True) for check in validation_results['checks'].values())
            validation_results['overall_status'] = 'passed' if all_passed else 'failed'
            
            logger.info(f"Data validation {'passed' if all_passed else 'failed'} for {self.s3_key}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            raise


class ModelEvaluationOperator(BaseOperator):
    """
    Custom operator for evaluating ML model performance.
    """
    
    template_fields = ['model_path', 'test_data_path']
    
    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        model_type: str,
        evaluation_metrics: List[str],
        bucket_name: str,
        output_path: str,
        aws_conn_id: str = 'aws_default',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.model_type = model_type
        self.evaluation_metrics = evaluation_metrics
        self.bucket_name = bucket_name
        self.output_path = output_path
        self.aws_conn_id = aws_conn_id
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Execute model evaluation."""
        s3_hook = S3Hook(aws_conn_id=self.aws_conn_id)
        
        try:
            # For this demo, we'll simulate model evaluation
            # In a real implementation, you would load the actual model and test data
            
            evaluation_results = {
                'model_path': self.model_path,
                'test_data_path': self.test_data_path,
                'model_type': self.model_type,
                'evaluation_timestamp': datetime.now().isoformat(),
                'metrics': {}
            }
            
            # Simulate different metrics based on model type
            if self.model_type == 'text_summarization':
                if 'bleu' in self.evaluation_metrics:
                    evaluation_results['metrics']['bleu'] = np.random.uniform(0.6, 0.8)
                if 'rouge' in self.evaluation_metrics:
                    evaluation_results['metrics']['rouge'] = np.random.uniform(0.65, 0.85)
                if 'meteor' in self.evaluation_metrics:
                    evaluation_results['metrics']['meteor'] = np.random.uniform(0.55, 0.75)
            
            elif self.model_type == 'anomaly_detection':
                if 'precision' in self.evaluation_metrics:
                    evaluation_results['metrics']['precision'] = np.random.uniform(0.8, 0.95)
                if 'recall' in self.evaluation_metrics:
                    evaluation_results['metrics']['recall'] = np.random.uniform(0.75, 0.9)
                if 'f1_score' in self.evaluation_metrics:
                    precision = evaluation_results['metrics'].get('precision', 0.85)
                    recall = evaluation_results['metrics'].get('recall', 0.82)
                    evaluation_results['metrics']['f1_score'] = 2 * (precision * recall) / (precision + recall)
                if 'auc_roc' in self.evaluation_metrics:
                    evaluation_results['metrics']['auc_roc'] = np.random.uniform(0.85, 0.95)
            
            # Add overall performance assessment
            avg_score = np.mean(list(evaluation_results['metrics'].values()))
            evaluation_results['overall_score'] = avg_score
            evaluation_results['performance_grade'] = (
                'excellent' if avg_score >= 0.9 else
                'good' if avg_score >= 0.8 else
                'fair' if avg_score >= 0.7 else
                'poor'
            )
            
            # Save evaluation results
            output_key = f"{self.output_path}/evaluation_results.json"
            s3_hook.load_string(
                string_data=json.dumps(evaluation_results, indent=2, default=str),
                key=output_key,
                bucket_name=self.bucket_name,
                replace=True
            )
            
            logger.info(f"Model evaluation completed. Overall score: {avg_score:.3f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise


class AWSResourceCheckOperator(BaseOperator):
    """
    Custom operator for checking AWS resource status and health.
    """
    
    def __init__(
        self,
        resource_type: str,
        resource_identifier: str,
        check_type: str = 'status',
        aws_conn_id: str = 'aws_default',
        region_name: str = 'eu-west-1',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.resource_type = resource_type
        self.resource_identifier = resource_identifier
        self.check_type = check_type
        self.aws_conn_id = aws_conn_id
        self.region_name = region_name
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Execute AWS resource check."""
        try:
            check_results = {
                'resource_type': self.resource_type,
                'resource_identifier': self.resource_identifier,
                'check_type': self.check_type,
                'check_timestamp': datetime.now().isoformat(),
                'status': 'unknown'
            }
            
            if self.resource_type == 'ec2_instance':
                ec2_hook = EC2Hook(region_name=self.region_name)
                client = ec2_hook.get_conn()
                
                response = client.describe_instances(
                    InstanceIds=[self.resource_identifier]
                )
                
                if response['Reservations']:
                    instance = response['Reservations'][0]['Instances'][0]
                    check_results.update({
                        'status': instance['State']['Name'],
                        'instance_type': instance['InstanceType'],
                        'launch_time': instance['LaunchTime'].isoformat(),
                        'availability_zone': instance['Placement']['AvailabilityZone']
                    })
                else:
                    check_results['status'] = 'not_found'
            
            elif self.resource_type == 's3_bucket':
                s3_hook = S3Hook(aws_conn_id=self.aws_conn_id)
                
                try:
                    bucket_exists = s3_hook.check_for_bucket(bucket_name=self.resource_identifier)
                    if bucket_exists:
                        check_results['status'] = 'exists'
                        
                        # Get bucket size if requested
                        if self.check_type == 'detailed':
                            try:
                                bucket_size = s3_hook.get_bucket_size(bucket_name=self.resource_identifier)
                                check_results['size_bytes'] = bucket_size
                                check_results['size_gb'] = round(bucket_size / (1024**3), 2) if bucket_size else 0
                            except:
                                pass
                    else:
                        check_results['status'] = 'not_found'
                except Exception as e:
                    check_results['status'] = 'error'
                    check_results['error'] = str(e)
            
            elif self.resource_type == 'ecr_repository':
                import boto3
                ecr_client = boto3.client('ecr', region_name=self.region_name)
                
                try:
                    response = ecr_client.describe_repositories(
                        repositoryNames=[self.resource_identifier]
                    )
                    
                    if response['repositories']:
                        repo = response['repositories'][0]
                        check_results.update({
                            'status': 'exists',
                            'repository_uri': repo['repositoryUri'],
                            'created_at': repo['createdAt'].isoformat(),
                            'registry_id': repo['registryId']
                        })
                        
                        # Get image count if detailed check
                        if self.check_type == 'detailed':
                            try:
                                images_response = ecr_client.describe_images(
                                    repositoryName=self.resource_identifier
                                )
                                check_results['image_count'] = len(images_response['imageDetails'])
                            except:
                                pass
                    else:
                        check_results['status'] = 'not_found'
                        
                except ecr_client.exceptions.RepositoryNotFoundException:
                    check_results['status'] = 'not_found'
                except Exception as e:
                    check_results['status'] = 'error'
                    check_results['error'] = str(e)
            
            else:
                raise ValueError(f"Unsupported resource type: {self.resource_type}")
            
            logger.info(f"Resource check completed for {self.resource_type}:{self.resource_identifier} - Status: {check_results['status']}")
            return check_results
            
        except Exception as e:
            logger.error(f"Error during AWS resource check: {str(e)}")
            raise


class DataQualityCheckOperator(BaseOperator):
    """
    Custom operator for comprehensive data quality checks.
    """
    
    template_fields = ['data_path', 'output_path']
    
    def __init__(
        self,
        data_path: str,
        quality_rules: Dict[str, Any],
        bucket_name: str,
        output_path: str,
        file_format: str = 'parquet',
        aws_conn_id: str = 'aws_default',
        fail_on_poor_quality: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.quality_rules = quality_rules
        self.bucket_name = bucket_name
        self.output_path = output_path
        self.file_format = file_format
        self.aws_conn_id = aws_conn_id
        self.fail_on_poor_quality = fail_on_poor_quality
    
    def execute(self, context: Context) -> Dict[str, Any]:
        """Execute data quality checks."""
        s3_hook = S3Hook(aws_conn_id=self.aws_conn_id)
        
        try:
            # For demo purposes, simulate reading data and performing quality checks
            # In a real implementation, you would read the actual data file
            
            quality_report = {
                'data_path': self.data_path,
                'check_timestamp': datetime.now().isoformat(),
                'file_format': self.file_format,
                'rules_applied': self.quality_rules,
                'checks': {}
            }
            
            # Simulate data statistics
            simulated_stats = {
                'total_rows': np.random.randint(1000, 10000),
                'total_columns': np.random.randint(5, 20),
                'null_percentage': np.random.uniform(0, 10),
                'duplicate_percentage': np.random.uniform(0, 5),
                'unique_values_ratio': np.random.uniform(0.6, 1.0)
            }
            
            quality_report['data_statistics'] = simulated_stats
            
            # Apply quality rules
            for rule_name, rule_config in self.quality_rules.items():
                if rule_name == 'null_threshold':
                    threshold = rule_config.get('max_percentage', 10)
                    passed = simulated_stats['null_percentage'] <= threshold
                    quality_report['checks']['null_check'] = {
                        'rule': f"Null percentage should be <= {threshold}%",
                        'actual_value': simulated_stats['null_percentage'],
                        'threshold': threshold,
                        'passed': passed
                    }
                
                elif rule_name == 'duplicate_threshold':
                    threshold = rule_config.get('max_percentage', 5)
                    passed = simulated_stats['duplicate_percentage'] <= threshold
                    quality_report['checks']['duplicate_check'] = {
                        'rule': f"Duplicate percentage should be <= {threshold}%",
                        'actual_value': simulated_stats['duplicate_percentage'],
                        'threshold': threshold,
                        'passed': passed
                    }
                
                elif rule_name == 'completeness_threshold':
                    threshold = rule_config.get('min_ratio', 0.8)
                    passed = simulated_stats['unique_values_ratio'] >= threshold
                    quality_report['checks']['completeness_check'] = {
                        'rule': f"Unique values ratio should be >= {threshold}",
                        'actual_value': simulated_stats['unique_values_ratio'],
                        'threshold': threshold,
                        'passed': passed
                    }
            
            # Calculate overall quality score
            passed_checks = sum(1 for check in quality_report['checks'].values() if check['passed'])
            total_checks = len(quality_report['checks'])
            quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
            
            quality_report['quality_score'] = quality_score
            quality_report['overall_status'] = 'passed' if quality_score >= 80 else 'failed'
            quality_report['passed_checks'] = passed_checks
            quality_report['total_checks'] = total_checks
            
            # Save quality report
            output_key = f"{self.output_path}/quality_report.json"
            s3_hook.load_string(
                string_data=json.dumps(quality_report, indent=2, default=str),
                key=output_key,
                bucket_name=self.bucket_name,
                replace=True
            )
            
            # Fail if quality is poor and configured to do so
            if self.fail_on_poor_quality and quality_report['overall_status'] == 'failed':
                raise ValueError(f"Data quality check failed with score {quality_score:.1f}%")
            
            logger.info(f"Data quality check completed with score: {quality_score:.1f}%")
            return quality_report
            
        except Exception as e:
            logger.error(f"Error during data quality check: {str(e)}")
            raise