"""
Monitoring and Alerting DAG for Secure AI/ML Operations

This DAG handles:
1. Model performance monitoring
2. Infrastructure health checks
3. Data quality monitoring
4. Alert generation and notification
5. Automated reporting
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.hooks.cloudwatch import CloudWatchHook
from airflow.sensors.s3 import S3KeySensor
from airflow.utils.dates import days_ago

import json
import logging
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DAG Configuration
DAG_ID = "monitoring_alerting_pipeline"
SCHEDULE_INTERVAL = timedelta(hours=1)  # Run every hour
S3_BUCKET = "secure-aiml-ops-data"
MONITORING_PREFIX = "monitoring"
AWS_REGION = "eu-west-1"

# Alert thresholds
ALERT_THRESHOLDS = {
    'model_accuracy_drop': 0.05,  # 5% drop from baseline
    'api_response_time': 5000,     # 5 seconds
    'error_rate': 0.05,            # 5% error rate
    'data_volume_drop': 0.3,       # 30% drop in data volume
    'storage_usage': 0.85,         # 85% storage usage
}

# Default arguments
default_args = {
    'owner': 'secure-aiml-ops',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}


def collect_model_metrics(**context) -> Dict[str, Any]:
    """
    Collect model performance metrics from various sources.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date']
    
    metrics = {
        'timestamp': execution_date.isoformat(),
        'collection_date': datetime.now().isoformat(),
        'models': {}
    }
    
    try:
        # Collect summarization model metrics
        try:
            # Simulate model performance metrics
            summ_metrics = {
                'model_type': 'text_summarization',
                'requests_count': np.random.randint(100, 1000),
                'avg_response_time_ms': np.random.normal(2000, 500),
                'success_rate': np.random.uniform(0.95, 0.99),
                'bleu_score': np.random.uniform(0.6, 0.8),
                'rouge_score': np.random.uniform(0.65, 0.85),
                'avg_input_length': np.random.randint(200, 800),
                'avg_output_length': np.random.randint(50, 150),
                'memory_usage_mb': np.random.randint(500, 1500),
                'cpu_usage_percent': np.random.uniform(30, 80)
            }
            
            metrics['models']['summarization'] = summ_metrics
            logger.info("✅ Summarization model metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect summarization metrics: {e}")
            metrics['models']['summarization'] = {'error': str(e)}
        
        # Collect anomaly detection model metrics
        try:
            anom_metrics = {
                'model_type': 'anomaly_detection',
                'requests_count': np.random.randint(50, 500),
                'avg_response_time_ms': np.random.normal(150, 50),
                'success_rate': np.random.uniform(0.96, 0.99),
                'precision': np.random.uniform(0.8, 0.95),
                'recall': np.random.uniform(0.75, 0.9),
                'f1_score': np.random.uniform(0.77, 0.92),
                'false_positive_rate': np.random.uniform(0.01, 0.05),
                'anomalies_detected': np.random.randint(5, 50),
                'memory_usage_mb': np.random.randint(200, 800),
                'cpu_usage_percent': np.random.uniform(20, 60)
            }
            
            metrics['models']['anomaly_detection'] = anom_metrics
            logger.info("✅ Anomaly detection model metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect anomaly detection metrics: {e}")
            metrics['models']['anomaly_detection'] = {'error': str(e)}
        
        # Save metrics to S3
        metrics_key = f"{MONITORING_PREFIX}/model_metrics/{execution_date.strftime('%Y/%m/%d')}/metrics_{execution_date.strftime('%H%M')}.json"
        
        s3_hook.load_string(
            string_data=json.dumps(metrics, indent=2, default=str),
            key=metrics_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"Model metrics saved to {metrics_key}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error collecting model metrics: {str(e)}")
        raise


def monitor_infrastructure_health(**context) -> Dict[str, Any]:
    """
    Monitor infrastructure components health and performance.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date']
    
    health_status = {
        'timestamp': execution_date.isoformat(),
        'check_date': datetime.now().isoformat(),
        'infrastructure': {}
    }
    
    try:
        # Monitor S3 bucket health
        try:
            bucket_info = s3_hook.get_bucket_lifecycle_configuration(bucket_name=S3_BUCKET)
            bucket_size = s3_hook.get_bucket_size(bucket_name=S3_BUCKET)
            
            s3_health = {
                'status': 'healthy',
                'bucket_size_bytes': bucket_size if bucket_size else 0,
                'bucket_size_gb': round((bucket_size or 0) / (1024**3), 2),
                'lifecycle_configured': bool(bucket_info),
                'region': AWS_REGION
            }
            
            health_status['infrastructure']['s3'] = s3_health
            logger.info("✅ S3 health check completed")
            
        except Exception as e:
            health_status['infrastructure']['s3'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ S3 health check failed: {e}")
        
        # Monitor ECR repository
        try:
            ecr_client = boto3.client('ecr', region_name=AWS_REGION)
            repo_name = 'secure-aiml-ops'
            
            # Get repository info
            repos = ecr_client.describe_repositories(repositoryNames=[repo_name])
            
            # Get image scan results
            images = ecr_client.describe_images(repositoryName=repo_name, maxResults=10)
            
            ecr_health = {
                'status': 'healthy',
                'repository_uri': repos['repositories'][0]['repositoryUri'],
                'image_count': len(images['imageDetails']),
                'latest_push': max([img.get('imagePushedAt', datetime.min) for img in images['imageDetails']]).isoformat() if images['imageDetails'] else None,
                'scan_on_push': repos['repositories'][0].get('imageScanningConfiguration', {}).get('scanOnPush', False)
            }
            
            health_status['infrastructure']['ecr'] = ecr_health
            logger.info("✅ ECR health check completed")
            
        except Exception as e:
            health_status['infrastructure']['ecr'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ ECR health check failed: {e}")
        
        # Monitor VPC and networking
        try:
            ec2_client = boto3.client('ec2', region_name=AWS_REGION)
            
            # Get VPC info
            vpcs = ec2_client.describe_vpcs(
                Filters=[{'Name': 'tag:Name', 'Values': ['secure-aiml-ops-vpc']}]
            )
            
            if vpcs['Vpcs']:
                vpc_id = vpcs['Vpcs'][0]['VpcId']
                
                # Get subnet info
                subnets = ec2_client.describe_subnets(
                    Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}]
                )
                
                vpc_health = {
                    'status': 'healthy',
                    'vpc_id': vpc_id,
                    'subnet_count': len(subnets['Subnets']),
                    'available_subnets': len([s for s in subnets['Subnets'] if s['State'] == 'available'])
                }
            else:
                vpc_health = {
                    'status': 'not_found',
                    'message': 'VPC not found'
                }
            
            health_status['infrastructure']['vpc'] = vpc_health
            logger.info("✅ VPC health check completed")
            
        except Exception as e:
            health_status['infrastructure']['vpc'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ VPC health check failed: {e}")
        
        # Overall health assessment
        healthy_components = sum(1 for comp in health_status['infrastructure'].values() 
                               if comp.get('status') == 'healthy')
        total_components = len(health_status['infrastructure'])
        
        health_status['overall'] = {
            'status': 'healthy' if healthy_components == total_components else 'degraded',
            'healthy_components': healthy_components,
            'total_components': total_components,
            'health_percentage': round((healthy_components / total_components) * 100, 2) if total_components > 0 else 0
        }
        
        # Save health status
        health_key = f"{MONITORING_PREFIX}/infrastructure_health/{execution_date.strftime('%Y/%m/%d')}/health_{execution_date.strftime('%H%M')}.json"
        
        s3_hook.load_string(
            string_data=json.dumps(health_status, indent=2, default=str),
            key=health_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"Infrastructure health status saved to {health_key}")
        return health_status
        
    except Exception as e:
        logger.error(f"Error monitoring infrastructure health: {str(e)}")
        raise


def check_data_quality(**context) -> Dict[str, Any]:
    """
    Monitor data quality and detect anomalies in incoming data.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date']
    
    quality_report = {
        'timestamp': execution_date.isoformat(),
        'check_date': datetime.now().isoformat(),
        'data_sources': {}
    }
    
    try:
        # Check recent data ingestion
        current_date = execution_date.strftime('%Y-%m-%d')
        
        # Check customer tickets data quality
        try:
            tickets_key = f"raw_data/customer_tickets/date={current_date}/customer_tickets.parquet"
            
            if s3_hook.check_for_key(key=tickets_key, bucket_name=S3_BUCKET):
                # In a real scenario, you'd read and analyze the data
                # For now, we'll simulate quality checks
                
                tickets_quality = {
                    'source': 'customer_tickets',
                    'status': 'good',
                    'record_count': np.random.randint(800, 1200),
                    'null_percentage': np.random.uniform(0, 5),
                    'duplicate_percentage': np.random.uniform(0, 2),
                    'completeness_score': np.random.uniform(0.95, 1.0),
                    'freshness_hours': np.random.uniform(0.5, 2.0),
                    'schema_valid': True,
                    'data_drift_score': np.random.uniform(0, 0.1)
                }
            else:
                tickets_quality = {
                    'source': 'customer_tickets',
                    'status': 'missing',
                    'error': 'Data file not found for current date'
                }
            
            quality_report['data_sources']['customer_tickets'] = tickets_quality
            logger.info("✅ Customer tickets data quality checked")
            
        except Exception as e:
            quality_report['data_sources']['customer_tickets'] = {
                'source': 'customer_tickets',
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ Customer tickets quality check failed: {e}")
        
        # Check financial transactions data quality
        try:
            transactions_key = f"raw_data/financial_transactions/date={current_date}/transactions.parquet"
            
            if s3_hook.check_for_key(key=transactions_key, bucket_name=S3_BUCKET):
                transactions_quality = {
                    'source': 'financial_transactions',
                    'status': 'good',
                    'record_count': np.random.randint(2000, 5000),
                    'null_percentage': np.random.uniform(0, 3),
                    'duplicate_percentage': np.random.uniform(0, 1),
                    'completeness_score': np.random.uniform(0.96, 1.0),
                    'freshness_hours': np.random.uniform(0.25, 1.5),
                    'schema_valid': True,
                    'anomaly_rate': np.random.uniform(0.01, 0.05),
                    'value_range_valid': True
                }
            else:
                transactions_quality = {
                    'source': 'financial_transactions',
                    'status': 'missing',
                    'error': 'Data file not found for current date'
                }
            
            quality_report['data_sources']['financial_transactions'] = transactions_quality
            logger.info("✅ Financial transactions data quality checked")
            
        except Exception as e:
            quality_report['data_sources']['financial_transactions'] = {
                'source': 'financial_transactions',
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ Financial transactions quality check failed: {e}")
        
        # Overall data quality assessment
        good_sources = sum(1 for source in quality_report['data_sources'].values() 
                          if source.get('status') == 'good')
        total_sources = len(quality_report['data_sources'])
        
        quality_report['overall'] = {
            'status': 'good' if good_sources == total_sources else 'degraded',
            'good_sources': good_sources,
            'total_sources': total_sources,
            'quality_percentage': round((good_sources / total_sources) * 100, 2) if total_sources > 0 else 0
        }
        
        # Save quality report
        quality_key = f"{MONITORING_PREFIX}/data_quality/{execution_date.strftime('%Y/%m/%d')}/quality_{execution_date.strftime('%H%M')}.json"
        
        s3_hook.load_string(
            string_data=json.dumps(quality_report, indent=2, default=str),
            key=quality_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"Data quality report saved to {quality_key}")
        return quality_report
        
    except Exception as e:
        logger.error(f"Error checking data quality: {str(e)}")
        raise


def analyze_alerts(**context) -> List[Dict[str, Any]]:
    """
    Analyze collected metrics and generate alerts based on thresholds.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date']
    
    alerts = []
    
    try:
        # Retrieve recent metrics for analysis
        metrics_key = f"{MONITORING_PREFIX}/model_metrics/{execution_date.strftime('%Y/%m/%d')}/metrics_{execution_date.strftime('%H%M')}.json"
        health_key = f"{MONITORING_PREFIX}/infrastructure_health/{execution_date.strftime('%Y/%m/%d')}/health_{execution_date.strftime('%H%M')}.json"
        quality_key = f"{MONITORING_PREFIX}/data_quality/{execution_date.strftime('%Y/%m/%d')}/quality_{execution_date.strftime('%H%M')}.json"
        
        # Analyze model performance alerts
        try:
            metrics_obj = s3_hook.get_key(key=metrics_key, bucket_name=S3_BUCKET)
            metrics = json.loads(metrics_obj.get()['Body'].read().decode('utf-8'))
            
            for model_name, model_metrics in metrics.get('models', {}).items():
                if 'error' in model_metrics:
                    alerts.append({
                        'type': 'model_error',
                        'severity': 'high',
                        'model': model_name,
                        'message': f"Model {model_name} is experiencing errors: {model_metrics['error']}",
                        'timestamp': execution_date.isoformat()
                    })
                    continue
                
                # Check response time
                response_time = model_metrics.get('avg_response_time_ms', 0)
                if response_time > ALERT_THRESHOLDS['api_response_time']:
                    alerts.append({
                        'type': 'performance',
                        'severity': 'medium',
                        'model': model_name,
                        'message': f"High response time for {model_name}: {response_time:.2f}ms",
                        'threshold': ALERT_THRESHOLDS['api_response_time'],
                        'actual_value': response_time,
                        'timestamp': execution_date.isoformat()
                    })
                
                # Check success rate
                success_rate = model_metrics.get('success_rate', 1.0)
                error_rate = 1 - success_rate
                if error_rate > ALERT_THRESHOLDS['error_rate']:
                    alerts.append({
                        'type': 'error_rate',
                        'severity': 'high',
                        'model': model_name,
                        'message': f"High error rate for {model_name}: {error_rate*100:.2f}%",
                        'threshold': ALERT_THRESHOLDS['error_rate'] * 100,
                        'actual_value': error_rate * 100,
                        'timestamp': execution_date.isoformat()
                    })
            
        except Exception as e:
            logger.warning(f"Could not analyze model metrics for alerts: {e}")
        
        # Analyze infrastructure health alerts
        try:
            health_obj = s3_hook.get_key(key=health_key, bucket_name=S3_BUCKET)
            health = json.loads(health_obj.get()['Body'].read().decode('utf-8'))
            
            overall_health = health.get('overall', {})
            if overall_health.get('status') != 'healthy':
                alerts.append({
                    'type': 'infrastructure',
                    'severity': 'high',
                    'message': f"Infrastructure health degraded: {overall_health.get('health_percentage', 0):.1f}% healthy",
                    'details': health.get('infrastructure', {}),
                    'timestamp': execution_date.isoformat()
                })
            
        except Exception as e:
            logger.warning(f"Could not analyze infrastructure health for alerts: {e}")
        
        # Analyze data quality alerts
        try:
            quality_obj = s3_hook.get_key(key=quality_key, bucket_name=S3_BUCKET)
            quality = json.loads(quality_obj.get()['Body'].read().decode('utf-8'))
            
            overall_quality = quality.get('overall', {})
            if overall_quality.get('status') != 'good':
                alerts.append({
                    'type': 'data_quality',
                    'severity': 'medium',
                    'message': f"Data quality issues detected: {overall_quality.get('quality_percentage', 0):.1f}% good",
                    'details': quality.get('data_sources', {}),
                    'timestamp': execution_date.isoformat()
                })
            
        except Exception as e:
            logger.warning(f"Could not analyze data quality for alerts: {e}")
        
        # Save alerts
        if alerts:
            alerts_key = f"{MONITORING_PREFIX}/alerts/{execution_date.strftime('%Y/%m/%d')}/alerts_{execution_date.strftime('%H%M')}.json"
            
            alerts_data = {
                'timestamp': execution_date.isoformat(),
                'alert_count': len(alerts),
                'high_severity_count': len([a for a in alerts if a.get('severity') == 'high']),
                'medium_severity_count': len([a for a in alerts if a.get('severity') == 'medium']),
                'alerts': alerts
            }
            
            s3_hook.load_string(
                string_data=json.dumps(alerts_data, indent=2, default=str),
                key=alerts_key,
                bucket_name=S3_BUCKET,
                replace=True
            )
            
            logger.warning(f"🚨 {len(alerts)} alerts generated and saved to {alerts_key}")
        else:
            logger.info("✅ No alerts generated - all systems healthy")
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error analyzing alerts: {str(e)}")
        raise


def generate_monitoring_report(**context) -> None:
    """
    Generate comprehensive monitoring report.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date']
    
    try:
        # Collect data from the last 24 hours for trending
        report_data = {
            'report_date': execution_date.isoformat(),
            'generation_time': datetime.now().isoformat(),
            'period': '24_hours',
            'summary': {},
            'trends': {},
            'recommendations': []
        }
        
        # Generate summary statistics
        report_data['summary'] = {
            'total_model_requests': np.random.randint(5000, 15000),
            'avg_response_time_ms': np.random.normal(1500, 300),
            'overall_success_rate': np.random.uniform(0.96, 0.99),
            'data_processed_gb': np.random.uniform(10, 50),
            'alerts_generated': np.random.randint(0, 5),
            'infrastructure_uptime': np.random.uniform(0.98, 1.0)
        }
        
        # Generate trend analysis
        report_data['trends'] = {
            'model_performance': 'stable',
            'data_volume': 'increasing',
            'response_times': 'decreasing',
            'error_rates': 'stable',
            'storage_usage': 'increasing'
        }
        
        # Generate recommendations
        recommendations = [
            "Monitor model performance closely during peak hours",
            "Consider scaling infrastructure if data volume continues to increase",
            "Review and optimize slow-performing queries",
            "Implement automated model retraining pipeline",
            "Set up proactive alerting for critical metrics"
        ]
        
        report_data['recommendations'] = recommendations[:np.random.randint(2, 5)]
        
        # Save monitoring report
        report_key = f"{MONITORING_PREFIX}/reports/daily_report_{execution_date.strftime('%Y-%m-%d')}.json"
        
        s3_hook.load_string(
            string_data=json.dumps(report_data, indent=2, default=str),
            key=report_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"📊 Daily monitoring report generated and saved to {report_key}")
        
    except Exception as e:
        logger.error(f"Error generating monitoring report: {str(e)}")
        raise


# Create the DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Monitoring and alerting pipeline for AI/ML infrastructure',
    schedule_interval=SCHEDULE_INTERVAL,
    tags=['monitoring', 'alerting', 'health-check', 'metrics'],
    catchup=False,
    max_active_runs=1,
)

# Monitoring and alerting tasks
collect_metrics_task = PythonOperator(
    task_id='collect_model_metrics',
    python_callable=collect_model_metrics,
    dag=dag,
)

monitor_infrastructure_task = PythonOperator(
    task_id='monitor_infrastructure_health',
    python_callable=monitor_infrastructure_health,
    dag=dag,
)

check_data_quality_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

analyze_alerts_task = PythonOperator(
    task_id='analyze_alerts',
    python_callable=analyze_alerts,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id='generate_monitoring_report',
    python_callable=generate_monitoring_report,
    dag=dag,
)

# Set task dependencies - run monitoring tasks in parallel, then analyze
[collect_metrics_task, monitor_infrastructure_task, check_data_quality_task] >> analyze_alerts_task >> generate_report_task

if __name__ == "__main__":
    dag.cli()