"""
Data Ingestion DAG for Secure AI/ML Operations

This DAG handles automated data collection from various sources including:
- Customer support tickets (simulated)
- Financial reports (simulated)
- External APIs
- File uploads

The DAG runs daily and stores raw data in S3 for downstream processing.
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago

import pandas as pd
import boto3
import json
import logging
from io import StringIO
import random
from faker import Faker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DAG Configuration
DAG_ID = "data_ingestion_pipeline"
SCHEDULE_INTERVAL = "0 2 * * *"  # Daily at 2:00 AM
S3_BUCKET = "secure-aiml-ops-data"  # This should match your actual bucket name
S3_KEY_PREFIX = "raw-data"

# Default arguments
default_args = {
    'owner': 'secure-aiml-ops',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}


def generate_customer_tickets(**context) -> str:
    """
    Generate simulated customer support tickets.
    In production, this would connect to actual ticket systems.
    """
    fake = Faker()
    tickets = []
    
    # Generate 50-200 tickets
    num_tickets = random.randint(50, 200)
    
    categories = ['billing', 'technical', 'account', 'feature_request', 'complaint']
    priorities = ['low', 'medium', 'high', 'urgent']
    statuses = ['open', 'in_progress', 'resolved', 'closed']
    
    for i in range(num_tickets):
        ticket = {
            'ticket_id': f"TICK-{fake.random_number(digits=6)}",
            'customer_id': f"CUST-{fake.random_number(digits=5)}",
            'category': random.choice(categories),
            'priority': random.choice(priorities),
            'status': random.choice(statuses),
            'title': fake.sentence(nb_words=6),
            'description': fake.text(max_nb_chars=500),
            'created_at': fake.date_time_between(start_date='-7d', end_date='now').isoformat(),
            'updated_at': fake.date_time_between(start_date='-1d', end_date='now').isoformat(),
            'agent_id': f"AGENT-{fake.random_number(digits=3)}",
            'customer_satisfaction': random.randint(1, 5) if random.random() > 0.3 else None,
        }
        tickets.append(ticket)
    
    # Convert to DataFrame and then to CSV
    df = pd.DataFrame(tickets)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Upload to S3
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    s3_key = f"{S3_KEY_PREFIX}/tickets/tickets_{execution_date}.csv"
    
    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        key=s3_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    logger.info(f"Generated {num_tickets} tickets and uploaded to s3://{S3_BUCKET}/{s3_key}")
    return s3_key


def generate_financial_reports(**context) -> str:
    """
    Generate simulated financial transaction data.
    In production, this would connect to actual financial systems.
    """
    fake = Faker()
    transactions = []
    
    # Generate 100-500 transactions
    num_transactions = random.randint(100, 500)
    
    transaction_types = ['deposit', 'withdrawal', 'transfer', 'payment', 'refund']
    currencies = ['USD', 'EUR', 'GBP', 'CAD']
    
    for i in range(num_transactions):
        transaction = {
            'transaction_id': f"TXN-{fake.random_number(digits=8)}",
            'account_id': f"ACC-{fake.random_number(digits=6)}",
            'customer_id': f"CUST-{fake.random_number(digits=5)}",
            'transaction_type': random.choice(transaction_types),
            'amount': round(random.uniform(10, 10000), 2),
            'currency': random.choice(currencies),
            'description': fake.sentence(nb_words=4),
            'timestamp': fake.date_time_between(start_date='-1d', end_date='now').isoformat(),
            'merchant': fake.company() if random.random() > 0.5 else None,
            'location': fake.city(),
            'status': random.choice(['completed', 'pending', 'failed']),
        }
        transactions.append(transaction)
    
    # Convert to DataFrame and then to CSV
    df = pd.DataFrame(transactions)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Upload to S3
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    s3_key = f"{S3_KEY_PREFIX}/financial/transactions_{execution_date}.csv"
    
    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        key=s3_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    logger.info(f"Generated {num_transactions} transactions and uploaded to s3://{S3_BUCKET}/{s3_key}")
    return s3_key


def fetch_external_data(**context) -> str:
    """
    Fetch data from external APIs.
    This is a placeholder for actual API integrations.
    """
    # Simulate external API data
    fake = Faker()
    api_data = []
    
    # Generate market data
    for i in range(20):
        data_point = {
            'symbol': random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']),
            'price': round(random.uniform(100, 300), 2),
            'volume': random.randint(1000000, 50000000),
            'timestamp': fake.date_time_between(start_date='-1h', end_date='now').isoformat(),
            'change_percent': round(random.uniform(-5, 5), 2),
        }
        api_data.append(data_point)
    
    # Convert to JSON
    json_data = json.dumps(api_data, indent=2)
    
    # Upload to S3
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    s3_key = f"{S3_KEY_PREFIX}/external/market_data_{execution_date}.json"
    
    s3_hook.load_string(
        string_data=json_data,
        key=s3_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    logger.info(f"Fetched external data and uploaded to s3://{S3_BUCKET}/{s3_key}")
    return s3_key


def validate_data_quality(**context) -> bool:
    """
    Validate the quality and integrity of ingested data.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    # Check if all expected files exist
    expected_files = [
        f"{S3_KEY_PREFIX}/tickets/tickets_{execution_date}.csv",
        f"{S3_KEY_PREFIX}/financial/transactions_{execution_date}.csv",
        f"{S3_KEY_PREFIX}/external/market_data_{execution_date}.json"
    ]
    
    validation_results = []
    
    for file_key in expected_files:
        try:
            # Check if file exists
            if s3_hook.check_for_key(key=file_key, bucket_name=S3_BUCKET):
                # Get file size
                obj = s3_hook.get_key(key=file_key, bucket_name=S3_BUCKET)
                file_size = obj.content_length
                
                validation_results.append({
                    'file': file_key,
                    'exists': True,
                    'size_bytes': file_size,
                    'valid': file_size > 0
                })
                logger.info(f"✅ File {file_key} exists and is valid ({file_size} bytes)")
            else:
                validation_results.append({
                    'file': file_key,
                    'exists': False,
                    'size_bytes': 0,
                    'valid': False
                })
                logger.error(f"❌ File {file_key} does not exist")
        except Exception as e:
            logger.error(f"❌ Error validating {file_key}: {str(e)}")
            validation_results.append({
                'file': file_key,
                'exists': False,
                'size_bytes': 0,
                'valid': False,
                'error': str(e)
            })
    
    # Upload validation report
    validation_report = {
        'execution_date': execution_date,
        'validation_timestamp': datetime.now().isoformat(),
        'files_validated': len(validation_results),
        'files_valid': sum(1 for r in validation_results if r['valid']),
        'results': validation_results
    }
    
    report_key = f"{S3_KEY_PREFIX}/validation/data_quality_report_{execution_date}.json"
    s3_hook.load_string(
        string_data=json.dumps(validation_report, indent=2),
        key=report_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    # Return True if all files are valid
    all_valid = all(r['valid'] for r in validation_results)
    logger.info(f"Data validation complete. All valid: {all_valid}")
    return all_valid


def send_completion_notification(**context) -> None:
    """
    Send notification that data ingestion is complete.
    """
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    # In production, this would send actual notifications (Slack, email, etc.)
    logger.info(f"🎉 Data ingestion pipeline completed successfully for {execution_date}")
    
    # Create a summary file
    summary = {
        'dag_id': DAG_ID,
        'execution_date': execution_date,
        'completion_time': datetime.now().isoformat(),
        'status': 'success',
        'files_ingested': [
            f"{S3_KEY_PREFIX}/tickets/tickets_{execution_date}.csv",
            f"{S3_KEY_PREFIX}/financial/transactions_{execution_date}.csv",
            f"{S3_KEY_PREFIX}/external/market_data_{execution_date}.json"
        ]
    }
    
    s3_hook = S3Hook(aws_conn_id='aws_default')
    summary_key = f"{S3_KEY_PREFIX}/summary/ingestion_summary_{execution_date}.json"
    
    s3_hook.load_string(
        string_data=json.dumps(summary, indent=2),
        key=summary_key,
        bucket_name=S3_BUCKET,
        replace=True
    )


# Create the DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Daily data ingestion pipeline for AI/ML workflows',
    schedule_interval=SCHEDULE_INTERVAL,
    tags=['data-ingestion', 'etl', 'ai-ml'],
    catchup=False,
    max_active_runs=1,
)

# Define tasks
generate_tickets_task = PythonOperator(
    task_id='generate_customer_tickets',
    python_callable=generate_customer_tickets,
    dag=dag,
    doc_md="Generate simulated customer support tickets and upload to S3"
)

generate_financial_task = PythonOperator(
    task_id='generate_financial_reports',
    python_callable=generate_financial_reports,
    dag=dag,
    doc_md="Generate simulated financial transaction data and upload to S3"
)

fetch_external_task = PythonOperator(
    task_id='fetch_external_data',
    python_callable=fetch_external_data,
    dag=dag,
    doc_md="Fetch data from external APIs and upload to S3"
)

validate_quality_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag,
    doc_md="Validate data quality and integrity of ingested files"
)

completion_notification_task = PythonOperator(
    task_id='send_completion_notification',
    python_callable=send_completion_notification,
    dag=dag,
    doc_md="Send completion notification and create summary report"
)

# Set task dependencies
[generate_tickets_task, generate_financial_task, fetch_external_task] >> validate_quality_task >> completion_notification_task

if __name__ == "__main__":
    dag.cli()