"""
Data Preprocessing DAG for Secure AI/ML Operations

This DAG handles data cleaning, transformation, and feature engineering on the raw data
collected by the data ingestion pipeline. It prepares the data for model training.

Key preprocessing steps:
1. Data cleaning and validation
2. Text preprocessing for NLP tasks
3. Feature engineering
4. Data normalization and scaling
5. Data splitting for training/validation
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

import pandas as pd
import numpy as np
import json
import logging
import re
from io import StringIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DAG Configuration
DAG_ID = "data_preprocessing_pipeline"
SCHEDULE_INTERVAL = "0 4 * * *"  # Daily at 4:00 AM (after data ingestion)
S3_BUCKET = "secure-aiml-ops-data"
RAW_DATA_PREFIX = "raw-data"
PROCESSED_DATA_PREFIX = "processed-data"

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


def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"Could not download NLTK data: {e}")


def clean_text(text: str) -> str:
    """Clean and preprocess text data"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    try:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Stem words
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    except:
        # Fallback if NLTK operations fail
        return text


def preprocess_tickets_data(**context) -> str:
    """
    Preprocess customer support tickets data.
    """
    download_nltk_data()
    
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    # Download raw tickets data
    tickets_key = f"{RAW_DATA_PREFIX}/tickets/tickets_{execution_date}.csv"
    
    try:
        tickets_obj = s3_hook.get_key(key=tickets_key, bucket_name=S3_BUCKET)
        tickets_data = tickets_obj.get()['Body'].read().decode('utf-8')
        tickets_df = pd.read_csv(StringIO(tickets_data))
        
        logger.info(f"Loaded {len(tickets_df)} tickets for preprocessing")
        
        # Data cleaning
        tickets_df['title_cleaned'] = tickets_df['title'].apply(clean_text)
        tickets_df['description_cleaned'] = tickets_df['description'].apply(clean_text)
        
        # Feature engineering
        tickets_df['title_length'] = tickets_df['title'].str.len()
        tickets_df['description_length'] = tickets_df['description'].str.len()
        tickets_df['text_combined'] = tickets_df['title_cleaned'] + ' ' + tickets_df['description_cleaned']
        tickets_df['word_count'] = tickets_df['text_combined'].str.split().str.len()
        
        # Create urgency score based on priority and category
        priority_scores = {'low': 1, 'medium': 2, 'high': 3, 'urgent': 4}
        category_multipliers = {
            'billing': 1.2, 'technical': 1.5, 'account': 1.1, 
            'feature_request': 0.8, 'complaint': 1.3
        }
        
        tickets_df['priority_score'] = tickets_df['priority'].map(priority_scores)
        tickets_df['category_multiplier'] = tickets_df['category'].map(category_multipliers)
        tickets_df['urgency_score'] = tickets_df['priority_score'] * tickets_df['category_multiplier']
        
        # Convert datetime columns
        tickets_df['created_at'] = pd.to_datetime(tickets_df['created_at'])
        tickets_df['updated_at'] = pd.to_datetime(tickets_df['updated_at'])
        tickets_df['resolution_time_hours'] = (
            tickets_df['updated_at'] - tickets_df['created_at']
        ).dt.total_seconds() / 3600
        
        # Create categorical encodings
        le_category = LabelEncoder()
        le_priority = LabelEncoder()
        le_status = LabelEncoder()
        
        tickets_df['category_encoded'] = le_category.fit_transform(tickets_df['category'])
        tickets_df['priority_encoded'] = le_priority.fit_transform(tickets_df['priority'])
        tickets_df['status_encoded'] = le_status.fit_transform(tickets_df['status'])
        
        # Save processed data
        processed_csv = StringIO()
        tickets_df.to_csv(processed_csv, index=False)
        
        processed_key = f"{PROCESSED_DATA_PREFIX}/tickets/tickets_processed_{execution_date}.csv"
        s3_hook.load_string(
            string_data=processed_csv.getvalue(),
            key=processed_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        # Save encoders for later use
        encoders = {
            'category_encoder': le_category.classes_.tolist(),
            'priority_encoder': le_priority.classes_.tolist(),
            'status_encoder': le_status.classes_.tolist()
        }
        
        encoders_key = f"{PROCESSED_DATA_PREFIX}/encoders/ticket_encoders_{execution_date}.json"
        s3_hook.load_string(
            string_data=json.dumps(encoders, indent=2),
            key=encoders_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"Processed tickets data uploaded to s3://{S3_BUCKET}/{processed_key}")
        return processed_key
        
    except Exception as e:
        logger.error(f"Error preprocessing tickets data: {str(e)}")
        raise


def preprocess_financial_data(**context) -> str:
    """
    Preprocess financial transaction data.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    # Download raw financial data
    financial_key = f"{RAW_DATA_PREFIX}/financial/transactions_{execution_date}.csv"
    
    try:
        financial_obj = s3_hook.get_key(key=financial_key, bucket_name=S3_BUCKET)
        financial_data = financial_obj.get()['Body'].read().decode('utf-8')
        financial_df = pd.read_csv(StringIO(financial_data))
        
        logger.info(f"Loaded {len(financial_df)} transactions for preprocessing")
        
        # Data cleaning
        financial_df['timestamp'] = pd.to_datetime(financial_df['timestamp'])
        financial_df['hour'] = financial_df['timestamp'].dt.hour
        financial_df['day_of_week'] = financial_df['timestamp'].dt.dayofweek
        financial_df['is_weekend'] = financial_df['day_of_week'].isin([5, 6])
        
        # Feature engineering
        financial_df['amount_log'] = np.log1p(financial_df['amount'])
        
        # Create transaction type encodings
        le_transaction_type = LabelEncoder()
        le_currency = LabelEncoder()
        le_status = LabelEncoder()
        
        financial_df['transaction_type_encoded'] = le_transaction_type.fit_transform(
            financial_df['transaction_type']
        )
        financial_df['currency_encoded'] = le_currency.fit_transform(financial_df['currency'])
        financial_df['status_encoded'] = le_status.fit_transform(financial_df['status'])
        
        # Anomaly detection features
        # Flag unusual amounts (outside 2 standard deviations)
        amount_mean = financial_df['amount'].mean()
        amount_std = financial_df['amount'].std()
        financial_df['is_unusual_amount'] = (
            (financial_df['amount'] > amount_mean + 2 * amount_std) |
            (financial_df['amount'] < amount_mean - 2 * amount_std)
        )
        
        # Flag unusual hours (very early morning)
        financial_df['is_unusual_hour'] = financial_df['hour'].isin([0, 1, 2, 3, 4, 5])
        
        # Normalize amounts by currency (simplified)
        currency_rates = {'USD': 1.0, 'EUR': 1.1, 'GBP': 1.25, 'CAD': 0.75}
        financial_df['amount_usd'] = financial_df.apply(
            lambda row: row['amount'] * currency_rates.get(row['currency'], 1.0), axis=1
        )
        
        # Save processed data
        processed_csv = StringIO()
        financial_df.to_csv(processed_csv, index=False)
        
        processed_key = f"{PROCESSED_DATA_PREFIX}/financial/transactions_processed_{execution_date}.csv"
        s3_hook.load_string(
            string_data=processed_csv.getvalue(),
            key=processed_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        # Save encoders
        encoders = {
            'transaction_type_encoder': le_transaction_type.classes_.tolist(),
            'currency_encoder': le_currency.classes_.tolist(),
            'status_encoder': le_status.classes_.tolist(),
            'currency_rates': currency_rates
        }
        
        encoders_key = f"{PROCESSED_DATA_PREFIX}/encoders/financial_encoders_{execution_date}.json"
        s3_hook.load_string(
            string_data=json.dumps(encoders, indent=2),
            key=encoders_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"Processed financial data uploaded to s3://{S3_BUCKET}/{processed_key}")
        return processed_key
        
    except Exception as e:
        logger.error(f"Error preprocessing financial data: {str(e)}")
        raise


def create_training_datasets(**context) -> Dict[str, str]:
    """
    Create training and validation datasets for model training.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    datasets_created = {}
    
    try:
        # Load processed tickets data for text summarization task
        tickets_key = f"{PROCESSED_DATA_PREFIX}/tickets/tickets_processed_{execution_date}.csv"
        tickets_obj = s3_hook.get_key(key=tickets_key, bucket_name=S3_BUCKET)
        tickets_data = tickets_obj.get()['Body'].read().decode('utf-8')
        tickets_df = pd.read_csv(StringIO(tickets_data))
        
        # Create summarization dataset (title as summary, description as text)
        summarization_df = tickets_df[['description_cleaned', 'title_cleaned']].copy()
        summarization_df.columns = ['text', 'summary']
        summarization_df = summarization_df.dropna()
        
        # Split into train/validation
        train_summ, val_summ = train_test_split(
            summarization_df, test_size=0.2, random_state=42
        )
        
        # Save summarization datasets
        train_summ_csv = StringIO()
        val_summ_csv = StringIO()
        train_summ.to_csv(train_summ_csv, index=False)
        val_summ.to_csv(val_summ_csv, index=False)
        
        train_summ_key = f"{PROCESSED_DATA_PREFIX}/ml-datasets/summarization_train_{execution_date}.csv"
        val_summ_key = f"{PROCESSED_DATA_PREFIX}/ml-datasets/summarization_val_{execution_date}.csv"
        
        s3_hook.load_string(
            string_data=train_summ_csv.getvalue(),
            key=train_summ_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        s3_hook.load_string(
            string_data=val_summ_csv.getvalue(),
            key=val_summ_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        datasets_created['summarization_train'] = train_summ_key
        datasets_created['summarization_val'] = val_summ_key
        
        # Load processed financial data for anomaly detection
        financial_key = f"{PROCESSED_DATA_PREFIX}/financial/transactions_processed_{execution_date}.csv"
        financial_obj = s3_hook.get_key(key=financial_key, bucket_name=S3_BUCKET)
        financial_data = financial_obj.get()['Body'].read().decode('utf-8')
        financial_df = pd.read_csv(StringIO(financial_data))
        
        # Create anomaly detection dataset
        anomaly_features = [
            'amount_log', 'hour', 'day_of_week', 'is_weekend',
            'transaction_type_encoded', 'currency_encoded', 'is_unusual_hour'
        ]
        
        anomaly_df = financial_df[anomaly_features + ['is_unusual_amount']].copy()
        anomaly_df = anomaly_df.dropna()
        
        # Split into train/validation
        train_anom, val_anom = train_test_split(
            anomaly_df, test_size=0.2, random_state=42
        )
        
        # Save anomaly detection datasets
        train_anom_csv = StringIO()
        val_anom_csv = StringIO()
        train_anom.to_csv(train_anom_csv, index=False)
        val_anom.to_csv(val_anom_csv, index=False)
        
        train_anom_key = f"{PROCESSED_DATA_PREFIX}/ml-datasets/anomaly_train_{execution_date}.csv"
        val_anom_key = f"{PROCESSED_DATA_PREFIX}/ml-datasets/anomaly_val_{execution_date}.csv"
        
        s3_hook.load_string(
            string_data=train_anom_csv.getvalue(),
            key=train_anom_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        s3_hook.load_string(
            string_data=val_anom_csv.getvalue(),
            key=val_anom_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        datasets_created['anomaly_train'] = train_anom_key
        datasets_created['anomaly_val'] = val_anom_key
        
        # Save dataset metadata
        metadata = {
            'execution_date': execution_date,
            'datasets_created': datasets_created,
            'dataset_sizes': {
                'summarization_train': len(train_summ),
                'summarization_val': len(val_summ),
                'anomaly_train': len(train_anom),
                'anomaly_val': len(val_anom)
            },
            'features': {
                'summarization': ['text', 'summary'],
                'anomaly_detection': anomaly_features + ['is_unusual_amount']
            }
        }
        
        metadata_key = f"{PROCESSED_DATA_PREFIX}/metadata/datasets_metadata_{execution_date}.json"
        s3_hook.load_string(
            string_data=json.dumps(metadata, indent=2),
            key=metadata_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"Created {len(datasets_created)} ML datasets")
        return datasets_created
        
    except Exception as e:
        logger.error(f"Error creating training datasets: {str(e)}")
        raise


def validate_preprocessing_quality(**context) -> bool:
    """
    Validate the quality of preprocessing results.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    validation_results = []
    
    # Check processed files
    expected_files = [
        f"{PROCESSED_DATA_PREFIX}/tickets/tickets_processed_{execution_date}.csv",
        f"{PROCESSED_DATA_PREFIX}/financial/transactions_processed_{execution_date}.csv",
        f"{PROCESSED_DATA_PREFIX}/ml-datasets/summarization_train_{execution_date}.csv",
        f"{PROCESSED_DATA_PREFIX}/ml-datasets/summarization_val_{execution_date}.csv",
        f"{PROCESSED_DATA_PREFIX}/ml-datasets/anomaly_train_{execution_date}.csv",
        f"{PROCESSED_DATA_PREFIX}/ml-datasets/anomaly_val_{execution_date}.csv"
    ]
    
    for file_key in expected_files:
        try:
            if s3_hook.check_for_key(key=file_key, bucket_name=S3_BUCKET):
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
    
    # Create validation report
    validation_report = {
        'execution_date': execution_date,
        'validation_timestamp': datetime.now().isoformat(),
        'files_validated': len(validation_results),
        'files_valid': sum(1 for r in validation_results if r['valid']),
        'results': validation_results
    }
    
    report_key = f"{PROCESSED_DATA_PREFIX}/validation/preprocessing_quality_report_{execution_date}.json"
    s3_hook.load_string(
        string_data=json.dumps(validation_report, indent=2),
        key=report_key,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    all_valid = all(r['valid'] for r in validation_results)
    logger.info(f"Preprocessing validation complete. All valid: {all_valid}")
    return all_valid


# Create the DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Data preprocessing pipeline for AI/ML workflows',
    schedule_interval=SCHEDULE_INTERVAL,
    tags=['data-preprocessing', 'etl', 'ai-ml', 'feature-engineering'],
    catchup=False,
    max_active_runs=1,
)

# Wait for data ingestion to complete
wait_for_ingestion = ExternalTaskSensor(
    task_id='wait_for_data_ingestion',
    external_dag_id='data_ingestion_pipeline',
    external_task_id='send_completion_notification',
    timeout=3600,  # 1 hour timeout
    poke_interval=300,  # Check every 5 minutes
    dag=dag,
)

# Preprocessing tasks
preprocess_tickets_task = PythonOperator(
    task_id='preprocess_tickets_data',
    python_callable=preprocess_tickets_data,
    dag=dag,
)

preprocess_financial_task = PythonOperator(
    task_id='preprocess_financial_data',
    python_callable=preprocess_financial_data,
    dag=dag,
)

create_datasets_task = PythonOperator(
    task_id='create_training_datasets',
    python_callable=create_training_datasets,
    dag=dag,
)

validate_preprocessing_task = PythonOperator(
    task_id='validate_preprocessing_quality',
    python_callable=validate_preprocessing_quality,
    dag=dag,
)

# Set task dependencies
wait_for_ingestion >> [preprocess_tickets_task, preprocess_financial_task]
[preprocess_tickets_task, preprocess_financial_task] >> create_datasets_task >> validate_preprocessing_task

if __name__ == "__main__":
    dag.cli()