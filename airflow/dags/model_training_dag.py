"""
Model Training DAG for Secure AI/ML Operations

This DAG handles automated model training using the preprocessed data.
It trains two main models:
1. Text Summarization Model (for customer ticket summarization)
2. Anomaly Detection Model (for financial transaction monitoring)

The DAG includes model evaluation, versioning, and artifact storage.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

import pandas as pd
import numpy as np
import json
import logging
import pickle
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    TrainingArguments, Trainer, 
    DataCollatorForSeq2Seq
)
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from io import StringIO, BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DAG Configuration
DAG_ID = "model_training_pipeline"
SCHEDULE_INTERVAL = "0 6 * * 1"  # Weekly on Monday at 6:00 AM
S3_BUCKET = "secure-aiml-ops-data"
PROCESSED_DATA_PREFIX = "processed-data"
MODELS_PREFIX = "models"

# Model configurations
SUMMARIZATION_MODEL_NAME = "t5-small"  # Using smaller model for resource efficiency
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

# Default arguments
default_args = {
    'owner': 'secure-aiml-ops',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'catchup': False,
}


class SummarizationDataset(torch.utils.data.Dataset):
    """Custom dataset for text summarization"""
    
    def __init__(self, texts, summaries, tokenizer, max_input_length, max_target_length):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        summary = str(self.summaries[idx])
        
        # Tokenize inputs
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }


def train_summarization_model(**context) -> str:
    """
    Train a text summarization model for customer tickets.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    try:
        # Load training data
        train_key = f"{PROCESSED_DATA_PREFIX}/ml-datasets/summarization_train_{execution_date}.csv"
        val_key = f"{PROCESSED_DATA_PREFIX}/ml-datasets/summarization_val_{execution_date}.csv"
        
        train_obj = s3_hook.get_key(key=train_key, bucket_name=S3_BUCKET)
        train_data = train_obj.get()['Body'].read().decode('utf-8')
        train_df = pd.read_csv(StringIO(train_data))
        
        val_obj = s3_hook.get_key(key=val_key, bucket_name=S3_BUCKET)
        val_data = val_obj.get()['Body'].read().decode('utf-8')
        val_df = pd.read_csv(StringIO(val_data))
        
        logger.info(f"Loaded {len(train_df)} training and {len(val_df)} validation samples")
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create datasets
        train_dataset = SummarizationDataset(
            train_df['text'].tolist(),
            train_df['summary'].tolist(),
            tokenizer,
            MAX_INPUT_LENGTH,
            MAX_TARGET_LENGTH
        )
        
        val_dataset = SummarizationDataset(
            val_df['text'].tolist(),
            val_df['summary'].tolist(),
            tokenizer,
            MAX_INPUT_LENGTH,
            MAX_TARGET_LENGTH
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='/tmp/summarization_model',
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Small batch size for resource efficiency
            per_device_eval_batch_size=2,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='/tmp/logs',
            logging_steps=50,
            eval_steps=500,
            save_steps=1000,
            evaluation_strategy='steps',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            prediction_loss_only=True,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train model
        logger.info("Starting model training...")
        trainer.train()
        
        # Save model
        model.save_pretrained('/tmp/summarization_model/final')
        tokenizer.save_pretrained('/tmp/summarization_model/final')
        
        # Create model artifacts
        model_artifacts = {
            'model_name': 'text_summarization',
            'model_type': 'transformer',
            'base_model': SUMMARIZATION_MODEL_NAME,
            'training_date': execution_date,
            'training_samples': len(train_df),
            'validation_samples': len(val_df),
            'max_input_length': MAX_INPUT_LENGTH,
            'max_target_length': MAX_TARGET_LENGTH,
            'training_args': training_args.to_dict()
        }
        
        # Upload model artifacts to S3
        model_key = f"{MODELS_PREFIX}/summarization/model_{execution_date}"
        
        # Upload model files (in production, you'd want to use a proper model storage solution)
        artifacts_json = json.dumps(model_artifacts, indent=2)
        artifacts_key = f"{model_key}/artifacts.json"
        
        s3_hook.load_string(
            string_data=artifacts_json,
            key=artifacts_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"Summarization model training completed and artifacts saved to {artifacts_key}")
        return artifacts_key
        
    except Exception as e:
        logger.error(f"Error training summarization model: {str(e)}")
        raise


def train_anomaly_detection_model(**context) -> str:
    """
    Train an anomaly detection model for financial transactions.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    try:
        # Load training data
        train_key = f"{PROCESSED_DATA_PREFIX}/ml-datasets/anomaly_train_{execution_date}.csv"
        val_key = f"{PROCESSED_DATA_PREFIX}/ml-datasets/anomaly_val_{execution_date}.csv"
        
        train_obj = s3_hook.get_key(key=train_key, bucket_name=S3_BUCKET)
        train_data = train_obj.get()['Body'].read().decode('utf-8')
        train_df = pd.read_csv(StringIO(train_data))
        
        val_obj = s3_hook.get_key(key=val_key, bucket_name=S3_BUCKET)
        val_data = val_obj.get()['Body'].read().decode('utf-8')
        val_df = pd.read_csv(StringIO(val_data))
        
        logger.info(f"Loaded {len(train_df)} training and {len(val_df)} validation samples")
        
        # Prepare features and labels
        feature_columns = [
            'amount_log', 'hour', 'day_of_week', 'is_weekend',
            'transaction_type_encoded', 'currency_encoded', 'is_unusual_hour'
        ]
        
        X_train = train_df[feature_columns]
        y_train = train_df['is_unusual_amount']
        X_val = val_df[feature_columns]
        y_val = val_df['is_unusual_amount']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train Isolation Forest for anomaly detection
        model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        # Fit on normal transactions only (where is_unusual_amount is False)
        normal_transactions = X_train_scaled[y_train == False]
        model.fit(normal_transactions)
        
        # Make predictions
        train_predictions = model.predict(X_train_scaled)
        val_predictions = model.predict(X_val_scaled)
        
        # Convert predictions (-1 for anomaly, 1 for normal) to binary (1 for anomaly, 0 for normal)
        train_pred_binary = (train_predictions == -1).astype(int)
        val_pred_binary = (val_predictions == -1).astype(int)
        
        # Calculate metrics
        train_report = classification_report(y_train, train_pred_binary, output_dict=True)
        val_report = classification_report(y_val, val_pred_binary, output_dict=True)
        
        # Model evaluation metrics
        model_metrics = {
            'training_accuracy': train_report['accuracy'],
            'validation_accuracy': val_report['accuracy'],
            'training_precision': train_report['1']['precision'],
            'validation_precision': val_report['1']['precision'],
            'training_recall': train_report['1']['recall'],
            'validation_recall': val_report['1']['recall'],
            'training_f1': train_report['1']['f1-score'],
            'validation_f1': val_report['1']['f1-score']
        }
        
        # Save model and scaler
        model_buffer = BytesIO()
        scaler_buffer = BytesIO()
        
        joblib.dump(model, model_buffer)
        joblib.dump(scaler, scaler_buffer)
        
        model_buffer.seek(0)
        scaler_buffer.seek(0)
        
        # Upload model to S3
        model_key = f"{MODELS_PREFIX}/anomaly_detection/model_{execution_date}/isolation_forest.pkl"
        scaler_key = f"{MODELS_PREFIX}/anomaly_detection/model_{execution_date}/scaler.pkl"
        
        s3_hook.load_bytes(
            bytes_data=model_buffer.getvalue(),
            key=model_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        s3_hook.load_bytes(
            bytes_data=scaler_buffer.getvalue(),
            key=scaler_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        # Create model artifacts
        model_artifacts = {
            'model_name': 'anomaly_detection',
            'model_type': 'isolation_forest',
            'training_date': execution_date,
            'training_samples': len(train_df),
            'validation_samples': len(val_df),
            'feature_columns': feature_columns,
            'contamination_rate': 0.1,
            'metrics': model_metrics,
            'model_path': model_key,
            'scaler_path': scaler_key
        }
        
        # Upload artifacts
        artifacts_json = json.dumps(model_artifacts, indent=2)
        artifacts_key = f"{MODELS_PREFIX}/anomaly_detection/model_{execution_date}/artifacts.json"
        
        s3_hook.load_string(
            string_data=artifacts_json,
            key=artifacts_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"Anomaly detection model training completed. Validation F1: {model_metrics['validation_f1']:.3f}")
        return artifacts_key
        
    except Exception as e:
        logger.error(f"Error training anomaly detection model: {str(e)}")
        raise


def evaluate_models(**context) -> Dict[str, Any]:
    """
    Evaluate trained models and create performance reports.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    evaluation_results = {}
    
    try:
        # Load model artifacts
        summ_artifacts_key = f"{MODELS_PREFIX}/summarization/model_{execution_date}/artifacts.json"
        anom_artifacts_key = f"{MODELS_PREFIX}/anomaly_detection/model_{execution_date}/artifacts.json"
        
        # Summarization model evaluation
        try:
            summ_obj = s3_hook.get_key(key=summ_artifacts_key, bucket_name=S3_BUCKET)
            summ_artifacts = json.loads(summ_obj.get()['Body'].read().decode('utf-8'))
            
            evaluation_results['summarization'] = {
                'model_name': summ_artifacts['model_name'],
                'training_date': summ_artifacts['training_date'],
                'training_samples': summ_artifacts['training_samples'],
                'status': 'completed',
                'base_model': summ_artifacts['base_model']
            }
        except Exception as e:
            evaluation_results['summarization'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Anomaly detection model evaluation
        try:
            anom_obj = s3_hook.get_key(key=anom_artifacts_key, bucket_name=S3_BUCKET)
            anom_artifacts = json.loads(anom_obj.get()['Body'].read().decode('utf-8'))
            
            evaluation_results['anomaly_detection'] = {
                'model_name': anom_artifacts['model_name'],
                'training_date': anom_artifacts['training_date'],
                'training_samples': anom_artifacts['training_samples'],
                'validation_f1': anom_artifacts['metrics']['validation_f1'],
                'validation_precision': anom_artifacts['metrics']['validation_precision'],
                'validation_recall': anom_artifacts['metrics']['validation_recall'],
                'status': 'completed'
            }
        except Exception as e:
            evaluation_results['anomaly_detection'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Create evaluation report
        evaluation_report = {
            'execution_date': execution_date,
            'evaluation_timestamp': datetime.now().isoformat(),
            'models_evaluated': list(evaluation_results.keys()),
            'results': evaluation_results,
            'overall_status': 'success' if all(
                r.get('status') == 'completed' for r in evaluation_results.values()
            ) else 'partial_failure'
        }
        
        # Upload evaluation report
        report_key = f"{MODELS_PREFIX}/evaluation/model_evaluation_report_{execution_date}.json"
        s3_hook.load_string(
            string_data=json.dumps(evaluation_report, indent=2),
            key=report_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"Model evaluation completed. Report saved to {report_key}")
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise


def update_model_registry(**context) -> None:
    """
    Update the model registry with new model versions.
    """
    s3_hook = S3Hook(aws_conn_id='aws_default')
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    try:
        # Load existing model registry or create new one
        registry_key = f"{MODELS_PREFIX}/registry/model_registry.json"
        
        try:
            registry_obj = s3_hook.get_key(key=registry_key, bucket_name=S3_BUCKET)
            model_registry = json.loads(registry_obj.get()['Body'].read().decode('utf-8'))
        except:
            model_registry = {
                'models': {},
                'last_updated': None,
                'version': '1.0.0'
            }
        
        # Update registry with new models
        model_registry['models'][execution_date] = {
            'summarization': {
                'artifacts_path': f"{MODELS_PREFIX}/summarization/model_{execution_date}/artifacts.json",
                'model_type': 'transformer',
                'status': 'active'
            },
            'anomaly_detection': {
                'artifacts_path': f"{MODELS_PREFIX}/anomaly_detection/model_{execution_date}/artifacts.json",
                'model_type': 'isolation_forest',
                'status': 'active'
            }
        }
        
        model_registry['last_updated'] = datetime.now().isoformat()
        
        # Upload updated registry
        s3_hook.load_string(
            string_data=json.dumps(model_registry, indent=2),
            key=registry_key,
            bucket_name=S3_BUCKET,
            replace=True
        )
        
        logger.info(f"Model registry updated with models from {execution_date}")
        
    except Exception as e:
        logger.error(f"Error updating model registry: {str(e)}")
        raise


# Create the DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Model training pipeline for AI/ML workflows',
    schedule_interval=SCHEDULE_INTERVAL,
    tags=['model-training', 'ai-ml', 'deep-learning', 'anomaly-detection'],
    catchup=False,
    max_active_runs=1,
)

# Wait for data preprocessing to complete
wait_for_preprocessing = ExternalTaskSensor(
    task_id='wait_for_data_preprocessing',
    external_dag_id='data_preprocessing_pipeline',
    external_task_id='validate_preprocessing_quality',
    timeout=7200,  # 2 hour timeout
    poke_interval=600,  # Check every 10 minutes
    dag=dag,
)

# Model training tasks
train_summarization_task = PythonOperator(
    task_id='train_summarization_model',
    python_callable=train_summarization_model,
    dag=dag,
)

train_anomaly_task = PythonOperator(
    task_id='train_anomaly_detection_model',
    python_callable=train_anomaly_detection_model,
    dag=dag,
)

evaluate_models_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag,
)

update_registry_task = PythonOperator(
    task_id='update_model_registry',
    python_callable=update_model_registry,
    dag=dag,
)

# Set task dependencies
wait_for_preprocessing >> [train_summarization_task, train_anomaly_task]
[train_summarization_task, train_anomaly_task] >> evaluate_models_task >> update_registry_task

if __name__ == "__main__":
    dag.cli()