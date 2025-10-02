"""
Data Processing Utilities
=======================

Data validation, transformation, and processing utilities for the Streamlit application.
Handles data cleaning, validation, and preprocessing tasks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import re
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import streamlit as st

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_text_input(text: str, 
                           min_length: int = 10,
                           max_length: int = 10000) -> ValidationResult:
        """
        Validate text input for processing
        
        Args:
            text: Input text to validate
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        
        if not isinstance(text, str):
            errors.append("Input must be a string")
            return ValidationResult(False, errors, warnings, {})
        
        text = text.strip()
        
        if len(text) == 0:
            errors.append("Text cannot be empty")
        elif len(text) < min_length:
            errors.append(f"Text too short (minimum {min_length} characters)")
        elif len(text) > max_length:
            errors.append(f"Text too long (maximum {max_length} characters)")
        
        # Check for potential issues
        if len(text.split()) < 5:
            warnings.append("Very short text may produce poor summaries")
        
        if not any(c.isalpha() for c in text):
            warnings.append("Text contains no alphabetic characters")
        
        # Language detection (simple heuristic)
        non_ascii_chars = sum(1 for c in text if ord(c) > 127)
        if non_ascii_chars / len(text) > 0.3:
            warnings.append("Text may not be in English")
        
        summary = {
            "length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r'[.!?]+', text)),
            "non_ascii_ratio": non_ascii_chars / len(text) if text else 0
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
    
    @staticmethod
    def validate_numerical_data(data: Union[List[float], np.ndarray, pd.Series],
                               allow_negative: bool = True,
                               min_value: Optional[float] = None,
                               max_value: Optional[float] = None) -> ValidationResult:
        """
        Validate numerical data for anomaly detection
        
        Args:
            data: Numerical data to validate
            allow_negative: Whether negative values are allowed
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        
        # Convert to numpy array for easier processing
        try:
            data_array = np.array(data, dtype=float)
        except (ValueError, TypeError):
            errors.append("Data must be numerical")
            return ValidationResult(False, errors, warnings, {})
        
        # Check for empty data
        if len(data_array) == 0:
            errors.append("Data cannot be empty")
            return ValidationResult(False, errors, warnings, {})
        
        # Check for NaN or infinite values
        nan_count = np.sum(np.isnan(data_array))
        inf_count = np.sum(np.isinf(data_array))
        
        if nan_count > 0:
            warnings.append(f"Data contains {nan_count} NaN values")
        
        if inf_count > 0:
            warnings.append(f"Data contains {inf_count} infinite values")
        
        # Clean data for further analysis
        clean_data = data_array[np.isfinite(data_array)]
        
        if len(clean_data) == 0:
            errors.append("No valid numerical data found")
            return ValidationResult(False, errors, warnings, {})
        
        # Value range checks
        if not allow_negative and np.any(clean_data < 0):
            negative_count = np.sum(clean_data < 0)
            errors.append(f"Negative values not allowed ({negative_count} found)")
        
        if min_value is not None and np.any(clean_data < min_value):
            below_min = np.sum(clean_data < min_value)
            errors.append(f"{below_min} values below minimum {min_value}")
        
        if max_value is not None and np.any(clean_data > max_value):
            above_max = np.sum(clean_data > max_value)
            errors.append(f"{above_max} values above maximum {max_value}")
        
        # Statistical checks
        std_dev = np.std(clean_data)
        if std_dev == 0:
            warnings.append("Data has zero variance")
        
        # Check for outliers using IQR method
        q1 = np.percentile(clean_data, 25)
        q3 = np.percentile(clean_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((clean_data < lower_bound) | (clean_data > upper_bound))
        
        if outliers > len(clean_data) * 0.1:  # More than 10% outliers
            warnings.append(f"High number of outliers detected ({outliers}/{len(clean_data)})")
        
        summary = {
            "count": len(data_array),
            "valid_count": len(clean_data),
            "mean": float(np.mean(clean_data)) if len(clean_data) > 0 else None,
            "std": float(np.std(clean_data)) if len(clean_data) > 0 else None,
            "min": float(np.min(clean_data)) if len(clean_data) > 0 else None,
            "max": float(np.max(clean_data)) if len(clean_data) > 0 else None,
            "nan_count": int(nan_count),
            "inf_count": int(inf_count),
            "outlier_count": int(outliers)
        }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            summary=summary
        )

class DataTransformer:
    """Data transformation utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text for processing
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text
    
    @staticmethod
    def normalize_numerical_data(data: Union[List[float], np.ndarray, pd.Series],
                                method: str = "z_score") -> np.ndarray:
        """
        Normalize numerical data
        
        Args:
            data: Input data to normalize
            method: Normalization method ('z_score', 'min_max', 'robust')
            
        Returns:
            Normalized data array
        """
        data_array = np.array(data, dtype=float)
        
        # Remove NaN and infinite values
        mask = np.isfinite(data_array)
        clean_data = data_array[mask]
        
        if len(clean_data) == 0:
            return data_array
        
        if method == "z_score":
            mean_val = np.mean(clean_data)
            std_val = np.std(clean_data)
            if std_val > 0:
                data_array[mask] = (clean_data - mean_val) / std_val
        
        elif method == "min_max":
            min_val = np.min(clean_data)
            max_val = np.max(clean_data)
            if max_val > min_val:
                data_array[mask] = (clean_data - min_val) / (max_val - min_val)
        
        elif method == "robust":
            median_val = np.median(clean_data)
            mad = np.median(np.abs(clean_data - median_val))
            if mad > 0:
                data_array[mask] = (clean_data - median_val) / mad
        
        return data_array
    
    @staticmethod
    def extract_features_from_text(text: str) -> Dict[str, Any]:
        """
        Extract features from text for analysis
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Basic statistics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text.strip()))
        
        # Character-based features
        alpha_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if c in '.,!?;:"()[]{}')
        upper_count = sum(1 for c in text if c.isupper())
        
        # Word-based features
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        long_words = sum(1 for word in words if len(word) > 6)
        
        # Readability approximation (simplified Flesch-Kincaid)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "alpha_ratio": alpha_count / char_count if char_count > 0 else 0,
            "digit_ratio": digit_count / char_count if char_count > 0 else 0,
            "punct_ratio": punct_count / char_count if char_count > 0 else 0,
            "upper_ratio": upper_count / char_count if char_count > 0 else 0,
            "long_word_ratio": long_words / word_count if word_count > 0 else 0
        }

class DataProcessor:
    """Main data processing class"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.transformer = DataTransformer()
    
    def process_text_for_summarization(self, text: str) -> Tuple[str, ValidationResult]:
        """
        Process text for summarization
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (processed_text, validation_result)
        """
        # Validate input
        validation = self.validator.validate_text_input(text)
        
        if not validation.is_valid:
            return text, validation
        
        # Clean and transform text
        processed_text = self.transformer.clean_text(text)
        
        return processed_text, validation
    
    def process_data_for_anomaly_detection(self, 
                                         data: Union[List[float], np.ndarray, pd.Series],
                                         normalize: bool = True) -> Tuple[np.ndarray, ValidationResult]:
        """
        Process data for anomaly detection
        
        Args:
            data: Input data
            normalize: Whether to normalize the data
            
        Returns:
            Tuple of (processed_data, validation_result)
        """
        # Validate input
        validation = self.validator.validate_numerical_data(data)
        
        if not validation.is_valid:
            return np.array(data), validation
        
        # Convert to numpy array
        data_array = np.array(data, dtype=float)
        
        # Normalize if requested
        if normalize:
            data_array = self.transformer.normalize_numerical_data(data_array)
        
        return data_array, validation
    
    def create_sample_dataset(self, 
                            dataset_type: str = "financial",
                            size: int = 1000,
                            anomaly_rate: float = 0.05) -> pd.DataFrame:
        """
        Create sample dataset for testing
        
        Args:
            dataset_type: Type of dataset to create
            size: Number of samples
            anomaly_rate: Proportion of anomalous samples
            
        Returns:
            Sample dataset
        """
        np.random.seed(42)  # For reproducibility
        
        if dataset_type == "financial":
            # Generate financial transaction data
            base_amounts = np.random.lognormal(3, 1, size)
            timestamps = pd.date_range(
                start=datetime.now() - timedelta(days=30),
                periods=size,
                freq='1min'
            )
            
            # Add seasonal patterns
            hour_pattern = np.sin(2 * np.pi * timestamps.hour / 24)
            day_pattern = np.sin(2 * np.pi * timestamps.dayofweek / 7)
            amounts = base_amounts * (1 + 0.3 * hour_pattern + 0.2 * day_pattern)
            
            # Add anomalies
            n_anomalies = int(size * anomaly_rate)
            anomaly_indices = np.random.choice(size, n_anomalies, replace=False)
            amounts[anomaly_indices] *= np.random.choice([0.1, 10], n_anomalies)
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'amount': amounts,
                'merchant_category': np.random.choice(['retail', 'gas', 'restaurant', 'online'], size),
                'card_type': np.random.choice(['credit', 'debit'], size),
                'is_anomaly': False
            })
            df.loc[anomaly_indices, 'is_anomaly'] = True
            
        elif dataset_type == "system_metrics":
            # Generate system metrics data
            timestamps = pd.date_range(
                start=datetime.now() - timedelta(hours=24),
                periods=size,
                freq='1min'
            )
            
            # Normal patterns
            cpu_base = 40 + 30 * np.sin(2 * np.pi * np.arange(size) / 144)  # Daily pattern
            memory_base = 60 + 20 * np.sin(2 * np.pi * np.arange(size) / 144 + np.pi/4)
            
            # Add noise
            cpu_usage = cpu_base + np.random.normal(0, 5, size)
            memory_usage = memory_base + np.random.normal(0, 3, size)
            
            # Add anomalies
            n_anomalies = int(size * anomaly_rate)
            anomaly_indices = np.random.choice(size, n_anomalies, replace=False)
            cpu_usage[anomaly_indices] += np.random.uniform(30, 50, n_anomalies)
            memory_usage[anomaly_indices] += np.random.uniform(20, 40, n_anomalies)
            
            # Clip to realistic ranges
            cpu_usage = np.clip(cpu_usage, 0, 100)
            memory_usage = np.clip(memory_usage, 0, 100)
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_io': 100 + 50 * np.random.exponential(1, size),
                'network_io': 50 + 30 * np.random.gamma(2, 2, size),
                'is_anomaly': False
            })
            df.loc[anomaly_indices, 'is_anomaly'] = True
            
        else:
            # Generic numerical dataset
            data = np.random.normal(0, 1, (size, 3))
            
            # Add anomalies
            n_anomalies = int(size * anomaly_rate)
            anomaly_indices = np.random.choice(size, n_anomalies, replace=False)
            data[anomaly_indices] += np.random.normal(0, 3, (n_anomalies, 3))
            
            df = pd.DataFrame(data, columns=['feature_1', 'feature_2', 'feature_3'])
            df['timestamp'] = pd.date_range(start=datetime.now() - timedelta(hours=1), periods=size, freq='1s')
            df['is_anomaly'] = False
            df.loc[anomaly_indices, 'is_anomaly'] = True
        
        return df
    
    def display_data_summary(self, df: pd.DataFrame):
        """Display data summary in Streamlit"""
        st.markdown("#### 📊 Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        
        with col3:
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        with col4:
            if 'is_anomaly' in df.columns:
                anomaly_count = df['is_anomaly'].sum()
                st.metric("Anomalies", anomaly_count)
        
        # Data types and basic statistics
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Data Types")
                dtypes_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Type': df.dtypes.values.astype(str)
                })
                st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("##### Basic Statistics")
                if len(numeric_cols) > 0:
                    stats_df = df[numeric_cols].describe().round(2)
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("No numeric columns found.")

# Utility functions for Streamlit integration
def validate_and_display_text_input(text: str) -> bool:
    """Validate text input and display results in Streamlit"""
    validator = DataValidator()
    result = validator.validate_text_input(text)
    
    if result.errors:
        for error in result.errors:
            st.error(f"❌ {error}")
    
    if result.warnings:
        for warning in result.warnings:
            st.warning(f"⚠️ {warning}")
    
    if result.is_valid:
        st.success("✅ Text validation passed")
        
        # Display summary
        summary = result.summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Characters", summary['length'])
        with col2:
            st.metric("Words", summary['word_count'])
        with col3:
            st.metric("Sentences", summary['sentence_count'])
    
    return result.is_valid

def validate_and_display_numerical_input(data: Union[List[float], np.ndarray]) -> bool:
    """Validate numerical input and display results in Streamlit"""
    validator = DataValidator()
    result = validator.validate_numerical_data(data)
    
    if result.errors:
        for error in result.errors:
            st.error(f"❌ {error}")
    
    if result.warnings:
        for warning in result.warnings:
            st.warning(f"⚠️ {warning}")
    
    if result.is_valid:
        st.success("✅ Data validation passed")
        
        # Display summary
        summary = result.summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Count", summary['count'])
        with col2:
            st.metric("Mean", f"{summary['mean']:.2f}" if summary['mean'] else "N/A")
        with col3:
            st.metric("Std Dev", f"{summary['std']:.2f}" if summary['std'] else "N/A")
        with col4:
            st.metric("Outliers", summary['outlier_count'])
    
    return result.is_valid