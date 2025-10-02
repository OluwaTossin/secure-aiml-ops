"""
Model Client Interface
====================

Client interface for interacting with ML models and inference endpoints.
Provides unified API for different model types and deployment methods.
"""

import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
import json
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

class ModelInferenceError(Exception):
    """Custom exception for model inference errors"""
    pass

class BaseModelClient:
    """Base model client with common functionality"""
    
    def __init__(self, 
                 base_url: str,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize model client
        
        Args:
            base_url: Base URL for model API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SecureAIMLOps/1.0.0'
        })
        
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'X-API-Key': api_key
            })
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     data: Optional[Dict] = None,
                     params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request payload
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            ModelInferenceError: If request fails after retries
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise ModelInferenceError(f"Request timed out after {self.max_retries} attempts")
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise ModelInferenceError(f"HTTP error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt == self.max_retries - 1:
                    raise ModelInferenceError(f"Request failed: {str(e)}")
                
            time.sleep(0.5)  # Brief pause between retries
    
    def health_check(self) -> Dict[str, Any]:
        """Check model service health"""
        try:
            return self._make_request('GET', '/health')
        except ModelInferenceError:
            return {'status': 'unhealthy', 'timestamp': datetime.now().isoformat()}

class TextSummarizationClient(BaseModelClient):
    """Client for text summarization models"""
    
    def summarize(self, 
                 text: str,
                 max_length: Optional[int] = None,
                 min_length: Optional[int] = None,
                 temperature: float = 1.0,
                 model_name: str = "default") -> Dict[str, Any]:
        """
        Generate text summary
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            temperature: Sampling temperature
            model_name: Specific model to use
            
        Returns:
            Summary response with text and metadata
        """
        payload = {
            "text": text,
            "model": model_name,
            "parameters": {
                "temperature": temperature
            }
        }
        
        if max_length:
            payload["parameters"]["max_length"] = max_length
        if min_length:
            payload["parameters"]["min_length"] = min_length
            
        try:
            response = self._make_request('POST', '/v1/summarize', data=payload)
            
            # Add processing metadata
            response['processing_time'] = response.get('processing_time', 0)
            response['confidence'] = response.get('confidence', 0.0)
            response['model_used'] = response.get('model', model_name)
            response['timestamp'] = datetime.now().isoformat()
            
            return response
            
        except ModelInferenceError as e:
            logger.error(f"Summarization failed: {e}")
            # Return fallback response for demo
            return {
                "summary": "Summary generation failed. Please try again.",
                "confidence": 0.0,
                "processing_time": 0,
                "model_used": model_name,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Batch summarization for multiple texts"""
        results = []
        for text in texts:
            result = self.summarize(text, **kwargs)
            results.append(result)
        return results

class AnomalyDetectionClient(BaseModelClient):
    """Client for anomaly detection models"""
    
    def detect_anomalies(self, 
                        data: Union[List[float], Dict[str, float]],
                        model_name: str = "isolation_forest",
                        threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect anomalies in data
        
        Args:
            data: Input data for anomaly detection
            model_name: Specific model to use
            threshold: Anomaly threshold
            
        Returns:
            Detection results with scores and predictions
        """
        payload = {
            "data": data,
            "model": model_name,
            "threshold": threshold
        }
        
        try:
            response = self._make_request('POST', '/v1/detect', data=payload)
            
            # Add processing metadata
            response['processing_time'] = response.get('processing_time', 0)
            response['model_used'] = response.get('model', model_name)
            response['timestamp'] = datetime.now().isoformat()
            
            return response
            
        except ModelInferenceError as e:
            logger.error(f"Anomaly detection failed: {e}")
            # Return fallback response for demo
            if isinstance(data, list):
                n_points = len(data)
            else:
                n_points = 1
                
            return {
                "anomalies": [False] * n_points,
                "scores": [0.1] * n_points,
                "threshold": threshold,
                "model_used": model_name,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def stream_detection(self, data_stream, **kwargs):
        """Real-time anomaly detection for streaming data"""
        for data_point in data_stream:
            yield self.detect_anomalies(data_point, **kwargs)

class SentimentAnalysisClient(BaseModelClient):
    """Client for sentiment analysis models"""
    
    def analyze_sentiment(self, 
                         text: str,
                         model_name: str = "transformer") -> Dict[str, Any]:
        """
        Analyze text sentiment
        
        Args:
            text: Input text for sentiment analysis
            model_name: Specific model to use
            
        Returns:
            Sentiment analysis results
        """
        payload = {
            "text": text,
            "model": model_name
        }
        
        try:
            response = self._make_request('POST', '/v1/sentiment', data=payload)
            
            response['processing_time'] = response.get('processing_time', 0)
            response['model_used'] = response.get('model', model_name)
            response['timestamp'] = datetime.now().isoformat()
            
            return response
            
        except ModelInferenceError as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                "model_used": model_name,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

class ModelClient:
    """Unified model client interface"""
    
    def __init__(self, 
                 base_url: str,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize unified model client
        
        Args:
            base_url: Base URL for model API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize specialized clients
        self.summarization = TextSummarizationClient(base_url, api_key, timeout, max_retries)
        self.anomaly_detection = AnomalyDetectionClient(base_url, api_key, timeout, max_retries)
        self.sentiment_analysis = SentimentAnalysisClient(base_url, api_key, timeout, max_retries)
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by type"""
        try:
            response = self.summarization._make_request('GET', '/v1/models')
            return response.get('models', {})
        except ModelInferenceError:
            # Return fallback model list for demo
            return {
                "summarization": ["t5-base", "t5-large", "bart-large", "pegasus"],
                "anomaly_detection": ["isolation_forest", "lof", "one_class_svm", "autoencoder"],
                "sentiment_analysis": ["transformer", "lstm", "cnn"]
            }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        try:
            response = self.summarization._make_request('GET', f'/v1/models/{model_name}')
            return response
        except ModelInferenceError:
            # Return fallback model info for demo
            return {
                "name": model_name,
                "type": "unknown",
                "version": "1.0.0",
                "description": "Model information not available",
                "parameters": {},
                "performance": {
                    "accuracy": 0.85,
                    "latency_ms": 100,
                    "throughput_rps": 50
                }
            }
    
    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all model services"""
        return {
            "summarization": self.summarization.health_check(),
            "anomaly_detection": self.anomaly_detection.health_check(),
            "sentiment_analysis": self.sentiment_analysis.health_check()
        }

class MockModelClient:
    """Mock model client for development and testing"""
    
    def __init__(self):
        """Initialize mock client with simulated responses"""
        self.summarization = MockSummarizationClient()
        self.anomaly_detection = MockAnomalyDetectionClient()
        self.sentiment_analysis = MockSentimentAnalysisClient()
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get mock model list"""
        return {
            "summarization": ["t5-base", "t5-large", "bart-large", "pegasus", "distilbart"],
            "anomaly_detection": ["isolation_forest", "lof", "one_class_svm", "autoencoder", "z_score"],
            "sentiment_analysis": ["transformer", "lstm", "cnn", "bert", "roberta"]
        }
    
    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Mock health check"""
        return {
            "summarization": {"status": "healthy", "timestamp": datetime.now().isoformat()},
            "anomaly_detection": {"status": "healthy", "timestamp": datetime.now().isoformat()},
            "sentiment_analysis": {"status": "healthy", "timestamp": datetime.now().isoformat()}
        }

class MockSummarizationClient:
    """Mock summarization client"""
    
    def summarize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate mock summary"""
        time.sleep(0.5)  # Simulate processing time
        
        word_count = len(text.split())
        summary_length = max(10, word_count // 4)
        
        summaries = [
            f"This comprehensive analysis of {word_count} words explores key concepts and findings.",
            f"The document presents important insights and recommendations based on detailed research.",
            f"Key findings from this {word_count}-word analysis reveal significant patterns and trends."
        ]
        
        return {
            "summary": summaries[hash(text) % len(summaries)],
            "confidence": 0.8 + 0.15 * np.random.rand(),
            "processing_time": 0.3 + 0.7 * np.random.rand(),
            "model_used": kwargs.get("model_name", "t5-base"),
            "timestamp": datetime.now().isoformat(),
            "input_length": word_count,
            "summary_length": summary_length
        }

class MockAnomalyDetectionClient:
    """Mock anomaly detection client"""
    
    def detect_anomalies(self, data: Union[List[float], Dict[str, float]], **kwargs) -> Dict[str, Any]:
        """Generate mock anomaly detection results"""
        time.sleep(0.2)  # Simulate processing time
        
        if isinstance(data, dict):
            data = list(data.values())
        
        n_points = len(data)
        anomaly_rate = 0.05  # 5% anomaly rate
        
        # Generate random anomalies
        anomalies = np.random.random(n_points) < anomaly_rate
        scores = np.random.random(n_points)
        
        # Make anomalous points have higher scores
        scores[anomalies] = 0.7 + 0.3 * np.random.random(np.sum(anomalies))
        
        return {
            "anomalies": anomalies.tolist(),
            "scores": scores.tolist(),
            "threshold": kwargs.get("threshold", 0.5),
            "n_anomalies": int(np.sum(anomalies)),
            "anomaly_rate": float(np.mean(anomalies)),
            "model_used": kwargs.get("model_name", "isolation_forest"),
            "processing_time": 0.1 + 0.3 * np.random.rand(),
            "timestamp": datetime.now().isoformat()
        }

class MockSentimentAnalysisClient:
    """Mock sentiment analysis client"""
    
    def analyze_sentiment(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate mock sentiment analysis"""
        time.sleep(0.3)  # Simulate processing time
        
        # Simple mock sentiment based on text characteristics
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing", "poor"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = 0.7 + 0.3 * np.random.rand()
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = 0.7 + 0.3 * np.random.rand()
        else:
            sentiment = "neutral"
            confidence = 0.5 + 0.3 * np.random.rand()
        
        # Generate score distribution
        scores = np.random.dirichlet([1, 1, 1])  # Random distribution summing to 1
        score_map = {"positive": scores[0], "neutral": scores[1], "negative": scores[2]}
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": score_map,
            "model_used": kwargs.get("model_name", "transformer"),
            "processing_time": 0.2 + 0.4 * np.random.rand(),
            "timestamp": datetime.now().isoformat()
        }

# Factory function to create model client
def create_model_client(config, use_mock: bool = False) -> Union[ModelClient, MockModelClient]:
    """
    Create model client from configuration
    
    Args:
        config: Application configuration
        use_mock: Whether to use mock client for development
        
    Returns:
        Model client instance
    """
    if use_mock or config.is_development:
        return MockModelClient()
    else:
        return ModelClient(
            base_url=config.model.api_base_url,
            api_key=config.model.api_key,
            timeout=config.model.timeout,
            max_retries=config.model.max_retries
        )