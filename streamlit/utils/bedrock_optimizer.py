"""
Bedrock Performance Optimization Module
=======================================

Advanced caching, streaming, and cost optimization for AWS Bedrock models.
Implements intelligent response caching, streaming capabilities, and usage analytics.

Author: Secure AI/ML Ops Team
Version: 1.0.0
"""

import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, Generator, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError
import streamlit as st

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached response entry"""
    response: Dict[str, Any]
    timestamp: float
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_estimate: float
    access_count: int = 0
    last_accessed: float = 0.0

@dataclass
class UsageMetrics:
    """Track usage metrics for cost optimization"""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time: float = 0.0
    model_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.model_usage is None:
            self.model_usage = defaultdict(int)

class BedrockOptimizer:
    """Advanced optimization layer for AWS Bedrock interactions"""
    
    # Cost estimates per 1000 tokens (approximate)
    MODEL_COSTS = {
        "anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 0.015, "output": 0.075},
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
        "amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
        "amazon.nova-lite-v1:0": {"input": 0.0002, "output": 0.0008},
        "amazon.nova-micro-v1:0": {"input": 0.000035, "output": 0.00014},
        "amazon.titan-text-express-v1": {"input": 0.0002, "output": 0.0006},
        "mistral.mistral-large-2402-v1:0": {"input": 0.004, "output": 0.012}
    }
    
    def __init__(self, cache_ttl: int = 3600, max_cache_size: int = 1000):
        """
        Initialize the Bedrock optimizer
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
            max_cache_size: Maximum number of cached responses
        """
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, CacheEntry] = {}
        self.metrics = UsageMetrics()
        self.lock = threading.Lock()
        
        # Initialize session state for persistence
        if "bedrock_cache" not in st.session_state:
            st.session_state.bedrock_cache = {}
        if "bedrock_metrics" not in st.session_state:
            st.session_state.bedrock_metrics = UsageMetrics()
        
        self._load_from_session()
    
    def _load_from_session(self):
        """Load cache and metrics from Streamlit session state"""
        self.cache = st.session_state.bedrock_cache
        self.metrics = st.session_state.bedrock_metrics
    
    def _save_to_session(self):
        """Save cache and metrics to Streamlit session state"""
        st.session_state.bedrock_cache = self.cache
        st.session_state.bedrock_metrics = self.metrics
    
    def _create_cache_key(self, prompt: str, model_id: str, params: Dict[str, Any]) -> str:
        """Create a unique cache key"""
        # Normalize parameters for consistent caching
        normalized_params = {
            "temperature": round(params.get("temperature", 0.7), 2),
            "max_tokens": params.get("max_tokens", 2000),
            "top_p": round(params.get("top_p", 0.9), 2)
        }
        
        cache_data = f"{prompt}|{model_id}|{json.dumps(normalized_params, sort_keys=True)}"
        return hashlib.sha256(cache_data.encode()).hexdigest()[:16]
    
    def _estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost of a request"""
        if model_id not in self.MODEL_COSTS:
            return 0.0
        
        costs = self.MODEL_COSTS[model_id]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost
    
    def _count_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for most models)"""
        return len(text) // 4
    
    def _cleanup_cache(self):
        """Remove expired entries and enforce size limits"""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = []
        for key, entry in self.cache.items():
            if current_time - entry.timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        # Enforce size limit (remove least recently accessed)
        if len(self.cache) > self.max_cache_size:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            entries_to_remove = len(self.cache) - self.max_cache_size
            for key, _ in sorted_entries[:entries_to_remove]:
                del self.cache[key]
    
    def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if available and valid"""
        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                current_time = time.time()
                
                if current_time - entry.timestamp < self.cache_ttl:
                    # Update access metrics
                    entry.access_count += 1
                    entry.last_accessed = current_time
                    self.metrics.cache_hits += 1
                    self._save_to_session()
                    
                    return {
                        **entry.response,
                        "from_cache": True,
                        "cache_age": current_time - entry.timestamp,
                        "access_count": entry.access_count
                    }
            
            self.metrics.cache_misses += 1
            return None
    
    def cache_response(self, cache_key: str, response: Dict[str, Any], 
                      model_id: str, prompt: str):
        """Cache a response with metadata"""
        with self.lock:
            input_tokens = self._count_tokens(prompt)
            output_tokens = self._count_tokens(response.get("text", ""))
            cost_estimate = self._estimate_cost(model_id, input_tokens, output_tokens)
            
            entry = CacheEntry(
                response=response,
                timestamp=time.time(),
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_estimate=cost_estimate,
                access_count=1,
                last_accessed=time.time()
            )
            
            self.cache[cache_key] = entry
            self._cleanup_cache()
            self._save_to_session()
    
    def update_metrics(self, model_id: str, input_tokens: int, output_tokens: int, 
                      response_time: float, from_cache: bool = False):
        """Update usage metrics"""
        with self.lock:
            self.metrics.total_requests += 1
            self.metrics.model_usage[model_id] += 1
            
            if not from_cache:
                self.metrics.total_input_tokens += input_tokens
                self.metrics.total_output_tokens += output_tokens
                self.metrics.total_cost += self._estimate_cost(model_id, input_tokens, output_tokens)
            
            # Update rolling average response time
            if self.metrics.average_response_time == 0:
                self.metrics.average_response_time = response_time
            else:
                self.metrics.average_response_time = (
                    self.metrics.average_response_time * 0.9 + response_time * 0.1
                )
            
            self._save_to_session()
    
    def get_cost_optimization_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        cache_hit_rate = (
            self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)
        )
        
        if cache_hit_rate < 0.3:
            recommendations.append(
                "💡 Low cache hit rate ({:.1%}). Consider asking similar questions to benefit from caching.".format(cache_hit_rate)
            )
        
        # Model usage analysis
        if self.metrics.model_usage:
            most_used_model = max(self.metrics.model_usage, key=self.metrics.model_usage.get)
            
            # Suggest cheaper alternatives for high-usage expensive models
            expensive_models = [
                "anthropic.claude-sonnet-4-5-20250929-v1:0",
                "anthropic.claude-3-5-sonnet-20240620-v1:0"
            ]
            
            if most_used_model in expensive_models and self.metrics.model_usage[most_used_model] > 10:
                recommendations.append(
                    f"💰 Consider using Claude 3 Haiku or Amazon Nova Micro for simpler queries to reduce costs."
                )
        
        if self.metrics.total_cost > 10:
            recommendations.append(
                f"📊 High usage detected (${self.metrics.total_cost:.2f}). Monitor usage patterns for optimization."
            )
        
        if len(recommendations) == 0:
            recommendations.append("✅ Your usage patterns look optimized!")
        
        return recommendations
    
    def get_usage_analytics(self) -> Dict[str, Any]:
        """Get comprehensive usage analytics"""
        cache_hit_rate = (
            self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)
        )
        
        return {
            "total_requests": self.metrics.total_requests,
            "total_tokens": self.metrics.total_input_tokens + self.metrics.total_output_tokens,
            "estimated_cost": self.metrics.total_cost,
            "cache_hit_rate": cache_hit_rate,
            "average_response_time": self.metrics.average_response_time,
            "most_used_model": max(self.metrics.model_usage, key=self.metrics.model_usage.get) if self.metrics.model_usage else "None",
            "model_distribution": dict(self.metrics.model_usage),
            "cache_size": len(self.cache),
            "cost_breakdown": self._get_cost_breakdown()
        }
    
    def _get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by model"""
        breakdown = defaultdict(float)
        for entry in self.cache.values():
            breakdown[entry.model_id] += entry.cost_estimate
        return dict(breakdown)
    
    def clear_cache(self) -> int:
        """Clear all cached responses and return count of cleared entries"""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            self._save_to_session()
            return count
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics for analysis"""
        return {
            "metrics": asdict(self.metrics),
            "cache_stats": {
                "total_entries": len(self.cache),
                "oldest_entry": min([e.timestamp for e in self.cache.values()]) if self.cache else 0,
                "most_accessed": max([e.access_count for e in self.cache.values()]) if self.cache else 0
            },
            "export_timestamp": datetime.now().isoformat()
        }

class StreamingResponseHandler:
    """Handle streaming responses with optimization"""
    
    def __init__(self, optimizer: BedrockOptimizer):
        self.optimizer = optimizer
    
    def stream_with_metrics(self, bedrock_client, model_id: str, body: Dict[str, Any], 
                           cache_key: str, prompt: str) -> Generator[str, None, Dict[str, Any]]:
        """Stream response while collecting metrics"""
        start_time = time.time()
        full_response = ""
        
        try:
            response = bedrock_client.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            for event in response['body']:
                chunk = json.loads(event['chunk']['bytes'])
                
                # Handle different model response formats
                text_chunk = ""
                if "anthropic" in model_id:
                    if chunk.get('type') == 'content_block_delta':
                        text_chunk = chunk.get('delta', {}).get('text', '')
                elif "amazon" in model_id and "nova" in model_id:
                    if 'contentBlockDelta' in chunk:
                        text_chunk = chunk['contentBlockDelta']['delta']['text']
                
                if text_chunk:
                    full_response += text_chunk
                    yield text_chunk
            
            # Calculate metrics
            response_time = time.time() - start_time
            input_tokens = self.optimizer._count_tokens(prompt)
            output_tokens = self.optimizer._count_tokens(full_response)
            
            # Update metrics
            self.optimizer.update_metrics(model_id, input_tokens, output_tokens, response_time)
            
            # Cache the complete response
            response_data = {
                "text": full_response,
                "model": model_id,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                },
                "response_time": response_time,
                "streamed": True
            }
            
            self.optimizer.cache_response(cache_key, response_data, model_id, prompt)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Streaming error for model {model_id}: {str(e)}")
            yield f"Error: {str(e)}"
            return {"error": str(e)}

# Global optimizer instance
_optimizer = None

def get_optimizer() -> BedrockOptimizer:
    """Get the global optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = BedrockOptimizer()
    return _optimizer

def reset_optimizer():
    """Reset the global optimizer (for testing)"""
    global _optimizer
    _optimizer = None