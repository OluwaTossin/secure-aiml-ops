import streamlit as st
import boto3
import json
from datetime import datetime
import time
import logging
from typing import List, Dict, Any
import hashlib
import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bedrock_optimizer import BedrockOptimizer, StreamingResponseHandler, get_optimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockChatbot:
    """Advanced AI Chatbot with AWS Bedrock integration and optimization"""
    
    def __init__(self):
        self.bedrock_client = boto3.client('bedrock-runtime', region_name='eu-west-1')
        self.optimizer = get_optimizer()
        self.streaming_handler = StreamingResponseHandler(self.optimizer)
        self.models = {
            "Claude Sonnet 4.5": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0", 
            "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "Amazon Nova Pro": "amazon.nova-pro-v1:0",
            "Amazon Nova Lite": "amazon.nova-lite-v1:0",
            "Amazon Nova Micro": "amazon.nova-micro-v1:0",
            "Titan Text Express": "amazon.titan-text-express-v1",
            "Mistral Large": "mistral.mistral-large-2402-v1:0"
        }
        
        # Initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
    
    def _call_anthropic_model(self, model_id: str, prompt: str, params: Dict) -> Dict[str, Any]:
        """Call Anthropic Claude models"""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": params.get("max_tokens", 2000),
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read())
        return {
            "text": result['content'][0]['text'],
            "usage": result.get('usage', {}),
            "model": model_id
        }
    
    def _call_amazon_model(self, model_id: str, prompt: str, params: Dict) -> Dict[str, Any]:
        """Call Amazon models (Nova, Titan)"""
        if "nova" in model_id:
            # Nova models use Claude-like format
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": params.get("max_tokens", 2000),
                    "temperature": params.get("temperature", 0.7),
                    "topP": params.get("top_p", 0.9)
                }
            }
        else:
            # Titan models
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": params.get("max_tokens", 2000),
                    "temperature": params.get("temperature", 0.7),
                    "topP": params.get("top_p", 0.9)
                }
            }
        
        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read())
        
        if "nova" in model_id:
            return {
                "text": result['output']['message']['content'][0]['text'],
                "usage": result.get('usage', {}),
                "model": model_id
            }
        else:
            return {
                "text": result['results'][0]['outputText'],
                "usage": result.get('usage', {}),
                "model": model_id
            }
    
    def _call_mistral_model(self, model_id: str, prompt: str, params: Dict) -> Dict[str, Any]:
        """Call Mistral models"""
        body = {
            "prompt": f"<s>[INST] {prompt} [/INST]",
            "max_tokens": params.get("max_tokens", 2000),
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9)
        }
        
        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read())
        return {
            "text": result['outputs'][0]['text'],
            "usage": result.get('usage', {}),
            "model": model_id
        }
    
    def generate_response(self, prompt: str, model_name: str, params: Dict) -> Dict[str, Any]:
        """Generate response from selected model with optimization and caching"""
        model_id = self.models[model_name]
        
        # Create cache key using optimizer
        cache_key = self.optimizer._create_cache_key(prompt, model_id, params)
        
        # Check cache first
        cached_response = self.optimizer.get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        try:
            start_time = time.time()
            
            # Route to appropriate model handler
            if "anthropic" in model_id:
                response = self._call_anthropic_model(model_id, prompt, params)
            elif "amazon" in model_id:
                response = self._call_amazon_model(model_id, prompt, params)
            elif "mistral" in model_id:
                response = self._call_mistral_model(model_id, prompt, params)
            else:
                raise ValueError(f"Unsupported model: {model_id}")
            
            response_time = time.time() - start_time
            response["response_time"] = response_time
            response["from_cache"] = False
            
            # Update metrics and cache response
            input_tokens = self.optimizer._count_tokens(prompt)
            output_tokens = self.optimizer._count_tokens(response["text"])
            self.optimizer.update_metrics(model_id, input_tokens, output_tokens, response_time)
            self.optimizer.cache_response(cache_key, response, model_id, prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling model {model_id}: {str(e)}")
            return {
                "text": f"Error: {str(e)}",
                "usage": {},
                "model": model_id,
                "from_cache": False,
                "error": True
            }
    
    def stream_response(self, prompt: str, model_name: str, params: Dict):
        """Stream response for real-time interaction with optimization"""
        model_id = self.models[model_name]
        
        # Check cache first for streaming too
        cache_key = self.optimizer._create_cache_key(prompt, model_id, params)
        cached_response = self.optimizer.get_cached_response(cache_key)
        if cached_response:
            # Simulate streaming for cached responses
            text = cached_response["text"]
            for i in range(0, len(text), 10):
                yield text[i:i+10]
                time.sleep(0.02)  # Small delay to simulate streaming
            return cached_response
        
        try:
            if "anthropic" in model_id:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": params.get("max_tokens", 2000),
                    "temperature": params.get("temperature", 0.7),
                    "top_p": params.get("top_p", 0.9),
                    "messages": [{"role": "user", "content": prompt}]
                }
            elif "amazon" in model_id and "nova" in model_id:
                body = {
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {
                        "maxTokens": params.get("max_tokens", 2000),
                        "temperature": params.get("temperature", 0.7),
                        "topP": params.get("top_p", 0.9)
                    }
                }
            else:
                # Fall back to regular response for non-streaming models
                response = self.generate_response(prompt, model_name, params)
                text = response.get("text", "")
                for i in range(0, len(text), 10):
                    yield text[i:i+10]
                    time.sleep(0.02)
                return response
            
            # Use the optimized streaming handler
            for chunk in self.streaming_handler.stream_with_metrics(
                self.bedrock_client, model_id, body, cache_key, prompt
            ):
                yield chunk
            
        except Exception as e:
            logger.error(f"Error streaming from model {model_id}: {str(e)}")
            yield f"Error: {str(e)}"

def render_chatbot_page():
    """Render the AI Chatbot page"""
    st.title("🤖 AI Chatbot")
    st.markdown("### Powered by AWS Bedrock Foundation Models")
    
    chatbot = BedrockChatbot()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Model Configuration")
        
        # Model selection
        selected_model = st.selectbox(
            "Choose AI Model:",
            options=list(chatbot.models.keys()),
            index=0,
            help="Select the foundation model for conversation"
        )
        
        st.markdown("---")
        st.subheader("Parameters")
        
        # Model parameters
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness. Lower = more focused, Higher = more creative"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=2000,
            step=100,
            help="Maximum response length"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="Nucleus sampling parameter"
        )
        
        # Streaming toggle
        use_streaming = st.toggle(
            "Enable Streaming",
            value=True,
            help="Stream responses in real-time"
        )
        
        st.markdown("---")
        
        # Chat controls
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Analytics and optimization
        st.markdown("---")
        st.subheader("📊 Analytics & Optimization")
        
        analytics = chatbot.optimizer.get_usage_analytics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Requests", analytics["total_requests"])
            st.metric("Cache Hit Rate", f"{analytics['cache_hit_rate']:.1%}")
        
        with col2:
            st.metric("Est. Cost", f"${analytics['estimated_cost']:.3f}")
            st.metric("Avg Response", f"{analytics['average_response_time']:.1f}s")
        
        # Cost optimization recommendations
        with st.expander("💡 Optimization Tips"):
            recommendations = chatbot.optimizer.get_cost_optimization_recommendations()
            for rec in recommendations:
                st.markdown(f"• {rec}")
        
        # Cache management
        if st.button("🧹 Clear Cache"):
            cleared = chatbot.optimizer.clear_cache()
            st.success(f"Cleared {cleared} cached responses!")
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Chat input
        prompt = st.chat_input("Ask me anything...", key="chat_input")
    
    with col2:
        if st.button("🎲 Random Question", help="Generate a random interesting question"):
            random_questions = [
                "What are the most promising applications of AI in healthcare?",
                "Explain quantum computing in simple terms",
                "What are the ethical implications of autonomous vehicles?",
                "How can we make AI more sustainable and environmentally friendly?",
                "What role will AI play in education in the next decade?",
                "Describe the future of human-AI collaboration",
                "What are the biggest challenges in natural language processing?",
                "How can AI help solve climate change?"
            ]
            import random
            prompt = random.choice(random_questions)
            st.session_state.random_prompt = prompt
    
    # Use random prompt if generated
    if hasattr(st.session_state, 'random_prompt'):
        prompt = st.session_state.random_prompt
        delattr(st.session_state, 'random_prompt')
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                with st.expander("ℹ️ Response Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model", metadata.get("model", "").split(".")[-1])
                    with col2:
                        if "usage" in metadata and "input_tokens" in metadata["usage"]:
                            st.metric("Input Tokens", metadata["usage"]["input_tokens"])
                    with col3:
                        if "usage" in metadata and "output_tokens" in metadata["usage"]:
                            st.metric("Output Tokens", metadata["usage"]["output_tokens"])
                    
                    if metadata.get("from_cache"):
                        st.success("🚀 Served from cache")
                    
                    st.caption(f"Generated at: {metadata.get('timestamp', 'Unknown')}")
    
    # Process new message
    if prompt:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            params = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            
            if use_streaming:
                # Streaming response
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    for chunk in chatbot.stream_response(prompt, selected_model, params):
                        if isinstance(chunk, str):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "metadata": {
                            "model": selected_model,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "streamed": True
                        }
                    })
                    
                except Exception as e:
                    error_msg = f"Streaming error: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "metadata": {"error": True}
                    })
            else:
                # Regular response
                with st.spinner(f"Generating response with {selected_model}..."):
                    response = chatbot.generate_response(prompt, selected_model, params)
                
                if response.get("error"):
                    st.error(response["text"])
                else:
                    st.write(response["text"])
                
                # Add to history with metadata
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["text"],
                    "metadata": {
                        "model": selected_model,
                        "usage": response.get("usage", {}),
                        "from_cache": response.get("from_cache", False),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                })
        
        st.rerun()

if __name__ == "__main__":
    render_chatbot_page()