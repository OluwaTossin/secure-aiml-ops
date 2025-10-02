"""
Model Management Dashboard
========================

Comprehensive model lifecycle management interface with deployment,
monitoring, and version control capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json

st.set_page_config(
    page_title="Model Management - Secure AI/ML Ops",
    page_icon="⚙️",
    layout="wide"
)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = [
        {
            "id": "model_001",
            "name": "text-summarization-v2.1",
            "type": "Text Summarization",
            "status": "deployed",
            "version": "2.1.0",
            "accuracy": 0.92,
            "latency": 89,
            "throughput": 1250,
            "deployment_date": "2024-01-15",
            "last_updated": "2024-01-20",
            "resource_usage": {"cpu": 45, "memory": 2.1, "gpu": 0.8},
            "endpoint": "https://api.secure-aiml.com/v1/summarize",
            "framework": "PyTorch",
            "size_mb": 850
        },
        {
            "id": "model_002", 
            "name": "anomaly-detection-v1.3",
            "type": "Anomaly Detection",
            "status": "deployed",
            "version": "1.3.2",
            "accuracy": 0.89,
            "latency": 45,
            "throughput": 2100,
            "deployment_date": "2024-01-10",
            "last_updated": "2024-01-18",
            "resource_usage": {"cpu": 35, "memory": 1.8, "gpu": 0.6},
            "endpoint": "https://api.secure-aiml.com/v1/detect",
            "framework": "Scikit-learn",
            "size_mb": 120
        },
        {
            "id": "model_003",
            "name": "sentiment-analysis-v3.0",
            "type": "Sentiment Analysis",
            "status": "staging",
            "version": "3.0.0",
            "accuracy": 0.94,
            "latency": 32,
            "throughput": 3200,
            "deployment_date": "2024-01-22",
            "last_updated": "2024-01-22",
            "resource_usage": {"cpu": 28, "memory": 1.2, "gpu": 0.4},
            "endpoint": "https://staging.secure-aiml.com/v1/sentiment",
            "framework": "TensorFlow",
            "size_mb": 420
        },
        {
            "id": "model_004",
            "name": "recommendation-engine-v1.1",
            "type": "Recommendation",
            "status": "training",
            "version": "1.1.0",
            "accuracy": 0.87,
            "latency": 156,
            "throughput": 850,
            "deployment_date": None,
            "last_updated": "2024-01-23",
            "resource_usage": {"cpu": 0, "memory": 0, "gpu": 0},
            "endpoint": None,
            "framework": "PyTorch",
            "size_mb": 1200
        }
    ]

# Page header
st.markdown("""
# ⚙️ Model Management

Comprehensive model lifecycle management with deployment, monitoring, and version control.
""")

# Sidebar navigation
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    page_mode = st.radio(
        "Select View",
        ["📊 Overview", "🚀 Deploy Model", "📈 Performance", "🔧 Configuration", "📋 Logs"],
        help="Choose the management view"
    )
    
    st.markdown("### 🎯 Quick Filters")
    
    status_filter = st.multiselect(
        "Filter by Status",
        ["deployed", "staging", "training", "failed", "retired"],
        default=["deployed", "staging", "training"]
    )
    
    type_filter = st.multiselect(
        "Filter by Type",
        ["Text Summarization", "Anomaly Detection", "Sentiment Analysis", "Recommendation"],
        default=["Text Summarization", "Anomaly Detection", "Sentiment Analysis", "Recommendation"]
    )
    
    # Resource monitoring
    st.markdown("### 💾 Resource Overview")
    
    total_cpu = sum(model["resource_usage"]["cpu"] for model in st.session_state.models if model["status"] == "deployed")
    total_memory = sum(model["resource_usage"]["memory"] for model in st.session_state.models if model["status"] == "deployed")
    total_gpu = sum(model["resource_usage"]["gpu"] for model in st.session_state.models if model["status"] == "deployed")
    
    st.metric("Total CPU Usage", f"{total_cpu}%")
    st.metric("Total Memory", f"{total_memory:.1f} GB")
    st.metric("Total GPU Usage", f"{total_gpu:.1f} GB")

# Filter models
filtered_models = [
    model for model in st.session_state.models
    if model["status"] in status_filter and model["type"] in type_filter
]

# Main content based on selected page mode
if page_mode == "📊 Overview":
    
    # Summary metrics
    st.markdown("### 📊 Model Portfolio Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_models = len(st.session_state.models)
        st.metric("Total Models", total_models)
    
    with col2:
        deployed_models = len([m for m in st.session_state.models if m["status"] == "deployed"])
        st.metric("Deployed Models", deployed_models, delta=f"{deployed_models/total_models:.0%}")
    
    with col3:
        avg_accuracy = np.mean([m["accuracy"] for m in st.session_state.models])
        st.metric("Avg Accuracy", f"{avg_accuracy:.1%}", delta="2.3%")
    
    with col4:
        avg_latency = np.mean([m["latency"] for m in st.session_state.models if m["status"] == "deployed"])
        st.metric("Avg Latency", f"{avg_latency:.0f}ms", delta="-12ms", delta_color="inverse")
    
    with col5:
        total_throughput = sum([m["throughput"] for m in st.session_state.models if m["status"] == "deployed"])
        st.metric("Total Throughput", f"{total_throughput:,}/hr", delta="15%")
    
    st.markdown("---")
    
    # Model status overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏷️ Model Registry")
        
        # Create DataFrame for display
        display_data = []
        for model in filtered_models:
            display_data.append({
                "Name": model["name"],
                "Type": model["type"],
                "Version": model["version"],
                "Status": model["status"],
                "Accuracy": f"{model['accuracy']:.1%}",
                "Latency": f"{model['latency']}ms" if model["status"] == "deployed" else "N/A",
                "Last Updated": model["last_updated"],
                "Framework": model["framework"]
            })
        
        df = pd.DataFrame(display_data)
        
        # Style the dataframe
        def color_status(val):
            colors = {
                'deployed': 'background-color: #d4edda; color: #155724',
                'staging': 'background-color: #fff3cd; color: #856404',
                'training': 'background-color: #d1ecf1; color: #0c5460',
                'failed': 'background-color: #f8d7da; color: #721c24',
                'retired': 'background-color: #e2e3e5; color: #383d41'
            }
            return colors.get(val, '')
        
        if not df.empty:
            styled_df = df.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No models match the current filter criteria.")
    
    with col2:
        st.markdown("### 📈 Status Distribution")
        
        status_counts = pd.Series([m["status"] for m in st.session_state.models]).value_counts()
        
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Model Status Distribution",
            color_discrete_map={
                'deployed': '#28a745',
                'staging': '#ffc107',
                'training': '#17a2b8',
                'failed': '#dc3545',
                'retired': '#6c757d'
            }
        )
        fig_status.update_layout(height=300)
        st.plotly_chart(fig_status, use_container_width=True)
        
        st.markdown("### 🔧 Framework Usage")
        
        framework_counts = pd.Series([m["framework"] for m in st.session_state.models]).value_counts()
        
        fig_framework = px.bar(
            x=framework_counts.values,
            y=framework_counts.index,
            orientation='h',
            title="Framework Distribution",
            color=framework_counts.values,
            color_continuous_scale='Blues'
        )
        fig_framework.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_framework, use_container_width=True)

elif page_mode == "🚀 Deploy Model":
    
    st.markdown("### 🚀 Model Deployment")
    
    tab1, tab2, tab3 = st.tabs(["📤 New Deployment", "🔄 Update Model", "🗂️ Model Registry"])
    
    with tab1:
        st.markdown("#### Deploy New Model")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### Model Information")
            
            model_name = st.text_input("Model Name", placeholder="e.g., text-classifier-v1.0")
            model_type = st.selectbox(
                "Model Type",
                ["Text Summarization", "Anomaly Detection", "Sentiment Analysis", "Classification", "Recommendation"]
            )
            model_version = st.text_input("Version", placeholder="e.g., 1.0.0")
            framework = st.selectbox("Framework", ["PyTorch", "TensorFlow", "Scikit-learn", "XGBoost", "ONNX"])
            
            st.markdown("##### Model Files")
            
            model_file = st.file_uploader("Upload Model File", type=['pkl', 'joblib', 'pt', 'h5', 'onnx'])
            config_file = st.file_uploader("Upload Config File", type=['json', 'yaml', 'yml'])
            requirements_file = st.file_uploader("Requirements File", type=['txt'])
        
        with col2:
            st.markdown("##### Deployment Configuration")
            
            environment = st.selectbox("Target Environment", ["staging", "production"])
            
            st.markdown("**Resource Allocation**")
            cpu_cores = st.slider("CPU Cores", 1, 8, 2)
            memory_gb = st.slider("Memory (GB)", 1, 16, 4)
            gpu_enabled = st.checkbox("Enable GPU")
            if gpu_enabled:
                gpu_memory = st.slider("GPU Memory (GB)", 1, 8, 2)
            
            st.markdown("**Scaling Configuration**")
            min_replicas = st.slider("Min Replicas", 1, 5, 1)
            max_replicas = st.slider("Max Replicas", 1, 20, 3)
            target_cpu = st.slider("Target CPU Utilization (%)", 30, 90, 70)
            
            st.markdown("**Health Checks**")
            health_endpoint = st.text_input("Health Check Endpoint", "/health")
            initial_delay = st.number_input("Initial Delay (seconds)", 30, 300, 60)
            
        # Deployment button
        if st.button("🚀 Deploy Model", type="primary", use_container_width=True):
            if model_name and model_type and model_version:
                with st.spinner("Deploying model..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    steps = [
                        "Validating model files...",
                        "Building container image...",
                        "Pushing to ECR...",
                        "Creating deployment manifest...",
                        "Deploying to Kubernetes...",
                        "Configuring load balancer...",
                        "Running health checks..."
                    ]
                    
                    for i, step in enumerate(steps):
                        status_text.text(step)
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(1)
                    
                    # Add new model to registry
                    new_model = {
                        "id": f"model_{len(st.session_state.models) + 1:03d}",
                        "name": model_name,
                        "type": model_type,
                        "status": "staging" if environment == "staging" else "deployed",
                        "version": model_version,
                        "accuracy": 0.85 + 0.1 * np.random.rand(),
                        "latency": 50 + 100 * np.random.rand(),
                        "throughput": 1000 + 1000 * np.random.rand(),
                        "deployment_date": datetime.now().strftime('%Y-%m-%d'),
                        "last_updated": datetime.now().strftime('%Y-%m-%d'),
                        "resource_usage": {"cpu": cpu_cores * 12.5, "memory": memory_gb, "gpu": gpu_memory if gpu_enabled else 0},
                        "endpoint": f"https://{'staging.' if environment == 'staging' else ''}api.secure-aiml.com/v1/{model_name.lower()}",
                        "framework": framework,
                        "size_mb": 200 + 800 * np.random.rand()
                    }
                    
                    st.session_state.models.append(new_model)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Model '{model_name}' deployed successfully to {environment}!")
                    st.balloons()
            else:
                st.error("Please fill in all required fields.")
    
    with tab2:
        st.markdown("#### Update Existing Model")
        
        if st.session_state.models:
            update_model = st.selectbox(
                "Select Model to Update",
                [f"{m['name']} (v{m['version']})" for m in st.session_state.models],
                help="Choose a model to update"
            )
            
            model_index = next(i for i, m in enumerate(st.session_state.models) 
                             if f"{m['name']} (v{m['version']})" == update_model)
            selected_model = st.session_state.models[model_index]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Current Configuration")
                st.info(f"""
                **Name:** {selected_model['name']}
                **Version:** {selected_model['version']}
                **Status:** {selected_model['status']}
                **Framework:** {selected_model['framework']}
                **Accuracy:** {selected_model['accuracy']:.1%}
                **Latency:** {selected_model['latency']:.0f}ms
                """)
            
            with col2:
                st.markdown("##### Update Configuration")
                
                new_version = st.text_input("New Version", selected_model['version'])
                update_type = st.selectbox("Update Type", ["Minor Update", "Major Update", "Hotfix"])
                
                if st.button("🔄 Update Model", use_container_width=True):
                    with st.spinner("Updating model..."):
                        time.sleep(3)
                    
                    # Update model in registry
                    st.session_state.models[model_index]['version'] = new_version
                    st.session_state.models[model_index]['last_updated'] = datetime.now().strftime('%Y-%m-%d')
                    
                    st.success(f"✅ Model updated to version {new_version}!")
        else:
            st.info("No models available for update.")
    
    with tab3:
        st.markdown("#### Model Registry Management")
        
        # Registry actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Import Models", use_container_width=True):
                st.info("Model import functionality coming soon.")
        
        with col2:
            if st.button("📤 Export Registry", use_container_width=True):
                registry_json = json.dumps(st.session_state.models, indent=2, default=str)
                st.download_button(
                    "📄 Download JSON",
                    registry_json,
                    f"model_registry_{datetime.now().strftime('%Y%m%d')}.json",
                    use_container_width=True
                )
        
        with col3:
            if st.button("🔄 Sync Registry", use_container_width=True):
                with st.spinner("Syncing with remote registry..."):
                    time.sleep(2)
                st.success("Registry synchronized!")

elif page_mode == "📈 Performance":
    
    st.markdown("### 📈 Model Performance Analytics")
    
    # Model selector
    deployed_models = [m for m in st.session_state.models if m["status"] in ["deployed", "staging"]]
    
    if deployed_models:
        selected_model_name = st.selectbox(
            "Select Model for Analysis",
            [m["name"] for m in deployed_models]
        )
        
        selected_model = next(m for m in deployed_models if m["name"] == selected_model_name)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Accuracy", f"{selected_model['accuracy']:.1%}", delta="1.2%")
        
        with col2:
            st.metric("Avg Latency", f"{selected_model['latency']:.0f}ms", delta="-5ms", delta_color="inverse")
        
        with col3:
            st.metric("Throughput", f"{selected_model['throughput']:.0f}/hr", delta="8%")
        
        with col4:
            error_rate = (1 - selected_model['accuracy']) * 100
            st.metric("Error Rate", f"{error_rate:.1f}%", delta="-0.5%", delta_color="inverse")
        
        # Performance charts
        tab1, tab2, tab3 = st.tabs(["📊 Accuracy Trends", "⚡ Latency Analysis", "🔄 Throughput Metrics"])
        
        with tab1:
            # Generate sample accuracy data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            accuracy_data = pd.DataFrame({
                'date': dates,
                'accuracy': selected_model['accuracy'] + 0.05 * np.sin(np.arange(30) / 5) + 0.02 * np.random.randn(30),
                'validation_accuracy': selected_model['accuracy'] + 0.03 * np.sin(np.arange(30) / 5) + 0.015 * np.random.randn(30)
            })
            
            fig_accuracy = go.Figure()
            
            fig_accuracy.add_trace(go.Scatter(
                x=accuracy_data['date'],
                y=accuracy_data['accuracy'],
                mode='lines+markers',
                name='Training Accuracy',
                line=dict(color='#1f77b4')
            ))
            
            fig_accuracy.add_trace(go.Scatter(
                x=accuracy_data['date'],
                y=accuracy_data['validation_accuracy'],
                mode='lines+markers',
                name='Validation Accuracy',
                line=dict(color='#ff7f0e')
            ))
            
            fig_accuracy.update_layout(
                title=f"Accuracy Trends - {selected_model['name']}",
                xaxis_title="Date",
                yaxis_title="Accuracy",
                height=400
            )
            
            st.plotly_chart(fig_accuracy, use_container_width=True)
        
        with tab2:
            # Latency distribution
            latency_data = np.random.gamma(2, selected_model['latency']/2, 1000)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_latency_hist = px.histogram(
                    x=latency_data,
                    nbins=50,
                    title="Latency Distribution",
                    labels={'x': 'Latency (ms)', 'y': 'Frequency'}
                )
                fig_latency_hist.update_layout(height=350)
                st.plotly_chart(fig_latency_hist, use_container_width=True)
            
            with col2:
                # Latency percentiles
                percentiles = [50, 75, 90, 95, 99]
                percentile_values = [np.percentile(latency_data, p) for p in percentiles]
                
                fig_percentiles = px.bar(
                    x=[f"P{p}" for p in percentiles],
                    y=percentile_values,
                    title="Latency Percentiles",
                    labels={'x': 'Percentile', 'y': 'Latency (ms)'}
                )
                fig_percentiles.update_layout(height=350)
                st.plotly_chart(fig_percentiles, use_container_width=True)
        
        with tab3:
            # Throughput over time
            hours = pd.date_range(end=datetime.now(), periods=24, freq='H')
            throughput_data = pd.DataFrame({
                'hour': hours,
                'requests': selected_model['throughput'] + 200 * np.sin(np.arange(24) / 4) + 100 * np.random.randn(24),
                'successful_requests': lambda x: x['requests'] * (0.95 + 0.03 * np.random.randn(24))
            })
            throughput_data['successful_requests'] = throughput_data['requests'] * (0.95 + 0.03 * np.random.randn(24))
            
            fig_throughput = go.Figure()
            
            fig_throughput.add_trace(go.Scatter(
                x=throughput_data['hour'],
                y=throughput_data['requests'],
                mode='lines+markers',
                name='Total Requests',
                line=dict(color='#1f77b4')
            ))
            
            fig_throughput.add_trace(go.Scatter(
                x=throughput_data['hour'],
                y=throughput_data['successful_requests'],
                mode='lines+markers',
                name='Successful Requests',
                line=dict(color='#2ca02c')
            ))
            
            fig_throughput.update_layout(
                title=f"Throughput Analysis - {selected_model['name']}",
                xaxis_title="Time",
                yaxis_title="Requests per Hour",
                height=400
            )
            
            st.plotly_chart(fig_throughput, use_container_width=True)
    
    else:
        st.info("No deployed models available for performance analysis.")

elif page_mode == "🔧 Configuration":
    
    st.markdown("### 🔧 Model Configuration")
    
    if filtered_models:
        config_model_name = st.selectbox(
            "Select Model to Configure",
            [m["name"] for m in filtered_models]
        )
        
        config_model = next(m for m in filtered_models if m["name"] == config_model_name)
        
        tab1, tab2, tab3 = st.tabs(["⚙️ Runtime Config", "🏗️ Infrastructure", "🔐 Security"])
        
        with tab1:
            st.markdown("#### Runtime Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Model Parameters")
                
                batch_size = st.slider("Batch Size", 1, 128, 32)
                max_sequence_length = st.slider("Max Sequence Length", 128, 2048, 512)
                temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
                top_k = st.slider("Top-K", 1, 100, 50)
                top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05)
            
            with col2:
                st.markdown("##### Performance Settings")
                
                enable_caching = st.checkbox("Enable Response Caching", value=True)
                cache_ttl = st.slider("Cache TTL (minutes)", 1, 60, 15)
                enable_batching = st.checkbox("Enable Request Batching", value=True)
                batch_timeout = st.slider("Batch Timeout (ms)", 10, 1000, 100)
                
                st.markdown("##### Logging & Monitoring")
                
                log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
                enable_metrics = st.checkbox("Enable Detailed Metrics", value=True)
                sample_rate = st.slider("Metrics Sample Rate", 0.01, 1.0, 0.1, 0.01)
        
        with tab2:
            st.markdown("#### Infrastructure Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Resource Allocation")
                
                cpu_request = st.number_input("CPU Request", 0.1, 8.0, config_model["resource_usage"]["cpu"] / 100, 0.1)
                cpu_limit = st.number_input("CPU Limit", 0.1, 16.0, cpu_request * 2, 0.1)
                memory_request = st.number_input("Memory Request (GB)", 0.5, 32.0, config_model["resource_usage"]["memory"], 0.1)
                memory_limit = st.number_input("Memory Limit (GB)", 0.5, 64.0, memory_request * 2, 0.1)
            
            with col2:
                st.markdown("##### Auto-scaling")
                
                enable_hpa = st.checkbox("Enable Horizontal Pod Autoscaler", value=True)
                min_replicas = st.slider("Min Replicas", 1, 10, 2)
                max_replicas = st.slider("Max Replicas", 1, 50, 10)
                target_cpu_utilization = st.slider("Target CPU Utilization (%)", 30, 90, 70)
                target_memory_utilization = st.slider("Target Memory Utilization (%)", 30, 90, 80)
                
                st.markdown("##### Health Checks")
                
                readiness_path = st.text_input("Readiness Probe Path", "/ready")
                liveness_path = st.text_input("Liveness Probe Path", "/health")
                probe_timeout = st.slider("Probe Timeout (s)", 1, 30, 5)
        
        with tab3:
            st.markdown("#### Security Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Authentication")
                
                auth_required = st.checkbox("Require Authentication", value=True)
                auth_method = st.selectbox("Authentication Method", ["API Key", "JWT Token", "OAuth2", "mTLS"])
                
                if auth_method == "API Key":
                    api_key_header = st.text_input("API Key Header", "X-API-Key")
                elif auth_method == "JWT Token":
                    jwt_issuer = st.text_input("JWT Issuer", "https://auth.secure-aiml.com")
                    jwt_audience = st.text_input("JWT Audience", "api.secure-aiml.com")
                
                rate_limiting = st.checkbox("Enable Rate Limiting", value=True)
                if rate_limiting:
                    rate_limit_rpm = st.slider("Requests per Minute", 10, 10000, 1000)
                    rate_limit_burst = st.slider("Burst Allowance", 1, 100, 10)
            
            with col2:
                st.markdown("##### Data Protection")
                
                encrypt_requests = st.checkbox("Encrypt Request Data", value=True)
                encrypt_responses = st.checkbox("Encrypt Response Data", value=True)
                data_retention_days = st.slider("Data Retention (days)", 1, 365, 30)
                
                pii_detection = st.checkbox("Enable PII Detection", value=True)
                if pii_detection:
                    pii_action = st.selectbox("PII Action", ["Block", "Mask", "Log Only"])
                
                st.markdown("##### Compliance")
                
                gdpr_compliance = st.checkbox("GDPR Compliance Mode", value=True)
                audit_logging = st.checkbox("Enable Audit Logging", value=True)
                data_locality = st.selectbox("Data Locality", ["EU", "US", "Global"])
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            with st.spinner("Saving configuration..."):
                time.sleep(2)
            st.success("✅ Configuration saved successfully!")
            st.info("Changes will be applied during the next deployment.")
    
    else:
        st.info("No models available for configuration.")

elif page_mode == "📋 Logs":
    
    st.markdown("### 📋 Model Logs & Monitoring")
    
    # Log filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        log_model = st.selectbox(
            "Select Model",
            ["All Models"] + [m["name"] for m in st.session_state.models]
        )
    
    with col2:
        log_level = st.selectbox(
            "Log Level",
            ["All", "ERROR", "WARNING", "INFO", "DEBUG"]
        )
    
    with col3:
        time_range = st.selectbox(
            "Time Range",
            ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"]
        )
    
    with col4:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    
    # Generate sample logs
    log_entries = []
    log_levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
    models = [m["name"] for m in st.session_state.models]
    
    for i in range(50):
        timestamp = datetime.now() - timedelta(minutes=i*5)
        level = np.random.choice(log_levels, p=[0.6, 0.2, 0.1, 0.1])
        model = np.random.choice(models)
        
        messages = {
            "INFO": [
                f"Model {model} processed request successfully",
                f"Health check passed for {model}",
                f"Cache hit for {model} request",
                f"Model {model} auto-scaled up"
            ],
            "WARNING": [
                f"High latency detected for {model}",
                f"Memory usage approaching limit for {model}",
                f"Rate limit threshold reached for {model}"
            ],
            "ERROR": [
                f"Request failed for {model}: Internal server error",
                f"Model {model} health check failed",
                f"Authentication failed for {model} request"
            ],
            "DEBUG": [
                f"Processing request for {model}",
                f"Loading model weights for {model}",
                f"Garbage collection triggered for {model}"
            ]
        }
        
        message = np.random.choice(messages[level])
        
        log_entries.append({
            "timestamp": timestamp,
            "level": level,
            "model": model,
            "message": message,
            "request_id": f"req_{i:06d}"
        })
    
    # Filter logs
    filtered_logs = log_entries
    if log_model != "All Models":
        filtered_logs = [log for log in filtered_logs if log["model"] == log_model]
    if log_level != "All":
        filtered_logs = [log for log in filtered_logs if log["level"] == log_level]
    
    # Display logs
    st.markdown("#### Recent Log Entries")
    
    for log in filtered_logs[:20]:
        timestamp_str = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        if log["level"] == "ERROR":
            st.error(f"🔴 **{timestamp_str}** | {log['model']} | {log['message']}")
        elif log["level"] == "WARNING":
            st.warning(f"🟡 **{timestamp_str}** | {log['model']} | {log['message']}")
        elif log["level"] == "INFO":
            st.info(f"🔵 **{timestamp_str}** | {log['model']} | {log['message']}")
        else:  # DEBUG
            st.text(f"⚪ **{timestamp_str}** | {log['model']} | {log['message']}")
    
    # Log analytics
    st.markdown("---")
    st.markdown("#### Log Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Log level distribution
        level_counts = pd.Series([log["level"] for log in filtered_logs]).value_counts()
        
        fig_levels = px.bar(
            x=level_counts.index,
            y=level_counts.values,
            title="Log Level Distribution",
            color=level_counts.values,
            color_continuous_scale='Reds'
        )
        fig_levels.update_layout(height=300)
        st.plotly_chart(fig_levels, use_container_width=True)
    
    with col2:
        # Logs over time
        log_df = pd.DataFrame(filtered_logs)
        log_df['hour'] = log_df['timestamp'].dt.floor('H')
        hourly_counts = log_df.groupby(['hour', 'level']).size().reset_index(name='count')
        
        fig_timeline = px.line(
            hourly_counts,
            x='hour',
            y='count',
            color='level',
            title="Log Volume Over Time"
        )
        fig_timeline.update_layout(height=300)
        st.plotly_chart(fig_timeline, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666;">
    <p>⚙️ Model Management System | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Managing {len(st.session_state.models)} models across {len(set(m['status'] for m in st.session_state.models))} environments</p>
</div>
""", unsafe_allow_html=True)