"""
Analytics Dashboard Page
======================

Advanced analytics dashboard with detailed metrics, charts, and system monitoring.
Provides comprehensive view of ML model performance and system health.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="Analytics Dashboard - Secure AI/ML Ops",
    page_icon="📊",
    layout="wide"
)

# Page header
st.markdown("""
# 📊 Analytics Dashboard

Comprehensive monitoring and analytics for your AI/ML operations platform.
""")

# Sidebar controls
with st.sidebar:
    st.markdown("### 📊 Dashboard Controls")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
        index=2
    )
    
    # Model filter
    selected_models = st.multiselect(
        "Filter Models",
        ["text-summarization", "anomaly-detection", "sentiment-analysis", "recommendation-engine"],
        default=["text-summarization", "anomaly-detection"]
    )
    
    # Metric selector
    primary_metric = st.selectbox(
        "Primary Metric",
        ["Accuracy", "Latency", "Throughput", "Error Rate"],
        index=0
    )
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# Create sample data
def generate_detailed_analytics_data():
    """Generate comprehensive analytics data"""
    
    # Time-based data
    if "Last 1 Hour" in time_range:
        periods, freq = 60, "1min"
    elif "Last 6 Hours" in time_range:
        periods, freq = 72, "5min"
    elif "Last 24 Hours" in time_range:
        periods, freq = 144, "10min"
    elif "Last 7 Days" in time_range:
        periods, freq = 168, "1H"
    else:  # Last 30 Days
        periods, freq = 720, "1H"
    
    dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
    
    # Model performance metrics
    performance_data = pd.DataFrame({
        'timestamp': dates,
        'accuracy': 0.85 + 0.1 * np.random.randn(periods).cumsum() * 0.01,
        'latency': 100 + 50 * np.random.randn(periods),
        'throughput': 1000 + 200 * np.random.randn(periods),
        'error_rate': 0.02 + 0.01 * np.abs(np.random.randn(periods)),
        'cpu_usage': 0.6 + 0.2 * np.random.randn(periods),
        'memory_usage': 0.5 + 0.15 * np.random.randn(periods),
        'requests_per_second': 50 + 20 * np.random.randn(periods)
    })
    
    # Model-specific data
    model_data = []
    for model in selected_models:
        for i, timestamp in enumerate(dates):
            model_data.append({
                'timestamp': timestamp,
                'model': model,
                'accuracy': 0.8 + 0.15 * np.random.rand() + 0.05 * np.sin(i/10),
                'latency': 80 + 40 * np.random.rand() + (10 if model == 'text-summarization' else 0),
                'requests': np.random.poisson(30),
                'errors': np.random.poisson(1)
            })
    
    model_df = pd.DataFrame(model_data)
    
    # Geographical data
    geo_data = pd.DataFrame({
        'region': ['US-East', 'US-West', 'EU-West', 'Asia-Pacific', 'South-America'],
        'requests': [15000, 12000, 18000, 8000, 3000],
        'avg_latency': [85, 95, 75, 120, 140],
        'success_rate': [99.8, 99.5, 99.9, 99.2, 98.8]
    })
    
    return performance_data, model_df, geo_data

# Generate data
performance_data, model_df, geo_data = generate_detailed_analytics_data()

# Main metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_accuracy = performance_data['accuracy'].mean()
    st.metric(
        "Avg Accuracy",
        f"{avg_accuracy:.2%}",
        delta=f"{(avg_accuracy - 0.85):.2%}"
    )

with col2:
    avg_latency = performance_data['latency'].mean()
    st.metric(
        "Avg Latency",
        f"{avg_latency:.0f}ms",
        delta=f"{avg_latency - 100:.0f}ms",
        delta_color="inverse"
    )

with col3:
    total_requests = performance_data['requests_per_second'].sum() * 3600
    st.metric(
        "Total Requests",
        f"{total_requests/1000:.1f}K",
        delta="12.5%"
    )

with col4:
    avg_error_rate = performance_data['error_rate'].mean()
    st.metric(
        "Error Rate",
        f"{avg_error_rate:.2%}",
        delta=f"{(avg_error_rate - 0.02):.2%}",
        delta_color="inverse"
    )

with col5:
    uptime = 99.7
    st.metric(
        "Uptime",
        f"{uptime:.1f}%",
        delta="0.2%"
    )

st.markdown("---")

# Main dashboard content
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🔍 Model Comparison", "🌍 Geographic", "🖥️ Infrastructure"])

with tab1:
    st.markdown("### Performance Trends")
    
    # Primary metric chart
    fig_main = go.Figure()
    
    metric_map = {
        "Accuracy": "accuracy",
        "Latency": "latency", 
        "Throughput": "throughput",
        "Error Rate": "error_rate"
    }
    
    metric_col = metric_map[primary_metric]
    
    fig_main.add_trace(go.Scatter(
        x=performance_data['timestamp'],
        y=performance_data[metric_col],
        mode='lines+markers',
        name=primary_metric,
        line=dict(color='#1f77b4', width=3),
        fill='tonexty' if primary_metric != "Error Rate" else None
    ))
    
    fig_main.update_layout(
        title=f"{primary_metric} Over Time",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Secondary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # System resource usage
        fig_resources = make_subplots(
            rows=2, cols=1,
            subplot_titles=['CPU Usage', 'Memory Usage'],
            vertical_spacing=0.1
        )
        
        fig_resources.add_trace(
            go.Scatter(x=performance_data['timestamp'], 
                      y=performance_data['cpu_usage']*100,
                      name='CPU %', line=dict(color='#ff7f0e')),
            row=1, col=1
        )
        
        fig_resources.add_trace(
            go.Scatter(x=performance_data['timestamp'], 
                      y=performance_data['memory_usage']*100,
                      name='Memory %', line=dict(color='#2ca02c')),
            row=2, col=1
        )
        
        fig_resources.update_layout(height=400, title="System Resources")
        st.plotly_chart(fig_resources, use_container_width=True)
    
    with col2:
        # Request distribution
        hourly_requests = performance_data.groupby(
            performance_data['timestamp'].dt.hour
        )['requests_per_second'].mean().reset_index()
        
        fig_requests = px.bar(
            hourly_requests,
            x='timestamp',
            y='requests_per_second',
            title='Average Requests per Hour',
            color='requests_per_second',
            color_continuous_scale='Blues'
        )
        fig_requests.update_layout(height=400)
        st.plotly_chart(fig_requests, use_container_width=True)

with tab2:
    st.markdown("### Model Performance Comparison")
    
    if selected_models:
        # Model accuracy comparison
        col1, col2 = st.columns(2)
        
        with col1:
            avg_model_metrics = model_df.groupby('model').agg({
                'accuracy': 'mean',
                'latency': 'mean',
                'requests': 'sum',
                'errors': 'sum'
            }).reset_index()
            
            fig_model_acc = px.bar(
                avg_model_metrics,
                x='model',
                y='accuracy',
                title='Average Model Accuracy',
                color='accuracy',
                color_continuous_scale='Greens'
            )
            fig_model_acc.update_layout(height=400)
            st.plotly_chart(fig_model_acc, use_container_width=True)
        
        with col2:
            fig_model_lat = px.bar(
                avg_model_metrics,
                x='model',
                y='latency',
                title='Average Model Latency',
                color='latency',
                color_continuous_scale='Reds'
            )
            fig_model_lat.update_layout(height=400)
            st.plotly_chart(fig_model_lat, use_container_width=True)
        
        # Model timeline comparison
        fig_timeline = px.line(
            model_df,
            x='timestamp',
            y='accuracy',
            color='model',
            title='Model Accuracy Over Time'
        )
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Model statistics table
        st.markdown("### Model Statistics")
        
        # Calculate error rates
        avg_model_metrics['error_rate'] = (
            avg_model_metrics['errors'] / avg_model_metrics['requests'] * 100
        )
        
        st.dataframe(
            avg_model_metrics.round(3),
            column_config={
                'model': 'Model',
                'accuracy': 'Accuracy',
                'latency': 'Avg Latency (ms)',
                'requests': 'Total Requests',
                'errors': 'Total Errors',
                'error_rate': 'Error Rate (%)'
            },
            use_container_width=True,
            hide_index=True
        )
    
    else:
        st.warning("Please select at least one model to view comparison.")

with tab3:
    st.markdown("### Geographic Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Geographic requests map (simulated)
        fig_geo = px.bar(
            geo_data,
            x='region',
            y='requests',
            title='Requests by Region',
            color='requests',
            color_continuous_scale='Blues'
        )
        fig_geo.update_layout(height=400)
        st.plotly_chart(fig_geo, use_container_width=True)
    
    with col2:
        # Regional performance metrics
        st.markdown("#### Regional Metrics")
        
        for idx, row in geo_data.iterrows():
            st.markdown(f"**{row['region']}**")
            st.metric(
                "Requests",
                f"{row['requests']:,}",
                help=f"Total requests from {row['region']}"
            )
            st.metric(
                "Latency",
                f"{row['avg_latency']:.0f}ms",
                help=f"Average latency for {row['region']}"
            )
            st.metric(
                "Success Rate",
                f"{row['success_rate']:.1f}%",
                help=f"Success rate for {row['region']}"
            )
            st.markdown("---")

with tab4:
    st.markdown("### Infrastructure Monitoring")
    
    # Infrastructure metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Container Health")
        containers = [
            {"name": "streamlit-app", "status": "healthy", "cpu": 45, "memory": 60},
            {"name": "model-server", "status": "healthy", "cpu": 70, "memory": 80},
            {"name": "redis-cache", "status": "healthy", "cpu": 20, "memory": 30},
            {"name": "postgres-db", "status": "warning", "cpu": 60, "memory": 85}
        ]
        
        for container in containers:
            status_color = "🟢" if container["status"] == "healthy" else "🟡"
            st.markdown(f"{status_color} **{container['name']}**")
            st.progress(container["cpu"]/100, text=f"CPU: {container['cpu']}%")
            st.progress(container["memory"]/100, text=f"Memory: {container['memory']}%")
            st.markdown("---")
    
    with col2:
        st.markdown("#### Network Metrics")
        
        # Network usage chart
        network_data = pd.DataFrame({
            'time': pd.date_range(end=datetime.now(), periods=24, freq='1H'),
            'inbound': np.random.normal(100, 20, 24),
            'outbound': np.random.normal(80, 15, 24)
        })
        
        fig_network = go.Figure()
        fig_network.add_trace(go.Scatter(
            x=network_data['time'],
            y=network_data['inbound'],
            name='Inbound',
            line=dict(color='#1f77b4')
        ))
        fig_network.add_trace(go.Scatter(
            x=network_data['time'],
            y=network_data['outbound'],
            name='Outbound',
            line=dict(color='#ff7f0e')
        ))
        fig_network.update_layout(
            title="Network Traffic (MB/s)",
            height=300,
            yaxis_title="MB/s"
        )
        st.plotly_chart(fig_network, use_container_width=True)
    
    with col3:
        st.markdown("#### Storage Metrics")
        
        storage_data = [
            {"volume": "/data", "used": 65, "total": "100GB"},
            {"volume": "/models", "used": 40, "total": "50GB"},
            {"volume": "/logs", "used": 25, "total": "20GB"},
            {"volume": "/cache", "used": 80, "total": "30GB"}
        ]
        
        for storage in storage_data:
            st.markdown(f"**{storage['volume']}** ({storage['total']})")
            st.progress(storage["used"]/100, text=f"Used: {storage['used']}%")
            st.markdown("---")
        
        # Storage trend
        storage_trend = pd.DataFrame({
            'date': pd.date_range(end=datetime.now(), periods=7, freq='1D'),
            'usage': [55, 58, 62, 65, 63, 67, 70]
        })
        
        fig_storage = px.line(
            storage_trend,
            x='date',
            y='usage',
            title='Storage Usage Trend (7 days)',
            markers=True
        )
        fig_storage.update_layout(height=250, yaxis_title="Usage %")
        st.plotly_chart(fig_storage, use_container_width=True)

# Real-time alerts section
st.markdown("---")
st.markdown("### 🚨 Recent Alerts")

alerts = [
    {"time": "2 min ago", "level": "warning", "message": "High memory usage detected on postgres-db container (85%)"},
    {"time": "15 min ago", "level": "info", "message": "Model deployment completed successfully: text-summarization-v2.1"},
    {"time": "1 hour ago", "level": "error", "message": "Temporary spike in error rate for anomaly-detection model"},
    {"time": "3 hours ago", "level": "info", "message": "Scheduled maintenance completed for ECR registry"}
]

for alert in alerts:
    if alert["level"] == "error":
        st.error(f"🔴 **{alert['time']}**: {alert['message']}")
    elif alert["level"] == "warning":
        st.warning(f"🟡 **{alert['time']}**: {alert['message']}")
    else:
        st.info(f"🔵 **{alert['time']}**: {alert['message']}")

# Footer with last update time
st.markdown("---")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")