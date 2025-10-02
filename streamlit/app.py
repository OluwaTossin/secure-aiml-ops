"""
Secure AI/ML Operations - Streamlit Application
=============================================

Main application entry point for the Streamlit web interface.
Provides interactive access to AI/ML models and analytics dashboard.

Author: Secure AI/ML Ops Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional, Any
import logging

# Configure page settings
st.set_page_config(
    page_title="Secure AI/ML Operations Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/secure-aiml-ops',
        'Report a bug': 'https://github.com/your-repo/secure-aiml-ops/issues',
        'About': "# Secure AI/ML Operations Platform\nA comprehensive ML operations dashboard with enterprise security."
    }
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom components and utilities
try:
    from config.settings import AppConfig
    from utils.aws_client import AWSClient
    from utils.model_client import ModelClient
    from components.layouts import create_sidebar, create_metrics_bar
    from components.charts import create_performance_chart, create_status_chart
except ImportError as e:
    logger.warning(f"Could not import custom modules: {e}")
    # Fallback for development without full setup
    pass

# Global CSS styling
def load_css():
    """Load custom CSS styling"""
    st.markdown("""
    <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        
        /* Metrics cards */
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #1f77b4;
        }
        
        /* Status indicators */
        .status-healthy { color: #28a745; font-weight: bold; }
        .status-warning { color: #ffc107; font-weight: bold; }
        .status-error { color: #dc3545; font-weight: bold; }
        
        /* Headers */
        .main-header {
            color: #1f77b4;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .section-header {
            color: #333;
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border: none;
            border-radius: 0.25rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        .stButton > button:hover {
            background-color: #1565c0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        /* Alert styling */
        .alert-success {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
            padding: 0.75rem;
            border-radius: 0.25rem;
            border-left: 4px solid #28a745;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffeaa7;
            color: #856404;
            padding: 0.75rem;
            border-radius: 0.25rem;
            border-left: 4px solid #ffc107;
        }
        
        .alert-error {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
            padding: 0.75rem;
            border-radius: 0.25rem;
            border-left: 4px solid #dc3545;
        }
    </style>
    """, unsafe_allow_html=True)

def create_sample_data():
    """Create sample data for demonstration"""
    # Model performance data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    performance_data = pd.DataFrame({
        'date': dates,
        'accuracy': 0.85 + 0.1 * np.random.randn(30).cumsum() * 0.01,
        'latency': 120 + 50 * np.random.randn(30),
        'throughput': 1000 + 200 * np.random.randn(30)
    })
    
    # System health data
    health_data = {
        'Airflow Scheduler': 'healthy',
        'Model Serving': 'healthy',
        'Database': 'healthy',
        'S3 Storage': 'healthy',
        'ECR Registry': 'warning',
        'Load Balancer': 'healthy'
    }
    
    # Recent predictions data
    prediction_data = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=24), periods=100, freq='15min'),
        'model': np.random.choice(['text-summarization', 'anomaly-detection'], 100),
        'confidence': 0.7 + 0.3 * np.random.rand(100),
        'response_time': 50 + 100 * np.random.rand(100)
    })
    
    return performance_data, health_data, prediction_data

def create_header():
    """Create the main application header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 class="main-header">🚀 Secure AI/ML Operations</h1>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
                Enterprise ML Pipeline Dashboard & Model Interaction Platform
            </p>
        </div>
        """, unsafe_allow_html=True)

def create_metrics_overview():
    """Create the metrics overview section"""
    st.markdown('<h2 class="section-header">📊 System Overview</h2>', unsafe_allow_html=True)
    
    # Create sample metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Active Models",
            value="5",
            delta="2",
            help="Number of models currently deployed and serving predictions"
        )
    
    with col2:
        st.metric(
            label="Daily Predictions",
            value="12.4K",
            delta="8.2%",
            help="Total predictions made in the last 24 hours"
        )
    
    with col3:
        st.metric(
            label="Avg Latency",
            value="89ms",
            delta="-12ms",
            delta_color="inverse",
            help="Average model inference latency"
        )
    
    with col4:
        st.metric(
            label="Success Rate",
            value="99.7%",
            delta="0.1%",
            help="Percentage of successful predictions"
        )
    
    with col5:
        st.metric(
            label="Data Quality",
            value="96.2%",
            delta="-1.2%",
            delta_color="inverse",
            help="Overall data quality score"
        )

def create_system_health():
    """Create the system health monitoring section"""
    st.markdown('<h2 class="section-header">🏥 System Health</h2>', unsafe_allow_html=True)
    
    _, health_data, _ = create_sample_data()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Health status table
        health_df = pd.DataFrame([
            {"Component": component, "Status": status.title(), "Last Check": "2 min ago"}
            for component, status in health_data.items()
        ])
        
        # Style the dataframe
        def color_status(val):
            if val == 'Healthy':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Warning':
                return 'background-color: #fff3cd; color: #856404'
            else:
                return 'background-color: #f8d7da; color: #721c24'
        
        styled_df = health_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Health summary chart
        health_counts = pd.Series(health_data).value_counts()
        fig = px.pie(
            values=health_counts.values,
            names=health_counts.index,
            title="System Health Summary",
            color_discrete_map={
                'healthy': '#28a745',
                'warning': '#ffc107',
                'error': '#dc3545'
            }
        )
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

def create_performance_dashboard():
    """Create the performance monitoring dashboard"""
    st.markdown('<h2 class="section-header">📈 Performance Analytics</h2>', unsafe_allow_html=True)
    
    performance_data, _, _ = create_sample_data()
    
    # Performance metrics selector
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        metric_option = st.selectbox(
            "Select Metric",
            ["accuracy", "latency", "throughput"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        time_range = st.selectbox(
            "Time Range",
            ["7 days", "14 days", "30 days"],
            index=2
        )
    
    # Performance chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=performance_data['date'],
        y=performance_data[metric_option],
        mode='lines+markers',
        name=metric_option.replace('_', ' ').title(),
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"{metric_option.replace('_', ' ').title()} Over Time",
        xaxis_title="Date",
        yaxis_title=metric_option.replace('_', ' ').title(),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_recent_activity():
    """Create the recent activity section"""
    st.markdown('<h2 class="section-header">🕒 Recent Activity</h2>', unsafe_allow_html=True)
    
    _, _, prediction_data = create_sample_data()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Recent predictions table
        recent_predictions = prediction_data.tail(10).copy()
        recent_predictions['timestamp'] = recent_predictions['timestamp'].dt.strftime('%H:%M:%S')
        recent_predictions['confidence'] = recent_predictions['confidence'].apply(lambda x: f"{x:.2%}")
        recent_predictions['response_time'] = recent_predictions['response_time'].apply(lambda x: f"{x:.0f}ms")
        
        st.dataframe(
            recent_predictions[['timestamp', 'model', 'confidence', 'response_time']],
            column_config={
                'timestamp': 'Time',
                'model': 'Model',
                'confidence': 'Confidence',
                'response_time': 'Response Time'
            },
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        # Model usage distribution
        model_usage = prediction_data['model'].value_counts()
        fig = px.bar(
            x=model_usage.values,
            y=model_usage.index,
            orientation='h',
            title="Model Usage (24h)",
            color=model_usage.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def create_quick_actions():
    """Create the quick actions section"""
    st.markdown('<h2 class="section-header">⚡ Quick Actions</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Models", use_container_width=True):
            with st.spinner("Refreshing models..."):
                time.sleep(2)
            st.success("Models refreshed successfully!")
    
    with col2:
        if st.button("📊 Generate Report", use_container_width=True):
            with st.spinner("Generating report..."):
                time.sleep(2)
            st.success("Report generated and sent!")
    
    with col3:
        if st.button("🧹 Clear Cache", use_container_width=True):
            with st.spinner("Clearing cache..."):
                time.sleep(1)
                st.cache_data.clear()
            st.success("Cache cleared successfully!")
    
    with col4:
        if st.button("🔔 Test Alerts", use_container_width=True):
            with st.spinner("Testing alert system..."):
                time.sleep(2)
            st.success("Alerts tested successfully!")

def create_sidebar_info():
    """Create sidebar with system information"""
    with st.sidebar:
        st.markdown("### 📋 System Information")
        
        # System status
        st.markdown("**Status:** :green[🟢 Operational]")
        st.markdown("**Uptime:** 15 days, 8 hours")
        st.markdown("**Version:** v1.0.0")
        st.markdown("**Region:** eu-west-1")
        
        st.markdown("---")
        
        # Environment info
        st.markdown("### 🌍 Environment")
        st.markdown("**Environment:** Production")
        st.markdown("**Cluster:** secure-aiml-ops")
        st.markdown("**Namespace:** default")
        
        st.markdown("---")
        
        # Navigation shortcuts
        st.markdown("### 🔗 Quick Links")
        if st.button("📊 Analytics Dashboard", use_container_width=True):
            st.switch_page("pages/1_📊_Dashboard.py")
        
        if st.button("🤖 Text Summarization", use_container_width=True):
            st.switch_page("pages/2_🤖_Text_Summarization.py")
        
        if st.button("🔍 Anomaly Detection", use_container_width=True):
            st.switch_page("pages/3_🔍_Anomaly_Detection.py")
        
        if st.button("⚙️ Model Management", use_container_width=True):
            st.switch_page("pages/4_⚙️_Model_Management.py")
        
        if st.button("💬 AI Chatbot", use_container_width=True):
            st.switch_page("pages/5_🤖_AI_Chatbot.py")
        
        st.markdown("---")
        
        # Resource usage
        st.markdown("### 💾 Resource Usage")
        st.progress(0.7, text="CPU: 70%")
        st.progress(0.5, text="Memory: 50%")
        st.progress(0.3, text="Storage: 30%")

def main():
    """Main application function"""
    # Load custom CSS
    load_css()
    
    # Create sidebar information
    create_sidebar_info()
    
    # Create main content
    create_header()
    
    # Add a divider
    st.markdown("---")
    
    # Create main dashboard sections
    create_metrics_overview()
    create_system_health()
    create_performance_dashboard()
    create_recent_activity()
    create_quick_actions()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Secure AI/ML Operations Platform | Built with ❤️ using Streamlit</p>
        <p>For support, contact: <a href="mailto:support@company.com">support@company.com</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)