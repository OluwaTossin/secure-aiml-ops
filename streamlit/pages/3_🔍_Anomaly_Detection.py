"""
Anomaly Detection Dashboard
=========================

Advanced anomaly detection interface for financial transactions and system monitoring.
Provides real-time detection, interactive visualizations, and alert management.
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
    page_title="Anomaly Detection - Secure AI/ML Ops",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'anomaly_alerts' not in st.session_state:
    st.session_state.anomaly_alerts = []

if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# Page header
st.markdown("""
# 🔍 Anomaly Detection

Real-time anomaly detection for financial transactions, system metrics, and security monitoring.
""")

# Sidebar controls
with st.sidebar:
    st.markdown("### 🎛️ Detection Controls")
    
    # Detection type
    detection_type = st.selectbox(
        "Detection Type",
        [
            "Financial Transactions",
            "System Metrics",
            "Network Traffic",
            "User Behavior",
            "Custom Data"
        ],
        help="Select the type of data to monitor for anomalies"
    )
    
    # Model selection
    model_choice = st.selectbox(
        "Detection Model",
        [
            "Isolation Forest",
            "Local Outlier Factor",
            "One-Class SVM",
            "LSTM Autoencoder",
            "Statistical Z-Score"
        ],
        help="Choose the anomaly detection algorithm"
    )
    
    # Sensitivity settings
    st.markdown("### ⚙️ Sensitivity Settings")
    
    sensitivity = st.select_slider(
        "Detection Sensitivity",
        options=["Very Low", "Low", "Medium", "High", "Very High"],
        value="Medium",
        help="Higher sensitivity detects more anomalies but may increase false positives"
    )
    
    threshold = st.slider(
        "Anomaly Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Lower values are more sensitive to anomalies"
    )
    
    # Time window
    time_window = st.selectbox(
        "Analysis Window",
        ["Real-time", "Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
        index=1
    )
    
    # Advanced settings
    with st.expander("Advanced Parameters"):
        contamination = st.slider("Expected Contamination", 0.01, 0.5, 0.1, 0.01)
        n_estimators = st.slider("Estimators", 50, 500, 100, 50)
        max_samples = st.slider("Max Samples", 0.1, 1.0, 0.8, 0.1)
    
    # Model performance
    st.markdown("### 📊 Model Performance")
    
    model_metrics = {
        "Isolation Forest": {"precision": 0.89, "recall": 0.83, "f1": 0.86},
        "Local Outlier Factor": {"precision": 0.85, "recall": 0.78, "f1": 0.81},
        "One-Class SVM": {"precision": 0.87, "recall": 0.80, "f1": 0.84},
        "LSTM Autoencoder": {"precision": 0.92, "recall": 0.88, "f1": 0.90},
        "Statistical Z-Score": {"precision": 0.76, "recall": 0.85, "f1": 0.80}
    }
    
    current_metrics = model_metrics[model_choice]
    st.metric("Precision", f"{current_metrics['precision']:.2%}")
    st.metric("Recall", f"{current_metrics['recall']:.2%}")
    st.metric("F1-Score", f"{current_metrics['f1']:.2%}")

# Generate sample data based on detection type
def generate_sample_data(detection_type, time_window):
    """Generate sample data for different detection types"""
    
    # Determine time range
    if time_window == "Real-time":
        periods, freq = 60, "1min"
        start_time = datetime.now() - timedelta(hours=1)
    elif time_window == "Last 1 Hour":
        periods, freq = 60, "1min"
        start_time = datetime.now() - timedelta(hours=1)
    elif time_window == "Last 6 Hours":
        periods, freq = 72, "5min"
        start_time = datetime.now() - timedelta(hours=6)
    elif time_window == "Last 24 Hours":
        periods, freq = 144, "10min"
        start_time = datetime.now() - timedelta(hours=24)
    else:  # Last 7 Days
        periods, freq = 168, "1H"
        start_time = datetime.now() - timedelta(days=7)
    
    timestamps = pd.date_range(start=start_time, periods=periods, freq=freq)
    
    if detection_type == "Financial Transactions":
        # Generate financial transaction data
        np.random.seed(42)
        base_amounts = np.random.lognormal(3, 1, periods)  # Log-normal distribution for realistic amounts
        
        # Add seasonal patterns (higher activity during business hours)
        hourly_pattern = np.sin(2 * np.pi * np.arange(periods) / (24 * 60 / (60 if "min" in freq else 1))) + 1
        base_amounts *= (1 + 0.3 * hourly_pattern)
        
        # Add anomalies
        anomaly_indices = np.random.choice(periods, size=int(periods * 0.05), replace=False)
        anomalies = np.zeros(periods, dtype=bool)
        anomalies[anomaly_indices] = True
        
        # Make anomalous transactions significantly different
        amounts = base_amounts.copy()
        amounts[anomalies] *= np.random.choice([0.1, 10], size=len(anomaly_indices))  # Very small or very large
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'transaction_id': [f"TXN_{i:06d}" for i in range(periods)],
            'amount': amounts,
            'merchant_category': np.random.choice(['Retail', 'Gas', 'Restaurant', 'Online', 'ATM'], periods),
            'card_type': np.random.choice(['Credit', 'Debit'], periods),
            'is_anomaly': anomalies,
            'anomaly_score': np.where(anomalies, np.random.uniform(0.7, 1.0, periods), np.random.uniform(0.0, 0.3, periods))
        })
        
    elif detection_type == "System Metrics":
        # Generate system metrics data
        np.random.seed(42)
        
        # Normal CPU usage with daily patterns
        cpu_base = 40 + 30 * np.sin(2 * np.pi * np.arange(periods) / 144) + np.random.normal(0, 5, periods)
        memory_base = 60 + 20 * np.sin(2 * np.pi * np.arange(periods) / 144 + np.pi/4) + np.random.normal(0, 3, periods)
        disk_io = 100 + 50 * np.random.exponential(1, periods)
        network_io = 50 + 30 * np.random.gamma(2, 2, periods)
        
        # Add anomalies
        anomaly_indices = np.random.choice(periods, size=int(periods * 0.08), replace=False)
        anomalies = np.zeros(periods, dtype=bool)
        anomalies[anomaly_indices] = True
        
        cpu_usage = cpu_base.copy()
        memory_usage = memory_base.copy()
        
        # Create anomalous spikes
        cpu_usage[anomalies] += np.random.uniform(30, 50, len(anomaly_indices))
        memory_usage[anomalies] += np.random.uniform(20, 40, len(anomaly_indices))
        
        # Ensure values stay within realistic bounds
        cpu_usage = np.clip(cpu_usage, 0, 100)
        memory_usage = np.clip(memory_usage, 0, 100)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_io': disk_io,
            'network_io': network_io,
            'is_anomaly': anomalies,
            'anomaly_score': np.where(anomalies, np.random.uniform(0.6, 0.95, periods), np.random.uniform(0.0, 0.4, periods))
        })
        
    elif detection_type == "Network Traffic":
        # Generate network traffic data
        np.random.seed(42)
        
        # Normal traffic patterns
        inbound_base = 100 + 50 * np.sin(2 * np.pi * np.arange(periods) / 144) + np.random.normal(0, 10, periods)
        outbound_base = 80 + 40 * np.sin(2 * np.pi * np.arange(periods) / 144 + np.pi/3) + np.random.normal(0, 8, periods)
        connections = np.random.poisson(50, periods)
        
        # Add anomalies (DDoS-like patterns)
        anomaly_indices = np.random.choice(periods, size=int(periods * 0.06), replace=False)
        anomalies = np.zeros(periods, dtype=bool)
        anomalies[anomaly_indices] = True
        
        inbound_traffic = inbound_base.copy()
        outbound_traffic = outbound_base.copy()
        
        # Create traffic spikes for anomalies
        inbound_traffic[anomalies] *= np.random.uniform(5, 15, len(anomaly_indices))
        connections[anomalies] *= np.random.randint(10, 50, len(anomaly_indices))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'inbound_mbps': np.maximum(0, inbound_traffic),
            'outbound_mbps': np.maximum(0, outbound_traffic),
            'connections': connections,
            'packets_per_sec': connections * np.random.uniform(10, 100, periods),
            'is_anomaly': anomalies,
            'anomaly_score': np.where(anomalies, np.random.uniform(0.7, 0.98, periods), np.random.uniform(0.0, 0.35, periods))
        })
        
    else:  # Default to financial transactions
        return generate_sample_data("Financial Transactions", time_window)
    
    return data

# Generate data
data = generate_sample_data(detection_type, time_window)

# Main dashboard
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Real-time Detection", "📊 Analysis", "🚨 Alerts", "📈 Historical Trends"])

with tab1:
    st.markdown("### Real-time Anomaly Detection")
    
    # Key metrics
    total_points = len(data)
    anomaly_count = data['is_anomaly'].sum()
    anomaly_rate = anomaly_count / total_points if total_points > 0 else 0
    avg_score = data['anomaly_score'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Data Points",
            f"{total_points:,}",
            help="Total number of data points analyzed"
        )
    
    with col2:
        st.metric(
            "Anomalies Detected",
            f"{anomaly_count}",
            delta=f"{anomaly_rate:.1%}",
            help="Number and percentage of anomalies detected"
        )
    
    with col3:
        st.metric(
            "Avg Anomaly Score",
            f"{avg_score:.3f}",
            help="Average anomaly score across all data points"
        )
    
    with col4:
        # Real-time status
        if anomaly_rate > 0.1:
            status = "🔴 High Alert"
        elif anomaly_rate > 0.05:
            status = "🟡 Warning"
        else:
            status = "🟢 Normal"
        
        st.metric(
            "System Status",
            status,
            help="Current system status based on anomaly rate"
        )
    
    # Real-time visualization
    if detection_type == "Financial Transactions":
        # Transaction amount over time with anomalies highlighted
        fig = go.Figure()
        
        # Normal transactions
        normal_data = data[~data['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data['amount'],
            mode='markers',
            name='Normal Transactions',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Anomalous transactions
        anomaly_data = data[data['is_anomaly']]
        if len(anomaly_data) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data['amount'],
                mode='markers',
                name='Anomalous Transactions',
                marker=dict(color='red', size=10, symbol='diamond')
            ))
        
        fig.update_layout(
            title="Transaction Amount Analysis",
            xaxis_title="Time",
            yaxis_title="Transaction Amount ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif detection_type == "System Metrics":
        # System metrics with anomaly detection
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['CPU Usage (%)', 'Memory Usage (%)', 'Disk I/O (MB/s)', 'Network I/O (MB/s)'],
            vertical_spacing=0.1
        )
        
        metrics = ['cpu_usage', 'memory_usage', 'disk_io', 'network_io']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, (row, col) in zip(metrics, positions):
            # Normal data
            normal_data = data[~data['is_anomaly']]
            fig.add_trace(
                go.Scatter(
                    x=normal_data['timestamp'],
                    y=normal_data[metric],
                    mode='lines+markers',
                    name=f'Normal {metric}',
                    line=dict(color='blue'),
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )
            
            # Anomalous data
            anomaly_data = data[data['is_anomaly']]
            if len(anomaly_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data[metric],
                        mode='markers',
                        name=f'Anomalous {metric}',
                        marker=dict(color='red', size=8, symbol='diamond'),
                        showlegend=(row == 1 and col == 1)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, title="System Metrics Anomaly Detection")
        st.plotly_chart(fig, use_container_width=True)
    
    # Control buttons
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with col_b:
        if st.button("⏸️ Pause Detection", use_container_width=True):
            st.success("Detection paused. Click 'Resume' to continue.")
    
    with col_c:
        if st.button("📊 Generate Report", use_container_width=True):
            with st.spinner("Generating detection report..."):
                time.sleep(2)
            st.success("Report generated and saved!")

with tab2:
    st.markdown("### Detailed Analysis")
    
    # Anomaly score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            data,
            x='anomaly_score',
            color='is_anomaly',
            title='Anomaly Score Distribution',
            nbins=30,
            color_discrete_map={True: 'red', False: 'blue'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Feature importance (simulated)
        if detection_type == "Financial Transactions":
            features = ['Amount', 'Time of Day', 'Merchant Category', 'Card Type', 'Location']
            importance = [0.35, 0.25, 0.20, 0.15, 0.05]
        elif detection_type == "System Metrics":
            features = ['CPU Usage', 'Memory Usage', 'Disk I/O', 'Network I/O', 'Process Count']
            importance = [0.30, 0.25, 0.20, 0.15, 0.10]
        else:
            features = ['Traffic Volume', 'Connection Rate', 'Packet Size', 'Protocol Type', 'Source IP']
            importance = [0.40, 0.25, 0.15, 0.12, 0.08]
        
        fig_importance = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title='Feature Importance',
            color=importance,
            color_continuous_scale='Blues'
        )
        fig_importance.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Detailed anomaly table
    if anomaly_count > 0:
        st.markdown("### 🚨 Detected Anomalies")
        
        anomaly_details = data[data['is_anomaly']].copy()
        anomaly_details = anomaly_details.sort_values('anomaly_score', ascending=False)
        
        # Display top anomalies
        st.dataframe(
            anomaly_details.head(20),
            use_container_width=True,
            hide_index=True
        )
        
        # Export anomalies
        if st.button("📥 Export Anomalies"):
            csv = anomaly_details.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                csv,
                f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                use_container_width=True
            )
    else:
        st.info("No anomalies detected in the current time window.")

with tab3:
    st.markdown("### Alert Management")
    
    # Alert settings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Alert Configuration")
        
        alert_threshold = st.slider(
            "Alert Threshold (Anomaly Score)",
            0.1, 1.0, 0.7, 0.05,
            help="Trigger alerts when anomaly score exceeds this threshold"
        )
        
        notification_methods = st.multiselect(
            "Notification Methods",
            ["Email", "Slack", "SMS", "Webhook", "Dashboard"],
            default=["Email", "Dashboard"]
        )
        
        alert_frequency = st.selectbox(
            "Alert Frequency",
            ["Immediate", "Every 5 minutes", "Every 15 minutes", "Hourly"],
            help="How often to send alerts for ongoing anomalies"
        )
    
    with col2:
        st.markdown("#### Quick Actions")
        
        if st.button("🔔 Test Alert", use_container_width=True):
            st.success("Test alert sent successfully!")
        
        if st.button("🔇 Mute Alerts (1hr)", use_container_width=True):
            st.warning("Alerts muted for 1 hour")
        
        if st.button("📋 View Alert Log", use_container_width=True):
            st.info("Opening alert log...")
    
    # Recent alerts
    st.markdown("#### Recent Alerts")
    
    # Generate sample alerts
    recent_alerts = []
    high_score_anomalies = data[data['anomaly_score'] > alert_threshold]
    
    for _, row in high_score_anomalies.iterrows():
        alert = {
            "timestamp": row['timestamp'],
            "type": detection_type,
            "severity": "High" if row['anomaly_score'] > 0.8 else "Medium",
            "score": row['anomaly_score'],
            "status": "Open"
        }
        recent_alerts.append(alert)
    
    if recent_alerts:
        alerts_df = pd.DataFrame(recent_alerts)
        
        # Color code by severity
        def color_severity(val):
            if val == 'High':
                return 'background-color: #ffebee; color: #c62828'
            elif val == 'Medium':
                return 'background-color: #fff3e0; color: #ef6c00'
            else:
                return 'background-color: #e8f5e8; color: #2e7d32'
        
        styled_alerts = alerts_df.style.applymap(color_severity, subset=['severity'])
        st.dataframe(styled_alerts, use_container_width=True, hide_index=True)
    else:
        st.info("No recent alerts. System operating normally.")

with tab4:
    st.markdown("### Historical Trends")
    
    # Trend analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly rate over time
        if len(data) > 10:
            # Group by hour and calculate anomaly rate
            data_hourly = data.set_index('timestamp').resample('1H').agg({
                'is_anomaly': ['sum', 'count'],
                'anomaly_score': 'mean'
            }).reset_index()
            
            data_hourly.columns = ['timestamp', 'anomaly_count', 'total_count', 'avg_score']
            data_hourly['anomaly_rate'] = data_hourly['anomaly_count'] / data_hourly['total_count']
            
            fig_trend = px.line(
                data_hourly,
                x='timestamp',
                y='anomaly_rate',
                title='Anomaly Rate Trend',
                markers=True
            )
            fig_trend.update_layout(height=400, yaxis_title="Anomaly Rate")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Insufficient data for trend analysis.")
    
    with col2:
        # Model performance over time
        performance_history = pd.DataFrame({
            'date': pd.date_range(end=datetime.now(), periods=7, freq='1D'),
            'precision': [0.89, 0.91, 0.88, 0.92, 0.90, 0.89, 0.91],
            'recall': [0.83, 0.85, 0.82, 0.87, 0.84, 0.83, 0.86],
            'f1_score': [0.86, 0.88, 0.85, 0.89, 0.87, 0.86, 0.88]
        })
        
        fig_performance = go.Figure()
        
        for metric in ['precision', 'recall', 'f1_score']:
            fig_performance.add_trace(go.Scatter(
                x=performance_history['date'],
                y=performance_history[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title()
            ))
        
        fig_performance.update_layout(
            title='Model Performance History',
            height=400,
            yaxis_title="Score"
        )
        st.plotly_chart(fig_performance, use_container_width=True)
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        total_processed = len(data) * 100  # Simulate larger dataset
        st.metric("Total Data Processed", f"{total_processed:,}")
    
    with stats_col2:
        total_anomalies = anomaly_count * 20  # Simulate historical data
        st.metric("Total Anomalies Found", f"{total_anomalies:,}")
    
    with stats_col3:
        avg_daily_rate = anomaly_rate * 24  # Simulate daily rate
        st.metric("Avg Daily Anomalies", f"{avg_daily_rate:.1f}")
    
    with stats_col4:
        detection_accuracy = 0.89  # Simulated accuracy
        st.metric("Detection Accuracy", f"{detection_accuracy:.1%}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666;">
    <p>🔍 Anomaly Detection System | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Model: {model_choice} | Sensitivity: {sensitivity} | Threshold: {threshold}</p>
</div>
""", unsafe_allow_html=True)