"""
Layout Components
===============

Reusable layout components for the Streamlit application.
Provides consistent UI layouts and navigation elements.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

def create_sidebar(config=None):
    """Create standardized sidebar with navigation and info"""
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Navigation buttons
        if st.button("🏠 Dashboard", use_container_width=True):
            st.switch_page("app.py")
        
        if st.button("📊 Analytics", use_container_width=True):
            st.switch_page("pages/1_📊_Dashboard.py")
        
        if st.button("🤖 Text Summarization", use_container_width=True):
            st.switch_page("pages/2_🤖_Text_Summarization.py")
        
        if st.button("🔍 Anomaly Detection", use_container_width=True):
            st.switch_page("pages/3_🔍_Anomaly_Detection.py")
        
        if st.button("⚙️ Model Management", use_container_width=True):
            st.switch_page("pages/4_⚙️_Model_Management.py")
        
        st.markdown("---")
        
        # System information
        st.markdown("### ℹ️ System Info")
        st.markdown("**Status:** 🟢 Operational")
        st.markdown("**Uptime:** 15d 8h 23m")
        st.markdown("**Version:** v1.0.0")
        
        if config:
            st.markdown(f"**Environment:** {config.environment}")
            st.markdown(f"**Region:** {config.aws.region}")
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        st.metric("Active Models", "5", delta="1")
        st.metric("Daily Requests", "12.4K", delta="8.2%")
        st.metric("Success Rate", "99.7%", delta="0.1%")

def create_metrics_bar(metrics: Dict[str, Any]):
    """Create horizontal metrics bar"""
    cols = st.columns(len(metrics))
    
    for i, (label, data) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(data, dict):
                st.metric(
                    label=label,
                    value=data.get('value', 'N/A'),
                    delta=data.get('delta'),
                    delta_color=data.get('delta_color', 'normal'),
                    help=data.get('help')
                )
            else:
                st.metric(label=label, value=data)

def create_header(title: str, subtitle: str = None, icon: str = "🚀"):
    """Create standardized page header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <h1 style="color: #1f77b4; font-size: 2.5rem; margin-bottom: 0.5rem;">
                {icon} {title}
            </h1>
            {f'<p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)

def create_status_card(title: str, status: str, details: Dict[str, Any] = None):
    """Create status card with colored indicator"""
    status_colors = {
        'healthy': '🟢',
        'warning': '🟡', 
        'error': '🔴',
        'unknown': '⚪'
    }
    
    color_indicator = status_colors.get(status.lower(), '⚪')
    
    with st.container():
        st.markdown(f"""
        <div style="
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid {'#28a745' if status.lower() == 'healthy' else '#ffc107' if status.lower() == 'warning' else '#dc3545'};
        ">
            <h4 style="margin: 0 0 0.5rem 0;">{color_indicator} {title}</h4>
            <p style="margin: 0; color: #666;">Status: {status.title()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if details:
            with st.expander("Details"):
                for key, value in details.items():
                    st.write(f"**{key}:** {value}")

def create_alert_banner(message: str, alert_type: str = "info"):
    """Create alert banner"""
    colors = {
        'info': {'bg': '#d1ecf1', 'border': '#bee5eb', 'text': '#0c5460'},
        'success': {'bg': '#d4edda', 'border': '#c3e6cb', 'text': '#155724'},
        'warning': {'bg': '#fff3cd', 'border': '#ffeaa7', 'text': '#856404'},
        'error': {'bg': '#f8d7da', 'border': '#f5c6cb', 'text': '#721c24'}
    }
    
    style = colors.get(alert_type, colors['info'])
    
    st.markdown(f"""
    <div style="
        background-color: {style['bg']};
        border: 1px solid {style['border']};
        color: {style['text']};
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    ">
        {message}
    </div>
    """, unsafe_allow_html=True)

def create_data_table(df: pd.DataFrame, 
                     title: str = None,
                     show_download: bool = True,
                     max_rows: int = 100):
    """Create standardized data table with optional download"""
    if title:
        st.markdown(f"#### {title}")
    
    # Limit rows for performance
    display_df = df.head(max_rows) if len(df) > max_rows else df
    
    if len(df) > max_rows:
        st.info(f"Showing first {max_rows} of {len(df)} rows")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    if show_download and not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            use_container_width=True
        )

def create_progress_tracker(steps: List[Dict[str, Any]], current_step: int = 0):
    """Create progress tracker for multi-step processes"""
    st.markdown("### Progress")
    
    progress_container = st.container()
    
    with progress_container:
        for i, step in enumerate(steps):
            col1, col2 = st.columns([1, 10])
            
            with col1:
                if i < current_step:
                    st.markdown("✅")
                elif i == current_step:
                    st.markdown("🔄")
                else:
                    st.markdown("⏳")
            
            with col2:
                if i < current_step:
                    st.markdown(f"~~{step['title']}~~")
                elif i == current_step:
                    st.markdown(f"**{step['title']}**")
                else:
                    st.markdown(step['title'])
                
                if 'description' in step:
                    st.markdown(f"<small>{step['description']}</small>", unsafe_allow_html=True)

def create_comparison_table(data: List[Dict[str, Any]], 
                          title: str = None,
                          highlight_best: bool = True):
    """Create comparison table with highlighting"""
    if title:
        st.markdown(f"#### {title}")
    
    df = pd.DataFrame(data)
    
    if highlight_best and not df.empty:
        # Simple highlighting for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #d4edda' if v else '' for v in is_max]
        
        styled_df = df.style.apply(highlight_max, subset=numeric_cols)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

def create_info_panel(title: str, content: Dict[str, Any]):
    """Create information panel with key-value pairs"""
    st.markdown(f"#### {title}")
    
    with st.container():
        for key, value in content.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{key}:**")
            with col2:
                st.markdown(str(value))

def create_action_buttons(actions: List[Dict[str, Any]]):
    """Create row of action buttons"""
    if not actions:
        return
    
    cols = st.columns(len(actions))
    
    for i, action in enumerate(actions):
        with cols[i]:
            if st.button(
                action['label'],
                key=action.get('key'),
                help=action.get('help'),
                disabled=action.get('disabled', False),
                use_container_width=True,
                type=action.get('type', 'secondary')
            ):
                if 'callback' in action:
                    action['callback']()
                if 'message' in action:
                    st.success(action['message'])

def create_expandable_section(title: str, content_func, expanded: bool = False):
    """Create expandable section with dynamic content"""
    with st.expander(title, expanded=expanded):
        content_func()

def create_two_column_layout(left_content_func, right_content_func, ratio: List[int] = [1, 1]):
    """Create two-column layout with custom ratio"""
    col1, col2 = st.columns(ratio)
    
    with col1:
        left_content_func()
    
    with col2:
        right_content_func()

def create_tabs_layout(tab_config: Dict[str, callable]):
    """Create tabs layout with dynamic content"""
    tab_names = list(tab_config.keys())
    tabs = st.tabs(tab_names)
    
    for i, (tab_name, content_func) in enumerate(tab_config.items()):
        with tabs[i]:
            content_func()

def create_footer():
    """Create standardized footer"""
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Secure AI/ML Operations Platform | Built with ❤️ using Streamlit</p>
        <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Version 1.0.0</p>
    </div>
    """, unsafe_allow_html=True)