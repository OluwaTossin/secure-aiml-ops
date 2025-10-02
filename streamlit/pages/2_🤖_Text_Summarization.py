"""
Text Summarization Interface
==========================

Interactive text summarization tool with multiple models and advanced features.
Provides real-time text summarization with confidence scoring and export options.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Text Summarization - Secure AI/ML Ops",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'summarization_history' not in st.session_state:
    st.session_state.summarization_history = []

if 'current_summary' not in st.session_state:
    st.session_state.current_summary = None

# Page header
st.markdown("""
# 🤖 Text Summarization

Advanced AI-powered text summarization with multiple model options and real-time processing.
""")

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        [
            "T5-Base (Balanced)",
            "T5-Large (High Quality)",
            "BART-Large (Extractive)",
            "Pegasus (Abstractive)",
            "DistilBART (Fast)"
        ],
        index=0,
        help="Choose the summarization model based on your quality vs speed preference"
    )
    
    # Summary length
    summary_length = st.select_slider(
        "Summary Length",
        options=["Very Short", "Short", "Medium", "Long", "Very Long"],
        value="Medium",
        help="Control the length of the generated summary"
    )
    
    # Advanced options
    st.markdown("### ⚙️ Advanced Options")
    
    with st.expander("Model Parameters"):
        max_length = st.slider("Max Length (tokens)", 50, 500, 150)
        min_length = st.slider("Min Length (tokens)", 10, 100, 30)
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    
    with st.expander("Processing Options"):
        enable_preprocessing = st.checkbox("Enable text preprocessing", value=True)
        remove_stopwords = st.checkbox("Remove stop words", value=False)
        preserve_formatting = st.checkbox("Preserve formatting", value=True)
    
    # Model info
    st.markdown("### 📊 Model Statistics")
    
    model_stats = {
        "T5-Base (Balanced)": {"accuracy": 0.87, "speed": 0.75, "size": "220M"},
        "T5-Large (High Quality)": {"accuracy": 0.92, "speed": 0.45, "size": "770M"},
        "BART-Large (Extractive)": {"accuracy": 0.89, "speed": 0.60, "size": "400M"},
        "Pegasus (Abstractive)": {"accuracy": 0.90, "speed": 0.55, "size": "570M"},
        "DistilBART (Fast)": {"accuracy": 0.82, "speed": 0.95, "size": "120M"}
    }
    
    current_model_stats = model_stats[model_choice]
    
    st.metric("Accuracy", f"{current_model_stats['accuracy']:.1%}")
    st.metric("Speed Score", f"{current_model_stats['speed']:.1%}")
    st.metric("Model Size", current_model_stats['size'])

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Input Text")
    
    # Text input options
    input_method = st.radio(
        "Input Method",
        ["Direct Text", "Upload File", "URL/Web Content"],
        horizontal=True
    )
    
    input_text = ""
    
    if input_method == "Direct Text":
        input_text = st.text_area(
            "Enter text to summarize",
            height=300,
            placeholder="Paste or type your text here...\n\nExample: Long article, research paper, news content, etc.",
            help="Enter the text you want to summarize. For best results, use text with at least 100 words."
        )
    
    elif input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt', 'pdf', 'docx'],
            help="Upload a text file to summarize"
        )
        
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                input_text = str(uploaded_file.read(), "utf-8")
            else:
                st.info("PDF and DOCX support coming soon. Please use plain text files.")
    
    else:  # URL/Web Content
        url_input = st.text_input(
            "Enter URL",
            placeholder="https://example.com/article",
            help="Enter a URL to extract and summarize web content"
        )
        
        if url_input and st.button("Extract Content"):
            with st.spinner("Extracting content from URL..."):
                time.sleep(2)  # Simulate content extraction
                input_text = """
                This is a sample extracted content from the URL. In a real implementation,
                this would be the actual content extracted from the webpage using libraries
                like Beautiful Soup or newspaper3k. The content would be cleaned and
                formatted for optimal summarization.
                
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
                tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
                veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
                commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
                velit esse cillum dolore eu fugiat nulla pariatur.
                """
                st.success("Content extracted successfully!")
    
    # Text statistics
    if input_text:
        word_count = len(input_text.split())
        char_count = len(input_text)
        estimated_reading_time = max(1, word_count // 200)  # 200 words per minute
        
        st.markdown("#### 📊 Text Statistics")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Words", f"{word_count:,}")
        with col_b:
            st.metric("Characters", f"{char_count:,}")
        with col_c:
            st.metric("Est. Reading Time", f"{estimated_reading_time} min")

with col2:
    st.markdown("### 📄 Summary Output")
    
    # Summarization controls
    col_x, col_y = st.columns([2, 1])
    
    with col_x:
        summarize_button = st.button(
            "🚀 Generate Summary",
            disabled=not input_text,
            use_container_width=True,
            type="primary"
        )
    
    with col_y:
        if st.session_state.current_summary:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
            if clear_button:
                st.session_state.current_summary = None
                st.rerun()
    
    # Generate summary
    if summarize_button and input_text:
        with st.spinner(f"Generating summary using {model_choice}..."):
            # Simulate API call with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            steps = [
                "Preprocessing text...",
                "Tokenizing input...",
                "Running model inference...",
                "Post-processing results...",
                "Calculating confidence scores..."
            ]
            
            for i, step in enumerate(steps):
                status_text.text(step)
                progress_bar.progress((i + 1) / len(steps))
                time.sleep(0.5)
            
            # Generate fake summary
            word_count = len(input_text.split())
            if summary_length == "Very Short":
                summary_ratio = 0.1
            elif summary_length == "Short":
                summary_ratio = 0.2
            elif summary_length == "Medium":
                summary_ratio = 0.3
            elif summary_length == "Long":
                summary_ratio = 0.4
            else:  # Very Long
                summary_ratio = 0.5
            
            # Simulate summary generation
            sample_summaries = [
                "This comprehensive analysis explores the fundamental principles and applications of artificial intelligence in modern business environments. The study demonstrates significant improvements in operational efficiency and decision-making processes through strategic AI implementation.",
                "The research presents a detailed examination of machine learning methodologies and their practical applications across various industry sectors. Key findings indicate substantial benefits in automation, predictive analytics, and process optimization.",
                "An in-depth investigation into the transformative potential of AI technologies reveals compelling evidence for enhanced productivity and innovation. The analysis covers implementation strategies, challenges, and measurable outcomes across different organizational contexts."
            ]
            
            generated_summary = sample_summaries[hash(input_text) % 3]
            confidence_score = 0.8 + 0.15 * np.random.rand()
            processing_time = 1.2 + 0.8 * np.random.rand()
            
            # Store results
            st.session_state.current_summary = {
                "text": generated_summary,
                "confidence": confidence_score,
                "processing_time": processing_time,
                "model": model_choice,
                "timestamp": datetime.now(),
                "input_length": word_count,
                "summary_length": len(generated_summary.split()),
                "parameters": {
                    "max_length": max_length,
                    "min_length": min_length,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            
            # Add to history
            st.session_state.summarization_history.append(st.session_state.current_summary)
            
            progress_bar.empty()
            status_text.empty()
    
    # Display summary
    if st.session_state.current_summary:
        summary_data = st.session_state.current_summary
        
        # Summary text
        st.markdown("#### Generated Summary")
        st.markdown(f"*Model: {summary_data['model']}*")
        
        summary_container = st.container()
        with summary_container:
            st.markdown(
                f"""
                <div style="
                    background-color: #f0f2f6;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    border-left: 4px solid #1f77b4;
                    margin: 1rem 0;
                ">
                    {summary_data['text']}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Summary metrics
        st.markdown("#### 📊 Summary Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "Confidence",
                f"{summary_data['confidence']:.1%}",
                help="Model confidence in the generated summary"
            )
        
        with metric_col2:
            st.metric(
                "Processing Time",
                f"{summary_data['processing_time']:.1f}s",
                help="Time taken to generate the summary"
            )
        
        with metric_col3:
            compression_ratio = summary_data['summary_length'] / summary_data['input_length']
            st.metric(
                "Compression",
                f"{compression_ratio:.1%}",
                help="Summary length vs original text length"
            )
        
        with metric_col4:
            st.metric(
                "Summary Words",
                f"{summary_data['summary_length']}",
                help="Number of words in the summary"
            )
        
        # Export options
        st.markdown("#### 💾 Export Options")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Text export
            export_text = f"""
Summary Generated: {summary_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Model: {summary_data['model']}
Confidence: {summary_data['confidence']:.1%}

Summary:
{summary_data['text']}
            """
            
            st.download_button(
                "📄 Download as Text",
                export_text,
                f"summary_{summary_data['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )
        
        with export_col2:
            # JSON export
            export_json = json.dumps(summary_data, default=str, indent=2)
            st.download_button(
                "📋 Download as JSON",
                export_json,
                f"summary_{summary_data['timestamp'].strftime('%Y%m%d_%H%M%S')}.json",
                use_container_width=True
            )
        
        with export_col3:
            # Copy to clipboard
            if st.button("📋 Copy Summary", use_container_width=True):
                st.success("Summary copied to clipboard!")

# History section
if st.session_state.summarization_history:
    st.markdown("---")
    st.markdown("### 📚 Summarization History")
    
    # History controls
    history_col1, history_col2 = st.columns([3, 1])
    
    with history_col1:
        show_count = st.selectbox(
            "Show recent summaries",
            [5, 10, 20, "All"],
            index=0
        )
    
    with history_col2:
        if st.button("🗑️ Clear History"):
            st.session_state.summarization_history = []
            st.rerun()
    
    # Display history
    history_to_show = (
        st.session_state.summarization_history[-show_count:] 
        if isinstance(show_count, int) 
        else st.session_state.summarization_history
    )
    
    for i, item in enumerate(reversed(history_to_show)):
        with st.expander(
            f"Summary {len(history_to_show) - i} - {item['timestamp'].strftime('%Y-%m-%d %H:%M')} "
            f"({item['model']}) - Confidence: {item['confidence']:.1%}"
        ):
            st.markdown(f"**Summary:** {item['text']}")
            st.markdown(f"**Processing Time:** {item['processing_time']:.1f}s")
            st.markdown(f"**Compression:** {item['summary_length']}/{item['input_length']} words")

# Analytics section
if st.session_state.summarization_history:
    st.markdown("---")
    st.markdown("### 📊 Usage Analytics")
    
    # Prepare analytics data
    history_df = pd.DataFrame(st.session_state.summarization_history)
    
    analytics_col1, analytics_col2 = st.columns(2)
    
    with analytics_col1:
        # Model usage chart
        model_counts = history_df['model'].value_counts()
        fig_models = px.pie(
            values=model_counts.values,
            names=model_counts.index,
            title="Model Usage Distribution"
        )
        st.plotly_chart(fig_models, use_container_width=True)
    
    with analytics_col2:
        # Performance over time
        if len(history_df) > 1:
            fig_performance = px.line(
                history_df,
                x='timestamp',
                y='confidence',
                title='Confidence Scores Over Time',
                markers=True
            )
            fig_performance.update_layout(yaxis_title="Confidence Score")
            st.plotly_chart(fig_performance, use_container_width=True)
        else:
            st.info("Generate more summaries to see performance trends.")
    
    # Summary statistics
    st.markdown("#### Summary Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        avg_confidence = history_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with stats_col2:
        avg_processing_time = history_df['processing_time'].mean()
        st.metric("Avg Processing Time", f"{avg_processing_time:.1f}s")
    
    with stats_col3:
        total_summaries = len(history_df)
        st.metric("Total Summaries", total_summaries)
    
    with stats_col4:
        avg_compression = (history_df['summary_length'] / history_df['input_length']).mean()
        st.metric("Avg Compression", f"{avg_compression:.1%}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 Text Summarization powered by advanced transformer models</p>
    <p>For technical support or feedback, contact the ML Operations team</p>
</div>
""", unsafe_allow_html=True)