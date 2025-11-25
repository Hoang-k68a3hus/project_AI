"""
Pipeline Monitoring Dashboard.
Streamlit dashboard for monitoring automation pipelines.

Usage:
    streamlit run scripts/pipeline_dashboard.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    print("Required packages not installed. Run:")
    print("  pip install streamlit pandas plotly")
    sys.exit(1)

from scripts.utils import PipelineTracker


# =============================================================================
# Configuration
# =============================================================================

DASHBOARD_CONFIG = {
    'page_title': 'VieComRec Pipeline Monitor',
    'page_icon': '🔧',
    'refresh_interval': 60,  # seconds
    'registry_path': PROJECT_ROOT / 'artifacts' / 'cf' / 'registry.json',
    'deployment_history_path': PROJECT_ROOT / 'logs' / 'deployment_history.json'
}


# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data(ttl=60)
def load_pipeline_stats(days: int = 7) -> Dict[str, Any]:
    """Load pipeline statistics."""
    try:
        tracker = PipelineTracker()
        return tracker.get_stats(days=days)
    except Exception as e:
        return {'error': str(e)}


@st.cache_data(ttl=60)
def load_recent_runs(limit: int = 50) -> List[Dict]:
    """Load recent pipeline runs."""
    try:
        tracker = PipelineTracker()
        runs = tracker.get_recent_runs(limit=limit)
        return [r.to_dict() for r in runs]
    except Exception as e:
        return []


@st.cache_data(ttl=60)
def load_registry() -> Dict[str, Any]:
    """Load model registry."""
    try:
        if DASHBOARD_CONFIG['registry_path'].exists():
            with open(DASHBOARD_CONFIG['registry_path'], 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {'models': [], 'current_best': None}


@st.cache_data(ttl=60)
def load_deployment_history() -> List[Dict]:
    """Load deployment history."""
    try:
        if DASHBOARD_CONFIG['deployment_history_path'].exists():
            with open(DASHBOARD_CONFIG['deployment_history_path'], 'r') as f:
                data = json.load(f)
                return data.get('deployments', [])[-20:]  # Last 20
    except Exception:
        pass
    return []


# =============================================================================
# Dashboard Components
# =============================================================================

def render_status_overview():
    """Render the status overview section."""
    st.header("📊 Pipeline Status Overview")
    
    stats = load_pipeline_stats(days=7)
    
    if 'error' in stats:
        st.error(f"Failed to load stats: {stats['error']}")
        return
    
    pipeline_stats = stats.get('stats_by_pipeline', {})
    
    if not pipeline_stats:
        st.info("No pipeline runs recorded yet")
        return
    
    # Create status cards
    cols = st.columns(len(pipeline_stats))
    
    for i, (pipeline_name, pstats) in enumerate(pipeline_stats.items()):
        with cols[i]:
            success_rate = pstats.get('success_rate')
            success_count = pstats.get('success', 0)
            failed_count = pstats.get('failed', 0)
            
            # Determine status color
            if success_rate is None:
                status_color = "gray"
                status_emoji = "⚪"
            elif success_rate >= 0.9:
                status_color = "green"
                status_emoji = "🟢"
            elif success_rate >= 0.5:
                status_color = "orange"
                status_emoji = "🟡"
            else:
                status_color = "red"
                status_emoji = "🔴"
            
            st.metric(
                label=f"{status_emoji} {pipeline_name.replace('_', ' ').title()}",
                value=f"{success_rate*100:.0f}%" if success_rate else "N/A",
                delta=f"{success_count} OK, {failed_count} Failed"
            )


def render_recent_runs():
    """Render recent pipeline runs."""
    st.header("🔄 Recent Pipeline Runs")
    
    runs = load_recent_runs(limit=30)
    
    if not runs:
        st.info("No pipeline runs recorded yet")
        return
    
    # Create DataFrame
    df = pd.DataFrame(runs)
    
    # Format columns
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['status_icon'] = df['status'].map({
        'success': '✅',
        'failed': '❌',
        'running': '🔄',
        'skipped': '⏭️',
        'cancelled': '🚫'
    })
    
    # Display filters
    col1, col2 = st.columns(2)
    with col1:
        pipeline_filter = st.selectbox(
            "Filter by Pipeline",
            options=['All'] + list(df['pipeline_name'].unique()),
            index=0
        )
    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            options=['All'] + list(df['status'].unique()),
            index=0
        )
    
    # Apply filters
    filtered_df = df.copy()
    if pipeline_filter != 'All':
        filtered_df = filtered_df[filtered_df['pipeline_name'] == pipeline_filter]
    if status_filter != 'All':
        filtered_df = filtered_df[filtered_df['status'] == status_filter]
    
    # Display table
    display_cols = ['status_icon', 'pipeline_name', 'started_at', 'duration_seconds', 'error_message']
    display_df = filtered_df[display_cols].rename(columns={
        'status_icon': 'Status',
        'pipeline_name': 'Pipeline',
        'started_at': 'Started',
        'duration_seconds': 'Duration (s)',
        'error_message': 'Error'
    })
    
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True
    )


def render_success_rate_chart():
    """Render success rate chart over time."""
    st.header("📈 Success Rate Trend")
    
    runs = load_recent_runs(limit=100)
    
    if len(runs) < 5:
        st.info("Not enough data for trend analysis")
        return
    
    # Create DataFrame
    df = pd.DataFrame(runs)
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['date'] = df['started_at'].dt.date
    df['is_success'] = (df['status'] == 'success').astype(int)
    
    # Aggregate by date and pipeline
    daily_stats = df.groupby(['date', 'pipeline_name']).agg({
        'is_success': 'mean',
        'run_id': 'count'
    }).reset_index()
    daily_stats.columns = ['date', 'pipeline', 'success_rate', 'run_count']
    
    # Create chart
    fig = px.line(
        daily_stats,
        x='date',
        y='success_rate',
        color='pipeline',
        title='Daily Success Rate by Pipeline',
        labels={'success_rate': 'Success Rate', 'date': 'Date'},
        markers=True
    )
    
    fig.update_layout(
        yaxis=dict(tickformat='.0%', range=[0, 1.05]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_duration_chart():
    """Render pipeline duration chart."""
    st.header("⏱️ Pipeline Duration")
    
    runs = load_recent_runs(limit=50)
    
    if not runs:
        st.info("No data available")
        return
    
    # Filter successful runs with duration
    df = pd.DataFrame(runs)
    df = df[df['status'] == 'success']
    df = df[df['duration_seconds'].notna()]
    
    if df.empty:
        st.info("No successful runs to analyze")
        return
    
    # Create box plot
    fig = px.box(
        df,
        x='pipeline_name',
        y='duration_seconds',
        title='Pipeline Duration Distribution',
        labels={'pipeline_name': 'Pipeline', 'duration_seconds': 'Duration (seconds)'}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_model_info():
    """Render model information."""
    st.header("🤖 Model Registry")
    
    registry = load_registry()
    
    if not registry.get('models'):
        st.info("No models registered")
        return
    
    current_best = registry.get('current_best')
    
    # Display current best
    st.subheader("Active Model")
    
    best_model = next(
        (m for m in registry['models'] if m['model_id'] == current_best),
        None
    )
    
    if best_model:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model ID", best_model.get('model_id', 'N/A'))
        with col2:
            metrics = best_model.get('metrics', {})
            st.metric("Recall@10", f"{metrics.get('recall@10', 0):.4f}")
        with col3:
            st.metric("NDCG@10", f"{metrics.get('ndcg@10', 0):.4f}")
    
    # Model history
    st.subheader("Model History")
    
    models_df = pd.DataFrame(registry['models'])
    if not models_df.empty:
        models_df['registered_at'] = pd.to_datetime(models_df['registered_at'])
        models_df = models_df.sort_values('registered_at', ascending=False)
        
        # Extract metrics
        models_df['recall@10'] = models_df['metrics'].apply(
            lambda x: x.get('recall@10', 0) if isinstance(x, dict) else 0
        )
        
        display_df = models_df[['model_id', 'model_type', 'registered_at', 'recall@10', 'is_active']]
        display_df = display_df.rename(columns={
            'model_id': 'Model ID',
            'model_type': 'Type',
            'registered_at': 'Registered',
            'recall@10': 'Recall@10',
            'is_active': 'Active'
        })
        
        st.dataframe(display_df, hide_index=True, use_container_width=True)


def render_deployment_history():
    """Render deployment history."""
    st.header("🚀 Deployment History")
    
    deployments = load_deployment_history()
    
    if not deployments:
        st.info("No deployments recorded")
        return
    
    df = pd.DataFrame(deployments)
    df['deployed_at'] = pd.to_datetime(df['deployed_at'])
    df = df.sort_values('deployed_at', ascending=False)
    
    # Add status indicator
    df['status_icon'] = df['status'].map({
        'success': '✅',
        'failed': '❌',
        'pending_restart': '⏳'
    })
    
    display_df = df[['status_icon', 'model_id', 'deployed_at', 'git_commit']].rename(columns={
        'status_icon': 'Status',
        'model_id': 'Model ID',
        'deployed_at': 'Deployed',
        'git_commit': 'Commit'
    })
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)


def render_quick_actions():
    """Render quick action buttons."""
    st.header("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("🏥 Health Check", use_container_width=True):
            st.info("Run: python scripts/health_check.py --json")
    
    with col3:
        if st.button("🧹 Cleanup", use_container_width=True):
            st.info("Run: python scripts/cleanup_logs.py --dry-run")
    
    with col4:
        if st.button("📊 Full Report", use_container_width=True):
            st.info("See all tabs for full report")


# =============================================================================
# Main Dashboard
# =============================================================================

def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title=DASHBOARD_CONFIG['page_title'],
        page_icon=DASHBOARD_CONFIG['page_icon'],
        layout="wide"
    )
    
    st.title(f"{DASHBOARD_CONFIG['page_icon']} {DASHBOARD_CONFIG['page_title']}")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        time_range = st.selectbox(
            "Time Range",
            options=[7, 14, 30],
            format_func=lambda x: f"Last {x} days",
            index=0
        )
        
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if auto_refresh:
            st.info(f"Refreshing every {DASHBOARD_CONFIG['refresh_interval']}s")
        
        st.divider()
        st.header("Links")
        st.markdown("""
        - [API Health](/health)
        - [API Docs](/docs)
        """)
    
    # Main content
    tabs = st.tabs([
        "📊 Overview",
        "🔄 Runs",
        "📈 Trends",
        "🤖 Models",
        "🚀 Deployments"
    ])
    
    with tabs[0]:
        render_status_overview()
        st.divider()
        render_quick_actions()
    
    with tabs[1]:
        render_recent_runs()
    
    with tabs[2]:
        render_success_rate_chart()
        st.divider()
        render_duration_chart()
    
    with tabs[3]:
        render_model_info()
    
    with tabs[4]:
        render_deployment_history()
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(DASHBOARD_CONFIG['refresh_interval'])
        st.rerun()


if __name__ == "__main__":
    main()
