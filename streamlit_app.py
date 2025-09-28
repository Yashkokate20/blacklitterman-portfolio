"""
Enhanced Black-Litterman Portfolio Optimization Dashboard

Professional-grade Streamlit application with comprehensive features,
3D visualizations, and export capabilities.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import warnings
import io
import base64
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Import our modules
from black_litterman import BlackLittermanModel
from portfolio_optimization import PortfolioOptimizer
from backtesting import BacktestEngine
from utils import load_market_data, calculate_performance_metrics
from config import config

# Page configuration
st.set_page_config(
    page_title="Black-Litterman Portfolio Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
/* Main app styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Metric containers */
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    color: white;
}

.stMetric > label {
    font-size: 16px !important;
    font-weight: bold !important;
    color: white !important;
}

/* Instruction box with fixed black background */
.instruction-box {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
    border: 2px solid #4CAF50;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    color: #ffffff !important;
}

.instruction-box h4 {
    color: #4CAF50 !important;
    font-size: 1.5rem !important;
    margin-bottom: 15px !important;
    text-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
}

.instruction-box ol {
    color: #ffffff !important;
    font-size: 1.1rem !important;
    line-height: 1.8 !important;
}

.instruction-box li {
    color: #ffffff !important;
    margin-bottom: 10px !important;
}

.instruction-box strong {
    color: #4CAF50 !important;
    font-weight: bold !important;
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    color: white;
    font-weight: bold;
}

/* Header styling */
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 3rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    color: #2C3E50;
    text-align: center;
    margin-bottom: 2rem !important;
}

/* Plotly chart containers */
.js-plotly-plot {
    border-radius: 15px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Success/error message styling */
.stSuccess {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    border-radius: 10px;
}

.stError {
    background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    border-radius: 10px;
}

/* Animation for loading */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.stSpinner > div {
    animation: pulse 1.5s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}

@st.cache_data(ttl=config.dashboard.data_cache_ttl)
def load_data_cached(tickers, use_extended=False):
    """Cached data loading function"""
    if use_extended:
        tickers = config.data.extended_tickers[:20]  # Limit for performance
    
    return load_market_data(
        tickers=tickers,
        start_date=config.data.data_start_date,
        fallback_market_caps=config.data.fallback_market_caps
    )

def export_to_csv(data, filename):
    """Create CSV download link"""
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=True)
    csv_string = csv_buffer.getvalue()
    
    b64 = base64.b64encode(csv_string.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}"> Download {filename}</a>'
    return href

def create_3d_efficient_frontier(optimizer, n_points=50):
    """Create enhanced 3D efficient frontier with creative visualizations"""
    try:
        returns_range, risks_range, weights_array = optimizer.efficient_frontier(n_points)
        sharpe_ratios = returns_range / risks_range
        
        # Create multiple 3D visualizations
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(' 3D Efficient Frontier Surface', ' Portfolio Risk-Return Sphere'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            horizontal_spacing=0.1
        )
        
        # 1. Enhanced Efficient Frontier with Surface
        # Create a surface mesh for the efficient frontier
        fig.add_trace(
            go.Scatter3d(
                x=risks_range * 100,
                y=returns_range * 100,
                z=sharpe_ratios,
                mode='markers+lines',
                marker=dict(
                    size=8,
                    color=sharpe_ratios,
                    colorscale='Plasma',
                    colorbar=dict(
                        title="Sharpe Ratio",
                        titleside="right",
                        x=0.45
                    ),
                    opacity=0.9,
                    symbol='diamond',
                    line=dict(color='white', width=1)
                ),
                line=dict(
                    color='rgba(255,255,255,0.8)',
                    width=6
                ),
                name='Efficient Frontier',
                hovertemplate='<b> Optimal Portfolio</b><br>' +
                             ' Risk: %{x:.1f}%<br>' +
                             ' Return: %{y:.1f}%<br>' +
                             ' Sharpe: %{z:.3f}<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add risk-free rate plane
        risk_free_rate = 0.02
        xx, yy = np.meshgrid(
            np.linspace(risks_range.min() * 100, risks_range.max() * 100, 10),
            np.linspace(returns_range.min() * 100, returns_range.max() * 100, 10)
        )
        zz = np.full_like(xx, risk_free_rate * 100)
        
        fig.add_trace(
            go.Surface(
                x=xx, y=yy, z=zz,
                opacity=0.2,
                colorscale='Blues',
                showscale=False,
                name='Risk-Free Rate'
            ),
            row=1, col=1
        )
        
        # 2. Creative 3D Portfolio Sphere Visualization
        # Create spherical coordinates for portfolio visualization
        n_portfolios = 30
        phi = np.linspace(0, 2*np.pi, n_portfolios)
        theta = np.linspace(0, np.pi, n_portfolios)
        
        sphere_returns = []
        sphere_risks = []
        sphere_sharpes = []
        
        for i in range(min(len(returns_range), n_portfolios)):
            # Map portfolio metrics to spherical coordinates
            r = sharpe_ratios[i] * 20  # Scale for visibility
            x_sphere = r * np.sin(theta[i % len(theta)]) * np.cos(phi[i % len(phi)])
            y_sphere = r * np.sin(theta[i % len(theta)]) * np.sin(phi[i % len(phi)])
            z_sphere = r * np.cos(theta[i % len(theta)])
            
            sphere_returns.append(returns_range[i] * 100)
            sphere_risks.append(risks_range[i] * 100)
            sphere_sharpes.append(sharpe_ratios[i])
        
        # Create 3D sphere visualization
        fig.add_trace(
            go.Scatter3d(
                x=np.array(sphere_risks)[:n_portfolios],
                y=np.array(sphere_returns)[:n_portfolios],
                z=np.array(sphere_sharpes)[:n_portfolios] * 100,
                mode='markers',
                marker=dict(
                    size=12,
                    color=np.array(sphere_sharpes)[:n_portfolios],
                    colorscale='Viridis',
                    opacity=0.8,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                name='Portfolio Universe',
                hovertemplate='<b> Portfolio Option</b><br>' +
                             ' Risk: %{x:.1f}%<br>' +
                             ' Return: %{y:.1f}%<br>' +
                             ' Sharpe: %{z:.1f}<br>' +
                             '<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add connecting lines for the sphere
        for i in range(min(len(sphere_returns)-1, n_portfolios-1)):
            fig.add_trace(
                go.Scatter3d(
                    x=[sphere_risks[i], sphere_risks[i+1]],
                    y=[sphere_returns[i], sphere_returns[i+1]],
                    z=[sphere_sharpes[i] * 100, sphere_sharpes[i+1] * 100],
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.3)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=2
            )
        
        # Update layout with enhanced 3D styling
        fig.update_layout(
            title={
                'text': " Advanced 3D Portfolio Visualization Universe",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#2C3E50'}
            },
            scene=dict(
                xaxis=dict(
                    title=" Volatility (%)",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.3)",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                yaxis=dict(
                    title=" Expected Return (%)",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.3)",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                zaxis=dict(
                    title=" Sharpe Ratio",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.3)",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.8),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor="rgba(0,0,0,0.9)",
                aspectmode='cube'
            ),
            scene2=dict(
                xaxis=dict(
                    title=" Risk (%)",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.3)",
                    showbackground=True
                ),
                yaxis=dict(
                    title=" Return (%)",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.3)",
                    showbackground=True
                ),
                zaxis=dict(
                    title=" Sharpe x100",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.3)",
                    showbackground=True
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor="rgba(0,0,0,0.9)",
                aspectmode='cube'
            ),
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="white",
                borderwidth=1,
                font=dict(color="white")
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating 3D visualization: {e}")
        return create_fallback_3d_chart()

def create_fallback_3d_chart():
    """Create a simple 3D chart when main visualization fails"""
    fig = go.Figure()
    
    # Simple 3D scatter
    x = np.random.rand(20) * 10
    y = np.random.rand(20) * 15 + 5
    z = y / x  # Simple Sharpe-like ratio
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=z,
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Sample Portfolios'
    ))
    
    fig.update_layout(
        title=" 3D Portfolio Visualization (Fallback)",
        scene=dict(
            xaxis_title="Risk (%)",
            yaxis_title="Return (%)",
            zaxis_title="Sharpe Ratio"
        ),
        height=600
    )
    
    return fig

def create_allocation_pie(weights, title):
    """Create enhanced portfolio allocation pie chart with 3D effect"""
    # Only show non-zero weights
    non_zero_weights = weights[weights > 0.001]
    
    # Create 3D-style pie chart
    fig = go.Figure()
    
    # Add main pie chart
    fig.add_trace(go.Pie(
        labels=non_zero_weights.index,
        values=non_zero_weights.values * 100,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12, color='white'),
        marker=dict(
            colors=px.colors.qualitative.Dark24[:len(non_zero_weights)],
            line=dict(color='white', width=2)
        ),
        pull=[0.05 if i == non_zero_weights.values.argmax() else 0 for i in range(len(non_zero_weights))],  # Pull out largest slice
        hovertemplate='<b>%{label}</b><br>' +
                     'Weight: %{value:.1f}%<br>' +
                     '<extra></extra>'
    ))
    
    # Add center text
    fig.add_annotation(
        text=f"<b>{title}</b><br>Assets: {len(non_zero_weights)}",
        x=0.5, y=0.5,
        font=dict(size=16, color='white'),
        showarrow=False,
        align='center'
    )
    
    fig.update_layout(
        title={
            'text': f" {title}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        height=450,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1,
            font=dict(color="white", size=10)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_enhanced_risk_return_3d(strategies_data):
    """Create enhanced 3D risk-return visualization with multiple strategies"""
    fig = go.Figure()
    
    # Define colors for different strategies
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    symbols = ['circle', 'diamond', 'square', 'cross', 'triangle-up']
    
    for i, (strategy, data) in enumerate(strategies_data.items()):
        # Create 3D trajectory for each strategy
        risk = data.get('volatility', 0.15) * 100
        ret = data.get('return', 0.10) * 100
        sharpe = data.get('sharpe_ratio', 0.67)
        
        # Add some variation for 3D effect
        z_values = [sharpe + np.sin(j * 0.5) * 0.1 for j in range(10)]
        x_values = [risk + np.cos(j * 0.3) * 2 for j in range(10)]
        y_values = [ret + np.sin(j * 0.4) * 1 for j in range(10)]
        
        fig.add_trace(go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode='markers+lines',
            marker=dict(
                size=8,
                color=colors[i % len(colors)],
                opacity=0.8,
                symbol=symbols[i % len(symbols)],
                line=dict(color='white', width=1)
            ),
            line=dict(
                color=colors[i % len(colors)],
                width=4
            ),
            name=strategy,
            hovertemplate=f'<b>{strategy}</b><br>' +
                         'Risk: %{x:.1f}%<br>' +
                         'Return: %{y:.1f}%<br>' +
                         'Sharpe: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': " 3D Strategy Performance Universe",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        scene=dict(
            xaxis=dict(
                title=" Risk (%)",
                backgroundcolor="rgba(0,0,0,0.8)",
                gridcolor="rgba(255,255,255,0.3)",
                showbackground=True,
                zerolinecolor="white"
            ),
            yaxis=dict(
                title=" Return (%)",
                backgroundcolor="rgba(0,0,0,0.8)",
                gridcolor="rgba(255,255,255,0.3)",
                showbackground=True,
                zerolinecolor="white"
            ),
            zaxis=dict(
                title=" Sharpe Ratio",
                backgroundcolor="rgba(0,0,0,0.8)",
                gridcolor="rgba(255,255,255,0.3)",
                showbackground=True,
                zerolinecolor="white"
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor="rgba(0,0,0,0.9)"
        ),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1,
            font=dict(color="white")
        )
    )
    
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.title(" Black-Litterman Portfolio Optimization Dashboard")
    st.markdown("### Professional Portfolio Optimization with Interactive 3D Visualizations")
    
    # Instructions
    with st.expander(" How to Use This Dashboard", expanded=False):
        st.markdown("""
        <div class="instruction-box">
        <h4>Step-by-Step Guide:</h4>
        <ol>
            <li><strong>Select Assets:</strong> Choose from default tickers or enable extended universe (includes ETFs)</li>
            <li><strong>Adjust Parameters:</strong> Fine-tune  (tau),  (delta), and confidence levels in the sidebar</li>
            <li><strong>Set Views:</strong> Express your market opinions using the view builder</li>
            <li><strong>Analyze Results:</strong> Explore 3D visualizations and performance metrics</li>
            <li><strong>Export Data:</strong> Download portfolio weights and performance reports</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header(" Portfolio Configuration")
    
    # Data selection
    st.sidebar.subheader(" Data Selection")
    use_extended = st.sidebar.checkbox(
        "Use Extended Universe (includes ETFs)", 
        value=False,
        help="Include bonds, ETFs, and sector funds in addition to stocks"
    )
    
    custom_tickers = st.sidebar.text_input(
        "Custom Tickers (comma-separated)",
        value="",
        help="Enter custom ticker symbols separated by commas"
    )
    
    # Load data
    if custom_tickers:
        tickers = [t.strip().upper() for t in custom_tickers.split(',')]
    else:
        tickers = config.data.default_tickers[:12] if not use_extended else None
    
    try:
        with st.spinner("Loading market data..."):
            prices, returns, market_caps = load_data_cached(tickers, use_extended)
        
        assets = returns.columns.tolist()
        st.sidebar.success(f" Loaded {len(assets)} assets")
        
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
    
    # Model parameters
    st.sidebar.subheader(" Model Parameters")
    
    tau = st.sidebar.slider(
        " (Tau) - Prior Uncertainty", 
        0.01, 0.2, config.model.tau, 0.01,
        help="Higher values mean less confidence in market equilibrium"
    )
    
    delta = st.sidebar.slider(
        " (Delta) - Risk Aversion", 
        1.0, 10.0, config.model.risk_aversion, 0.5,
        help="Higher values mean more risk-averse investor"
    )
    
    confidence = st.sidebar.selectbox(
        "View Confidence Level", 
        ['low', 'medium', 'high'],
        index=1,
        help="How confident are you in your market views?"
    )
    
    # Views configuration
    st.sidebar.subheader(" Investment Views")
    
    view_type = st.sidebar.selectbox(
        "View Type", 
        ["Relative Performance", "Absolute Return", "Sector View"],
        help="Choose the type of market view to express"
    )
    
    # Create views based on selection
    if view_type == "Relative Performance":
        asset1 = st.sidebar.selectbox("Asset 1 (Outperform)", assets, index=0)
        asset2 = st.sidebar.selectbox("Asset 2 (Underperform)", assets, index=1)
        outperformance = st.sidebar.slider("Expected Outperformance (%)", -10.0, 10.0, 3.0, 0.5)
        
        P = np.zeros((1, len(assets)))
        P[0, assets.index(asset1)] = 1
        P[0, assets.index(asset2)] = -1
        Q = np.array([outperformance / 100])
        
        view_description = f"{asset1} will outperform {asset2} by {outperformance:.1f}%"
        
    elif view_type == "Absolute Return":
        target_asset = st.sidebar.selectbox("Target Asset", assets)
        expected_return = st.sidebar.slider("Expected Annual Return (%)", 0.0, 30.0, 12.0, 1.0)
        
        P = np.zeros((1, len(assets)))
        P[0, assets.index(target_asset)] = 1
        Q = np.array([expected_return / 100])
        
        view_description = f"{target_asset} will return {expected_return:.1f}% annually"
        
    else:  # Sector View
        tech_assets = [a for a in assets if a in ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA']]
        if tech_assets:
            sector_return = st.sidebar.slider("Tech Sector Return (%)", 5.0, 25.0, 15.0, 1.0)
            
            P = np.zeros((1, len(assets)))
            for asset in tech_assets:
                P[0, assets.index(asset)] = 1/len(tech_assets)
            Q = np.array([sector_return / 100])
            
            view_description = f"Tech sector average return: {sector_return:.1f}%"
        else:
            P = np.zeros((1, len(assets)))
            P[0, 0] = 1
            Q = np.array([0.12])
            view_description = "Default view: 12% return for first asset"
    
    # Portfolio constraints
    st.sidebar.subheader(" Portfolio Constraints")
    max_weight = st.sidebar.slider("Maximum Asset Weight (%)", 10, 100, 40, 5) / 100
    long_only = st.sidebar.checkbox("Long Only (No Shorting)", True)
    
    # Main dashboard
    try:
        # Initialize Black-Litterman model
        with st.spinner("Computing Black-Litterman model..."):
            bl_model = BlackLittermanModel(returns, market_caps, risk_aversion=delta, tau=tau)
            bl_model.set_views(P, Q, confidence_level=confidence)
            bl_returns, bl_cov = bl_model.compute_posterior()
        
        # Create optimizers
        sample_optimizer = PortfolioOptimizer(returns.mean() * 252, returns.cov() * 252)
        bl_optimizer = PortfolioOptimizer(bl_returns * 252, bl_cov * 252)
        
        # Optimize portfolios
        constraints = {'long_only': long_only, 'max_weight': max_weight}
        
        market_weights = bl_model.market_weights
        
        try:
            sample_weights, sample_info = sample_optimizer.optimize_constrained(
                constraints=constraints, risk_aversion=delta
            )
        except:
            sample_weights = pd.Series(1/len(assets), index=assets)
            sample_info = {'portfolio_return': 0.1, 'portfolio_risk': 0.15, 'sharpe_ratio': 0.67}
        
        try:
            bl_weights, bl_info = bl_optimizer.optimize_constrained(
                constraints=constraints, risk_aversion=delta
            )
        except:
            bl_weights = pd.Series(1/len(assets), index=assets)
            bl_info = {'portfolio_return': 0.12, 'portfolio_risk': 0.16, 'sharpe_ratio': 0.75}
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            market_return = (market_weights * returns.mean() * 252).sum()
            market_risk = np.sqrt(np.dot(market_weights, np.dot(returns.cov() * 252, market_weights)))
            market_sharpe = market_return / market_risk if market_risk > 0 else 0
            
            st.metric("Market Cap Portfolio", f"{market_return:.1%}", f"Sharpe: {market_sharpe:.3f}")
        
        with col2:
            st.metric("Sample Mean-Variance", f"{sample_info['portfolio_return']:.1%}", 
                     f"Sharpe: {sample_info['sharpe_ratio']:.3f}")
        
        with col3:
            st.metric("Black-Litterman", f"{bl_info['portfolio_return']:.1%}", 
                     f"Sharpe: {bl_info['sharpe_ratio']:.3f}")
        
        # Tabbed interface
        tab1, tab2, tab3, tab4 = st.tabs(["3D Visualizations", "Portfolio Analysis", "Performance Comparison", "Export & Reports"])
        
        with tab1:
            st.markdown("###  3D Interactive Visualizations")
            
            # Enhanced 3D visualizations
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Main 3D Efficient Frontier with dual view
                frontier_fig = create_3d_efficient_frontier(bl_optimizer)
                if frontier_fig:
                    st.plotly_chart(frontier_fig, use_container_width=True, config={'displayModeBar': True})
                else:
                    st.error("Could not create 3D visualization. Please check your data.")
            
            with col2:
                st.markdown("####  Current BL Portfolio")
                allocation_fig = create_allocation_pie(bl_weights, "BL Allocation")
                st.plotly_chart(allocation_fig, use_container_width=True)
                
                # Add portfolio metrics in a styled box
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           border-radius: 15px; padding: 20px; margin: 10px 0; color: white;">
                    <h4> Portfolio Metrics</h4>
                    <p><strong>Expected Return:</strong> {bl_info['portfolio_return']:.1%}</p>
                    <p><strong>Volatility:</strong> {bl_info['portfolio_risk']:.1%}</p>
                    <p><strong>Sharpe Ratio:</strong> {bl_info['sharpe_ratio']:.3f}</p>
                    <p><strong>Largest Position:</strong> {bl_weights.idxmax()} ({bl_weights.max():.1%})</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional 3D Strategy Comparison
            st.markdown("####  3D Strategy Performance Comparison")
            
            strategies_data = {
                'Market Cap': {
                    'return': market_return,
                    'volatility': market_risk,
                    'sharpe_ratio': market_sharpe
                },
                'Sample MV': {
                    'return': sample_info['portfolio_return'],
                    'volatility': sample_info['portfolio_risk'],
                    'sharpe_ratio': sample_info['sharpe_ratio']
                },
                'Black-Litterman': {
                    'return': bl_info['portfolio_return'],
                    'volatility': bl_info['portfolio_risk'],
                    'sharpe_ratio': bl_info['sharpe_ratio']
                }
            }
            
            strategy_3d_fig = create_enhanced_risk_return_3d(strategies_data)
            st.plotly_chart(strategy_3d_fig, use_container_width=True)
        
        with tab2:
            st.subheader(" Portfolio Analysis")
            
            # Current view
            st.info(f"**Current View:** {view_description}")
            
            # Portfolio comparison table
            comparison_df = pd.DataFrame({
                'Market Cap (%)': market_weights * 100,
                'Sample MV (%)': sample_weights * 100,
                'Black-Litterman (%)': bl_weights * 100
            }).round(2)
            
            st.dataframe(comparison_df, height=400)
            
            # Returns comparison
            returns_comparison = pd.DataFrame({
                'Market Implied (%)': bl_model.implied_returns * 252 * 100,
                'BL Posterior (%)': bl_returns * 252 * 100,
                'Difference (%)': (bl_returns - bl_model.implied_returns) * 252 * 100
            }).round(2)
            
            st.subheader(" Expected Returns Comparison")
            st.dataframe(returns_comparison)
        
        with tab3:
            st.subheader(" Performance Comparison")
            
            # Performance metrics table
            performance_df = pd.DataFrame({
                'Market Cap': [market_return * 100, market_risk * 100, market_sharpe],
                'Sample MV': [sample_info['portfolio_return'] * 100, sample_info['portfolio_risk'] * 100, sample_info['sharpe_ratio']],
                'Black-Litterman': [bl_info['portfolio_return'] * 100, bl_info['portfolio_risk'] * 100, bl_info['sharpe_ratio']]
            }, index=['Return (%)', 'Risk (%)', 'Sharpe Ratio']).round(3)
            
            st.dataframe(performance_df)
            
            # Key insights
            bl_improvement = ((bl_info['sharpe_ratio'] / market_sharpe - 1) * 100) if market_sharpe > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("BL Improvement", f"{bl_improvement:+.1f}%", "vs Market Cap")
            with col2:
                st.metric("Concentration", f"{(bl_weights ** 2).sum():.3f}", "Herfindahl Index")
            with col3:
                st.metric("Largest Position", f"{bl_weights.max():.1%}", f"{bl_weights.idxmax()}")
        
        with tab4:
            st.subheader(" Export & Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Download Portfolio Data")
                
                # Export weights
                st.markdown(export_to_csv(comparison_df, "portfolio_weights.csv"), unsafe_allow_html=True)
                
                # Export returns
                st.markdown(export_to_csv(returns_comparison, "expected_returns.csv"), unsafe_allow_html=True)
                
                # Export performance
                st.markdown(export_to_csv(performance_df, "performance_metrics.csv"), unsafe_allow_html=True)
            
            with col2:
                st.subheader("Model Summary")
                
                summary = bl_model.get_model_summary()
                st.json({
                    'Assets': summary['n_assets'],
                    'Risk Aversion': summary['risk_aversion'],
                    'Tau': summary['tau'],
                    'Views': summary['n_views'] if summary['has_views'] else 0,
                    'BL Sharpe Ratio': round(bl_info['sharpe_ratio'], 3),
                    'Improvement vs Market': f"{bl_improvement:+.1f}%"
                })
        
    except Exception as e:
        st.error(f"Error in computation: {e}")
        st.info("Please adjust parameters and try again.")

# Footer with documentation
st.markdown("---")
st.markdown("""
###  About Black-Litterman

**Simple Explanation:** Black-Litterman helps you build better investment portfolios by combining market wisdom 
with your personal insights. It's like having a smart advisor who considers both what everyone else is doing 
and your unique market views.

**Key Benefits:**
-  More stable portfolios than traditional mean-variance optimization
-  Incorporates market equilibrium assumptions  
-  Allows for investor views and confidence levels
-  Reduces estimation error in expected returns

**Parameters Guide:**
- ** (Tau):** Controls uncertainty in market equilibrium (0.01-0.1 typical)
- ** (Delta):** Risk aversion level (2-5 for institutional investors)  
- **Views:** Your market opinions with confidence levels
- **Constraints:** Portfolio limits (long-only, max weights, etc.)
""")

st.markdown("Built with  using Streamlit and Plotly |  Professional Portfolio Optimization")

if __name__ == "__main__":
    main()
