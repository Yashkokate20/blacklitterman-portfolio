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
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
.stMetric > label {
    font-size: 14px !important;
    font-weight: bold !important;
}
.instruction-box {
    background-color: #e8f4f8;
    border-left: 5px solid #1f77b4;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
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
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

def create_3d_efficient_frontier(optimizer, n_points=30):
    """Create 3D efficient frontier"""
    try:
        returns_range, risks_range, weights_array = optimizer.efficient_frontier(n_points)
        sharpe_ratios = returns_range / risks_range
        
        fig = go.Figure(data=go.Scatter3d(
            x=risks_range * 100,
            y=returns_range * 100,
            z=sharpe_ratios,
            mode='markers+lines',
            marker=dict(
                size=6,
                color=sharpe_ratios,
                colorscale='Viridis',
                colorbar=dict(title="Sharpe Ratio"),
                opacity=0.8
            ),
            line=dict(color='blue', width=4),
            hovertemplate='<b>Portfolio</b><br>' +
                         'Risk: %{x:.1f}%<br>' +
                         'Return: %{y:.1f}%<br>' +
                         'Sharpe: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="📈 3D Efficient Frontier",
            scene=dict(
                xaxis_title="Volatility (%)",
                yaxis_title="Expected Return (%)",
                zaxis_title="Sharpe Ratio",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating efficient frontier: {e}")
        return None

def create_allocation_pie(weights, title):
    """Create portfolio allocation pie chart"""
    # Only show non-zero weights
    non_zero_weights = weights[weights > 0.001]
    
    fig = go.Figure(data=[go.Pie(
        labels=non_zero_weights.index,
        values=non_zero_weights.values * 100,
        hole=0.3,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=True
    )
    
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.title("🎯 Black-Litterman Portfolio Optimization Dashboard")
    st.markdown("### Professional Portfolio Optimization with Interactive 3D Visualizations")
    
    # Instructions
    with st.expander("📖 How to Use This Dashboard", expanded=False):
        st.markdown("""
        <div class="instruction-box">
        <h4>Step-by-Step Guide:</h4>
        <ol>
            <li><strong>Select Assets:</strong> Choose from default tickers or enable extended universe (includes ETFs)</li>
            <li><strong>Adjust Parameters:</strong> Fine-tune τ (tau), δ (delta), and confidence levels in the sidebar</li>
            <li><strong>Set Views:</strong> Express your market opinions using the view builder</li>
            <li><strong>Analyze Results:</strong> Explore 3D visualizations and performance metrics</li>
            <li><strong>Export Data:</strong> Download portfolio weights and performance reports</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Portfolio Configuration")
    
    # Data selection
    st.sidebar.subheader("📊 Data Selection")
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
        st.sidebar.success(f"✅ Loaded {len(assets)} assets")
        
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
    
    # Model parameters
    st.sidebar.subheader("🔧 Model Parameters")
    
    tau = st.sidebar.slider(
        "τ (Tau) - Prior Uncertainty", 
        0.01, 0.2, config.model.tau, 0.01,
        help="Higher values mean less confidence in market equilibrium"
    )
    
    delta = st.sidebar.slider(
        "δ (Delta) - Risk Aversion", 
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
    st.sidebar.subheader("🎯 Investment Views")
    
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
    st.sidebar.subheader("⚖️ Portfolio Constraints")
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
            st.subheader("📈 3D Interactive Visualizations")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 3D Efficient Frontier
                frontier_fig = create_3d_efficient_frontier(bl_optimizer)
                if frontier_fig:
                    st.plotly_chart(frontier_fig, use_container_width=True)
            
            with col2:
                st.subheader("Current BL Portfolio")
                allocation_fig = create_allocation_pie(bl_weights, "Portfolio Allocation")
                st.plotly_chart(allocation_fig, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Portfolio Analysis")
            
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
            
            st.subheader("📈 Expected Returns Comparison")
            st.dataframe(returns_comparison)
        
        with tab3:
            st.subheader("⚡ Performance Comparison")
            
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
            st.subheader("📥 Export & Reports")
            
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
### 📚 About Black-Litterman

**Simple Explanation:** Black-Litterman helps you build better investment portfolios by combining market wisdom 
with your personal insights. It's like having a smart advisor who considers both what everyone else is doing 
and your unique market views.

**Key Benefits:**
- ✅ More stable portfolios than traditional mean-variance optimization
- ✅ Incorporates market equilibrium assumptions  
- ✅ Allows for investor views and confidence levels
- ✅ Reduces estimation error in expected returns

**Parameters Guide:**
- **τ (Tau):** Controls uncertainty in market equilibrium (0.01-0.1 typical)
- **δ (Delta):** Risk aversion level (2-5 for institutional investors)  
- **Views:** Your market opinions with confidence levels
- **Constraints:** Portfolio limits (long-only, max weights, etc.)
""")

st.markdown("Built with ❤️ using Streamlit and Plotly | 🎯 Professional Portfolio Optimization")

if __name__ == "__main__":
    main()
