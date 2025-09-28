"""
Black-Litterman Portfolio Optimization - Interactive 3D Dashboard

This Streamlit app provides an interactive 3D dashboard for exploring
Black-Litterman portfolio optimization with real-time parameter adjustments.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Import our modules
from black_litterman import BlackLittermanModel
from portfolio_optimization import PortfolioOptimizer
import yfinance as yf
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Black-Litterman Portfolio Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}
.stMetric > label {
    font-size: 14px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load and cache market data"""
    try:
        # Default tickers
        tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'UNH']
        
        # Download data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)  # 5 years
        
        prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Get market caps
        market_caps = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                market_cap = info.get('marketCap', 100e9)  # Default 100B
                market_caps[ticker] = market_cap / 1e9
            except:
                market_caps[ticker] = 100  # Default 100B
        
        market_caps = pd.Series(market_caps)
        
        # Clean data
        prices = prices.dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        
        return prices, returns, market_caps
        
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        # Return sample data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
        
        # Generate synthetic data
        returns_data = np.random.multivariate_normal(
            mean=[0.001] * len(tickers),
            cov=np.eye(len(tickers)) * 0.0004 + np.full((len(tickers), len(tickers)), 0.0001),
            size=len(dates)
        )
        
        returns = pd.DataFrame(returns_data, index=dates, columns=tickers)
        prices = (returns + 1).cumprod() * 100
        market_caps = pd.Series([3000, 2800, 1500, 1700, 800], index=tickers)
        
        return prices, returns, market_caps

def create_3d_efficient_frontier(optimizer, n_points=50):
    """Create 3D efficient frontier visualization"""
    try:
        returns_range, risks_range, weights_array = optimizer.efficient_frontier(n_points)
        sharpe_ratios = returns_range / risks_range
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=risks_range * 100,
            y=returns_range * 100,
            z=sharpe_ratios,
            mode='markers+lines',
            marker=dict(
                size=5,
                color=sharpe_ratios,
                colorscale='Viridis',
                colorbar=dict(title="Sharpe Ratio"),
                opacity=0.8
            ),
            line=dict(color='blue', width=3),
            text=[f'Risk: {r:.1%}<br>Return: {ret:.1%}<br>Sharpe: {s:.3f}' 
                  for r, ret, s in zip(risks_range, returns_range, sharpe_ratios)],
            hovertemplate='<b>Portfolio Point</b><br>' +
                         'Volatility: %{x:.1f}%<br>' +
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
        
        return fig, weights_array
        
    except Exception as e:
        st.error(f"Error creating efficient frontier: {e}")
        return None, None

def create_portfolio_allocation_chart(weights, assets):
    """Create portfolio allocation pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=assets,
        values=weights * 100,
        hole=0.3,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="🥧 Portfolio Allocation",
        height=400,
        showlegend=True
    )
    
    return fig

def create_views_impact_chart(implied_returns, bl_returns, assets):
    """Create chart showing impact of views"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Market Implied',
        x=assets,
        y=implied_returns * 252 * 100,
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        name='BL Posterior',
        x=assets,
        y=bl_returns * 252 * 100,
        marker_color='darkblue',
        opacity=0.8
    ))
    
    fig.update_layout(
        title="📊 Impact of Views on Expected Returns",
        xaxis_title="Assets",
        yaxis_title="Annual Expected Return (%)",
        barmode='group',
        height=400
    )
    
    return fig

# Main app
def main():
    st.title("🎯 Black-Litterman Portfolio Optimization Dashboard")
    st.markdown("### Interactive 3D Visualization with Real-Time Parameter Tuning")
    
    # Load data
    with st.spinner("Loading market data..."):
        prices, returns, market_caps = load_data()
    
    assets = returns.columns.tolist()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Model Parameters")
    
    # Basic parameters
    tau = st.sidebar.slider("τ (Tau) - Prior Uncertainty", 0.01, 0.2, 0.05, 0.01,
                           help="Higher τ means less confidence in market equilibrium")
    
    delta = st.sidebar.slider("δ (Delta) - Risk Aversion", 1.0, 10.0, 3.0, 0.5,
                             help="Higher δ means more risk-averse investor")
    
    confidence = st.sidebar.selectbox("View Confidence Level", 
                                     ['low', 'medium', 'high'],
                                     index=1,
                                     help="How confident are you in your views?")
    
    # Views section
    st.sidebar.header("🎯 Investment Views")
    
    # Simple view interface
    view_type = st.sidebar.selectbox("View Type", 
                                    ["Relative Performance", "Absolute Return", "Sector View"])
    
    if view_type == "Relative Performance":
        asset1 = st.sidebar.selectbox("Asset 1 (Outperform)", assets, index=0)
        asset2 = st.sidebar.selectbox("Asset 2 (Underperform)", assets, index=1)
        outperformance = st.sidebar.slider("Expected Outperformance (%)", -10.0, 10.0, 3.0, 0.5)
        
        # Create P and Q matrices
        P = np.zeros((1, len(assets)))
        P[0, assets.index(asset1)] = 1
        P[0, assets.index(asset2)] = -1
        Q = np.array([outperformance / 100])
        
        st.sidebar.write(f"📝 View: {asset1} will outperform {asset2} by {outperformance:.1f}%")
        
    elif view_type == "Absolute Return":
        target_asset = st.sidebar.selectbox("Target Asset", assets)
        expected_return = st.sidebar.slider("Expected Annual Return (%)", 0.0, 30.0, 12.0, 1.0)
        
        P = np.zeros((1, len(assets)))
        P[0, assets.index(target_asset)] = 1
        Q = np.array([expected_return / 100])
        
        st.sidebar.write(f"📝 View: {target_asset} will return {expected_return:.1f}%")
    
    else:  # Sector View
        tech_assets = [a for a in assets if a in ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA']]
        if tech_assets:
            sector_return = st.sidebar.slider("Tech Sector Return (%)", 5.0, 25.0, 15.0, 1.0)
            
            P = np.zeros((1, len(assets)))
            for asset in tech_assets:
                if asset in assets:
                    P[0, assets.index(asset)] = 1/len(tech_assets)
            Q = np.array([sector_return / 100])
            
            st.sidebar.write(f"📝 View: Tech sector average return of {sector_return:.1f}%")
        else:
            P = np.zeros((1, len(assets)))
            P[0, 0] = 1
            Q = np.array([0.12])
    
    # Portfolio constraints
    st.sidebar.header("⚖️ Portfolio Constraints")
    max_weight = st.sidebar.slider("Maximum Asset Weight (%)", 10, 100, 40, 5) / 100
    long_only = st.sidebar.checkbox("Long Only (No Shorting)", True)
    
    # Compute Black-Litterman model
    try:
        with st.spinner("Computing Black-Litterman model..."):
            # Initialize model
            bl_model = BlackLittermanModel(
                returns=returns,
                market_caps=market_caps,
                risk_aversion=delta,
                tau=tau
            )
            
            # Set views
            bl_model.set_views(P, Q, confidence_level=confidence)
            
            # Compute posterior
            bl_returns, bl_cov = bl_model.compute_posterior()
            
            # Create optimizers
            sample_optimizer = PortfolioOptimizer(
                expected_returns=returns.mean() * 252,
                covariance_matrix=returns.cov() * 252
            )
            
            bl_optimizer = PortfolioOptimizer(
                expected_returns=bl_returns * 252,
                covariance_matrix=bl_cov * 252
            )
            
            # Optimize portfolios
            constraints = {
                'long_only': long_only,
                'max_weight': max_weight
            }
            
            market_weights = bl_model.market_weights
            sample_weights, sample_info = sample_optimizer.optimize_constrained(
                constraints=constraints, risk_aversion=delta
            )
            bl_weights, bl_info = bl_optimizer.optimize_constrained(
                constraints=constraints, risk_aversion=delta
            )
        
        # Main dashboard layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Market Cap Portfolio", 
                     f"{(market_weights * returns.mean() * 252).sum():.1%}",
                     f"Sharpe: {((market_weights * returns.mean() * 252).sum() / np.sqrt(np.dot(market_weights, np.dot(returns.cov() * 252, market_weights)))):.3f}")
        
        with col2:
            st.metric("Sample Mean-Variance",
                     f"{sample_info['portfolio_return']:.1%}",
                     f"Sharpe: {sample_info['sharpe_ratio']:.3f}")
        
        with col3:
            st.metric("Black-Litterman",
                     f"{bl_info['portfolio_return']:.1%}",
                     f"Sharpe: {bl_info['sharpe_ratio']:.3f}")
        
        # 3D Visualizations
        st.header("🌐 3D Interactive Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["3D Efficient Frontier", "Portfolio Allocations", "Views Impact"])
        
        with tab1:
            st.subheader("📈 3D Efficient Frontier Explorer")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create 3D efficient frontier
                frontier_fig, weights_array = create_3d_efficient_frontier(bl_optimizer)
                if frontier_fig:
                    selected_point = st.plotly_chart(frontier_fig, use_container_width=True)
            
            with col2:
                st.subheader("Portfolio Details")
                
                # Show current BL portfolio
                bl_allocation_fig = create_portfolio_allocation_chart(bl_weights, assets)
                st.plotly_chart(bl_allocation_fig, use_container_width=True)
                
                # Portfolio statistics
                st.write("**Black-Litterman Portfolio:**")
                stats_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight (%)': bl_weights * 100,
                    'Expected Return (%)': bl_returns * 252 * 100
                }).round(2)
                st.dataframe(stats_df, height=300)
        
        with tab2:
            st.subheader("🥧 Portfolio Allocation Comparison")
            
            # Create comparison of all three strategies
            comparison_fig = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
                subplot_titles=('Market Cap', 'Sample MV', 'Black-Litterman')
            )
            
            comparison_fig.add_trace(go.Pie(
                labels=assets, values=market_weights * 100, name="Market Cap"
            ), 1, 1)
            
            comparison_fig.add_trace(go.Pie(
                labels=assets, values=sample_weights * 100, name="Sample MV"
            ), 1, 2)
            
            comparison_fig.add_trace(go.Pie(
                labels=assets, values=bl_weights * 100, name="Black-Litterman"
            ), 1, 3)
            
            comparison_fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
            comparison_fig.update_layout(height=500, showlegend=False)
            
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Weight comparison table
            weights_comparison = pd.DataFrame({
                'Market Cap (%)': market_weights * 100,
                'Sample MV (%)': sample_weights * 100,
                'Black-Litterman (%)': bl_weights * 100
            }, index=assets).round(2)
            
            st.subheader("📊 Weights Comparison Table")
            st.dataframe(weights_comparison)
        
        with tab3:
            st.subheader("🎯 Impact of Views on Expected Returns")
            
            # Views impact chart
            views_fig = create_views_impact_chart(bl_model.implied_returns, bl_returns, assets)
            st.plotly_chart(views_fig, use_container_width=True)
            
            # Detailed comparison
            views_comparison = pd.DataFrame({
                'Market Implied (%)': bl_model.implied_returns * 252 * 100,
                'BL Posterior (%)': bl_returns * 252 * 100,
                'Difference (%)': (bl_returns - bl_model.implied_returns) * 252 * 100
            }, index=assets).round(2)
            
            st.dataframe(views_comparison)
            
            # Show current views
            st.subheader("📝 Current Views")
            if view_type == "Relative Performance":
                st.write(f"• {asset1} will outperform {asset2} by {outperformance:.1f}% annually")
            elif view_type == "Absolute Return":
                st.write(f"• {target_asset} will return {expected_return:.1f}% annually")
            else:
                st.write(f"• Technology sector will return {sector_return:.1f}% annually")
            
            st.write(f"• Confidence level: {confidence}")
            st.write(f"• View uncertainty (Ω diagonal): {np.diag(bl_model.Omega).round(4)}")
        
        # Performance metrics
        st.header("📊 Performance Metrics Comparison")
        
        metrics_df = pd.DataFrame({
            'Market Cap': [
                (market_weights * returns.mean() * 252).sum() * 100,
                np.sqrt(np.dot(market_weights, np.dot(returns.cov() * 252, market_weights))) * 100,
                ((market_weights * returns.mean() * 252).sum() / np.sqrt(np.dot(market_weights, np.dot(returns.cov() * 252, market_weights))))
            ],
            'Sample MV': [
                sample_info['portfolio_return'] * 100,
                sample_info['portfolio_risk'] * 100,
                sample_info['sharpe_ratio']
            ],
            'Black-Litterman': [
                bl_info['portfolio_return'] * 100,
                bl_info['portfolio_risk'] * 100,
                bl_info['sharpe_ratio']
            ]
        }, index=['Expected Return (%)', 'Volatility (%)', 'Sharpe Ratio']).round(3)
        
        st.dataframe(metrics_df)
        
        # Key insights
        st.header("💡 Key Insights")
        
        bl_improvement = ((bl_info['sharpe_ratio'] / ((market_weights * returns.mean() * 252).sum() / np.sqrt(np.dot(market_weights, np.dot(returns.cov() * 252, market_weights)))) - 1) * 100)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BL vs Market Cap",
                     f"{bl_improvement:+.1f}%",
                     "Sharpe Improvement")
        
        with col2:
            st.metric("Portfolio Concentration",
                     f"{(bl_weights ** 2).sum():.3f}",
                     "Herfindahl Index")
        
        with col3:
            st.metric("Largest Position",
                     f"{bl_weights.max():.1%}",
                     f"{bl_weights.idxmax()}")
    
    except Exception as e:
        st.error(f"Error in computation: {e}")
        st.info("Please adjust parameters and try again.")

# Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 About Black-Litterman
    
    **ELI5 Explanation:** Black-Litterman is like having a smart friend help you spend your allowance on toys. 
    It looks at what everyone else is buying (market portfolio), listens to your special insights, 
    and combines both wisely to give you the perfect mix!
    
    **Why professionals use it:**
    - ✅ Smarter than just copying everyone else
    - ✅ Safer than just following your hunches  
    - ✅ Mathematically blends crowd wisdom with your insights
    - ✅ Reduces big mistakes from overconfidence
    
    **Key Parameters:**
    - **τ (Tau):** How uncertain we are about market equilibrium (typically 0.01-0.1)
    - **δ (Delta):** Risk aversion level (typically 2-5 for institutional investors)
    - **Views:** Your insights about asset performance
    - **Confidence:** How sure you are about your views
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Plotly | 🎯 Black-Litterman Portfolio Optimization")

if __name__ == "__main__":
    main()
