"""
Unit tests for Black-Litterman model implementation
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from black_litterman import BlackLittermanModel
from utils import generate_synthetic_data, validate_matrix_properties

class TestBlackLittermanModel:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
        prices, returns, market_caps = generate_synthetic_data(
            tickers, '2020-01-01', '2023-12-31', n_regimes=2
        )
        return returns, market_caps
    
    def test_initialization(self, sample_data):
        """Test model initialization"""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        assert bl_model.n_assets == len(returns.columns)
        assert bl_model.assets == list(returns.columns)
        assert bl_model.tau > 0
        assert bl_model.risk_aversion > 0
        assert len(bl_model.market_weights) == bl_model.n_assets
        assert abs(bl_model.market_weights.sum() - 1.0) < 1e-6
    
    def test_market_weights_computation(self, sample_data):
        """Test market weights computation"""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        # Check weights sum to 1
        assert abs(bl_model.market_weights.sum() - 1.0) < 1e-6
        
        # Check all weights are non-negative
        assert all(bl_model.market_weights >= 0)
        
        # Check largest market cap has largest weight
        largest_cap_asset = market_caps.idxmax()
        largest_weight_asset = bl_model.market_weights.idxmax()
        assert largest_cap_asset == largest_weight_asset
    
    def test_risk_aversion_estimation(self, sample_data):
        """Test risk aversion estimation"""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        # Risk aversion should be positive and reasonable
        assert bl_model.risk_aversion > 0
        assert bl_model.risk_aversion <= 10  # Upper bound check
    
    def test_implied_returns(self, sample_data):
        """Test implied equilibrium returns computation"""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        # Check implied returns shape
        assert len(bl_model.implied_returns) == bl_model.n_assets
        
        # Check implied returns are reasonable (not too extreme)
        annual_implied = bl_model.implied_returns * 252
        assert all(annual_implied > -2.0)  # Not less than -200%
        assert all(annual_implied < 20.0)  # Not more than 2000% (synthetic data can be extreme)
    
    def test_views_setting(self, sample_data):
        """Test setting investor views"""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        # Create simple relative view
        P = np.zeros((1, bl_model.n_assets))
        P[0, 0] = 1
        P[0, 1] = -1
        Q = np.array([0.05])
        
        bl_model.set_views(P, Q, confidence_level='medium')
        
        assert bl_model.P is not None
        assert bl_model.Q is not None
        assert bl_model.Omega is not None
        assert bl_model.P.shape == (1, bl_model.n_assets)
        assert bl_model.Q.shape == (1,)
        assert bl_model.Omega.shape == (1, 1)
    
    def test_posterior_computation(self, sample_data):
        """Test Black-Litterman posterior computation"""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        # Set views
        P = np.zeros((1, bl_model.n_assets))
        P[0, 0] = 1
        P[0, 1] = -1
        Q = np.array([0.05])
        bl_model.set_views(P, Q)
        
        # Compute posterior
        bl_returns, bl_cov = bl_model.compute_posterior()
        
        # Check shapes
        assert bl_returns.shape == (bl_model.n_assets,)
        assert bl_cov.shape == (bl_model.n_assets, bl_model.n_assets)
        
        # Check covariance matrix properties
        validation = validate_matrix_properties(bl_cov.values, check_positive_definite=True)
        assert validation['is_positive_definite']
        assert validation['is_symmetric']
        
        # Check returns are different from implied (views should have impact)
        diff = (bl_returns - bl_model.implied_returns).abs().sum()
        assert diff > 1e-8
    
    def test_confidence_levels(self, sample_data):
        """Test different confidence levels"""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        P = np.zeros((1, bl_model.n_assets))
        P[0, 0] = 1
        P[0, 1] = -1
        Q = np.array([0.05])
        
        # Test different confidence levels
        confidence_levels = ['low', 'medium', 'high']
        omegas = []
        
        for conf in confidence_levels:
            bl_model.set_views(P, Q, confidence_level=conf)
            omegas.append(bl_model.Omega[0, 0])
        
        # Higher confidence should lead to lower uncertainty (lower omega)
        assert omegas[2] < omegas[1] < omegas[0]  # high < medium < low
    
    def test_multiple_views(self, sample_data):
        """Test handling multiple views"""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        # Create two views
        P = np.zeros((2, bl_model.n_assets))
        P[0, 0] = 1
        P[0, 1] = -1  # First asset outperforms second
        P[1, 2] = 1   # Third asset absolute return
        Q = np.array([0.05, 0.15])
        
        bl_model.set_views(P, Q)
        bl_returns, bl_cov = bl_model.compute_posterior()
        
        # Should work without errors
        assert bl_returns.shape == (bl_model.n_assets,)
        assert bl_cov.shape == (bl_model.n_assets, bl_model.n_assets)
    
    def test_edge_cases(self, sample_data):
        """Test edge cases and error handling"""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        # Test with no views (should return implied returns)
        bl_returns_no_views, bl_cov_no_views = bl_model.compute_posterior()
        
        # Should be close to implied returns
        diff = (bl_returns_no_views - bl_model.implied_returns).abs().max()
        assert diff < 1e-6
    
    def test_numerical_stability(self, sample_data):
        """Test numerical stability with extreme parameters"""
        returns, market_caps = sample_data
        
        # Test with very small tau
        bl_model = BlackLittermanModel(returns, market_caps, tau=1e-6)
        P = np.zeros((1, bl_model.n_assets))
        P[0, 0] = 1
        P[0, 1] = -1
        Q = np.array([0.05])
        
        bl_model.set_views(P, Q)
        bl_returns, bl_cov = bl_model.compute_posterior()
        
        # Should not have NaN or inf values
        assert not np.isnan(bl_returns).any()
        assert not np.isinf(bl_returns).any()
        assert not np.isnan(bl_cov).any().any()
        assert not np.isinf(bl_cov).any().any()
    
    def test_model_summary(self, sample_data):
        """Test model summary generation"""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        summary = bl_model.get_model_summary()
        
        # Check required fields
        required_fields = ['n_assets', 'risk_aversion', 'tau', 'market_weights', 'implied_returns_annual', 'has_views']
        for field in required_fields:
            assert field in summary
        
        assert summary['n_assets'] == bl_model.n_assets
        assert summary['has_views'] == False
        
        # Test with views
        P = np.zeros((1, bl_model.n_assets))
        P[0, 0] = 1
        P[0, 1] = -1
        Q = np.array([0.05])
        bl_model.set_views(P, Q)
        
        summary_with_views = bl_model.get_model_summary()
        assert summary_with_views['has_views'] == True
        assert 'n_views' in summary_with_views
        assert 'view_values' in summary_with_views
