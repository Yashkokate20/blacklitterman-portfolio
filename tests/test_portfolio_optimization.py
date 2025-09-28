"""
Unit tests for portfolio optimization module
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_optimization import PortfolioOptimizer
from utils import generate_synthetic_data

class TestPortfolioOptimizer:
    
    @pytest.fixture
    def sample_optimizer(self):
        """Create sample optimizer for testing"""
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
        prices, returns, market_caps = generate_synthetic_data(tickers, '2020-01-01', '2023-12-31')
        
        expected_returns = returns.mean() * 252
        covariance_matrix = returns.cov() * 252
        
        return PortfolioOptimizer(expected_returns, covariance_matrix)
    
    def test_initialization(self, sample_optimizer):
        """Test optimizer initialization"""
        optimizer = sample_optimizer
        
        assert optimizer.n_assets > 0
        assert len(optimizer.assets) == optimizer.n_assets
        assert optimizer.expected_returns.shape == (optimizer.n_assets,)
        assert optimizer.covariance_matrix.shape == (optimizer.n_assets, optimizer.n_assets)
    
    def test_input_validation(self):
        """Test input validation"""
        # Mismatched dimensions should raise error
        expected_returns = pd.Series([0.1, 0.2, 0.3], index=['A', 'B', 'C'])
        covariance_matrix = pd.DataFrame(
            [[0.04, 0.02], [0.02, 0.09]], 
            index=['A', 'B'], 
            columns=['A', 'B']
        )
        
        with pytest.raises(ValueError):
            PortfolioOptimizer(expected_returns, covariance_matrix)
    
    def test_unconstrained_optimization(self, sample_optimizer):
        """Test unconstrained mean-variance optimization"""
        optimizer = sample_optimizer
        
        weights = optimizer.optimize_unconstrained(risk_aversion=3.0)
        
        # Check shape
        assert len(weights) == optimizer.n_assets
        
        # Check no NaN or inf values
        assert not weights.isnull().any()
        assert not np.isinf(weights).any()
        
        # Weights can be negative (short selling allowed)
        # But should be reasonable
        assert all(weights > -10)  # Not too extreme short positions
        assert all(weights < 10)   # Not too extreme long positions
    
    def test_constrained_optimization_basic(self, sample_optimizer):
        """Test basic constrained optimization"""
        optimizer = sample_optimizer
        
        weights, info = optimizer.optimize_constrained(
            constraints={'long_only': True, 'max_weight': 0.4},
            risk_aversion=3.0
        )
        
        # Check shape
        assert len(weights) == optimizer.n_assets
        
        # Check constraints
        assert abs(weights.sum() - 1.0) < 1e-6  # Budget constraint
        assert all(weights >= -1e-6)  # Long-only (allow small numerical errors)
        assert all(weights <= 0.4 + 1e-6)  # Max weight constraint
        
        # Check info dictionary
        required_fields = ['status', 'portfolio_return', 'portfolio_risk', 'sharpe_ratio']
        for field in required_fields:
            assert field in info
        
        assert info['status'] in ['optimal', 'optimal_inaccurate']
        assert info['portfolio_return'] > 0  # Should be positive
        assert info['portfolio_risk'] > 0    # Should be positive
    
    def test_different_constraints(self, sample_optimizer):
        """Test different constraint combinations"""
        optimizer = sample_optimizer
        
        # Test with minimum weight constraint
        weights1, info1 = optimizer.optimize_constrained(
            constraints={'long_only': True, 'min_weight': 0.05, 'max_weight': 0.3},
            risk_aversion=3.0
        )
        
        assert all(weights1 >= 0.05 - 1e-6)
        assert all(weights1 <= 0.3 + 1e-6)
        
        # Test allowing short selling
        weights2, info2 = optimizer.optimize_constrained(
            constraints={'long_only': False, 'max_weight': 0.5},
            risk_aversion=3.0
        )
        
        # Some weights might be negative
        assert abs(weights2.sum() - 1.0) < 1e-6
    
    def test_target_return_optimization(self, sample_optimizer):
        """Test optimization with target return constraint"""
        optimizer = sample_optimizer
        
        target_return = 0.12  # 12% annual return
        
        weights, info = optimizer.optimize_constrained(
            constraints={'long_only': True},
            target_return=target_return
        )
        
        # Check that target return is achieved (approximately)
        # Note: with synthetic data, the optimizer might find higher returns feasible
        achieved_return = info['portfolio_return']
        assert achieved_return >= target_return - 1e-3  # Should at least achieve target
        
        # Risk should be minimized for this return level
        assert info['portfolio_risk'] > 0
    
    def test_efficient_frontier(self, sample_optimizer):
        """Test efficient frontier computation"""
        optimizer = sample_optimizer
        
        returns, risks, weights_array = optimizer.efficient_frontier(
            n_points=10,
            constraints={'long_only': True}
        )
        
        # Check shapes
        assert len(returns) <= 10  # Some points might be infeasible
        assert len(risks) == len(returns)
        assert weights_array.shape[0] == len(returns)
        assert weights_array.shape[1] == optimizer.n_assets
        
        # Returns should be increasing
        if len(returns) > 1:
            assert all(returns[i] <= returns[i+1] for i in range(len(returns)-1))
        
        # All portfolios should satisfy budget constraint
        for i in range(len(returns)):
            portfolio_weights = weights_array[i]
            assert abs(portfolio_weights.sum() - 1.0) < 1e-3
    
    def test_portfolio_stats(self, sample_optimizer):
        """Test portfolio statistics computation"""
        optimizer = sample_optimizer
        
        # Use equal weights
        equal_weights = pd.Series(1/optimizer.n_assets, index=optimizer.assets)
        stats = optimizer.compute_portfolio_stats(equal_weights)
        
        # Check required fields
        required_fields = ['return', 'volatility', 'sharpe_ratio', 'weights_sum', 'max_weight', 'min_weight', 'n_nonzero', 'concentration']
        for field in required_fields:
            assert field in stats
        
        # Check values
        assert abs(stats['weights_sum'] - 1.0) < 1e-6
        assert stats['return'] > -1  # Not less than -100%
        assert stats['volatility'] > 0
        assert stats['max_weight'] >= stats['min_weight']
        assert stats['n_nonzero'] <= optimizer.n_assets
        assert 0 <= stats['concentration'] <= 1
    
    def test_solver_fallback(self, sample_optimizer):
        """Test that optimization works with different solvers"""
        optimizer = sample_optimizer
        
        # This should work with any available solver
        weights, info = optimizer.optimize_constrained(
            constraints={'long_only': True},
            risk_aversion=3.0
        )
        
        assert info['status'] in ['optimal', 'optimal_inaccurate']
        assert len(weights) == optimizer.n_assets
    
    def test_numerical_stability(self, sample_optimizer):
        """Test numerical stability with extreme parameters"""
        optimizer = sample_optimizer
        
        # Test with very high risk aversion
        weights_conservative, info_conservative = optimizer.optimize_constrained(
            constraints={'long_only': True},
            risk_aversion=100.0
        )
        
        # Should prefer low-risk assets
        assert not np.isnan(weights_conservative).any()
        assert not np.isinf(weights_conservative).any()
        
        # Test with very low risk aversion
        weights_aggressive, info_aggressive = optimizer.optimize_constrained(
            constraints={'long_only': True},
            risk_aversion=0.1
        )
        
        # Should prefer high-return assets
        assert not np.isnan(weights_aggressive).any()
        assert not np.isinf(weights_aggressive).any()
    
    def test_empty_constraints(self, sample_optimizer):
        """Test optimization with no additional constraints"""
        optimizer = sample_optimizer
        
        weights, info = optimizer.optimize_constrained(
            constraints=None,
            risk_aversion=3.0
        )
        
        # Should still satisfy budget constraint
        assert abs(weights.sum() - 1.0) < 1e-6
        assert info['status'] in ['optimal', 'optimal_inaccurate']
    
    def test_performance_consistency(self, sample_optimizer):
        """Test that computed performance matches expected performance"""
        optimizer = sample_optimizer
        
        weights, info = optimizer.optimize_constrained(
            constraints={'long_only': True},
            risk_aversion=3.0
        )
        
        # Manually compute portfolio stats
        manual_stats = optimizer.compute_portfolio_stats(weights)
        
        # Should match optimization info
        assert abs(manual_stats['return'] - info['portfolio_return']) < 1e-6
        assert abs(manual_stats['volatility'] - info['portfolio_risk']) < 1e-6
        assert abs(manual_stats['sharpe_ratio'] - info['sharpe_ratio']) < 1e-6
