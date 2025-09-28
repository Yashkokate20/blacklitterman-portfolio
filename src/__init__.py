"""
Black-Litterman Portfolio Optimization Package

This package provides a complete implementation of the Black-Litterman model
for portfolio optimization, including data handling, optimization, backtesting,
and visualization capabilities.
"""

__version__ = "1.0.0"
__author__ = "Senior Quant Developer"

# Import main classes and functions for easy access
from .black_litterman import BlackLittermanModel
from .portfolio_optimization import PortfolioOptimizer  
from .backtesting import BacktestEngine
from .visualization import PortfolioVisualizer

__all__ = [
    "BlackLittermanModel",
    "PortfolioOptimizer", 
    "BacktestEngine",
    "PortfolioVisualizer"
]
