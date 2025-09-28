"""
Black-Litterman Portfolio Optimization Configuration

Centralized configuration file for all model parameters, data sources,
and system settings.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class ModelParameters:
    """Black-Litterman model parameters"""
    # Core BL parameters
    tau: float = 0.05              # Prior uncertainty scaling factor
    risk_aversion: float = 3.0     # Risk aversion coefficient (delta)
    
    # Default parameter ranges for sensitivity analysis
    tau_range: List[float] = None
    risk_aversion_range: List[float] = None
    
    def __post_init__(self):
        if self.tau_range is None:
            self.tau_range = [0.01, 0.025, 0.05, 0.1, 0.2]
        if self.risk_aversion_range is None:
            self.risk_aversion_range = [1.0, 2.5, 5.0, 7.5, 10.0]

@dataclass
class DataConfiguration:
    """Data loading and processing configuration"""
    # Default tickers - diversified portfolio across sectors
    default_tickers: List[str] = None
    
    # Extended universe with bonds and ETFs
    extended_tickers: List[str] = None
    
    # Data source settings
    data_start_date: str = "2019-01-01"
    lookback_years: int = 5
    min_observations: int = 500
    
    # Fallback market caps (in billions USD)
    fallback_market_caps: Dict[str, float] = None
    
    def __post_init__(self):
        if self.default_tickers is None:
            self.default_tickers = [
                # Technology
                'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA',
                # Finance
                'JPM', 'BAC', 'WFC', 'GS',
                # Healthcare
                'JNJ', 'UNH', 'PFE', 'ABBV',
                # Consumer
                'AMZN', 'WMT', 'PG', 'KO', 'PEP',
                # Industrial
                'CAT', 'BA', 'GE'
            ]
        
        if self.extended_tickers is None:
            self.extended_tickers = self.default_tickers + [
                # ETFs
                'SPY',   # S&P 500 ETF
                'QQQ',   # NASDAQ ETF
                'IWM',   # Russell 2000 ETF
                'VTI',   # Total Stock Market ETF
                'VXUS',  # Total International Stock ETF
                'BND',   # Total Bond Market ETF
                'TLT',   # 20+ Year Treasury ETF
                'GLD',   # Gold ETF
                'VNQ',   # Real Estate ETF
                'XLE',   # Energy Sector ETF
                'XLF',   # Financial Sector ETF
                'XLK',   # Technology Sector ETF
                'XLV',   # Healthcare Sector ETF
                'XLI',   # Industrial Sector ETF
                'XLP',   # Consumer Staples ETF
                'XLY',   # Consumer Discretionary ETF
                'XLU',   # Utilities Sector ETF
                'XLB',   # Materials Sector ETF
                'XLRE'   # Real Estate Sector ETF
            ]
        
        if self.fallback_market_caps is None:
            self.fallback_market_caps = {
                # Large Cap Stocks (Billions USD)
                'AAPL': 3000, 'MSFT': 2800, 'GOOGL': 1700, 'AMZN': 1500,
                'NVDA': 1600, 'META': 800, 'TSLA': 800, 'JPM': 450,
                'JNJ': 420, 'UNH': 500, 'BAC': 250, 'WFC': 180,
                'PFE': 200, 'ABBV': 290, 'WMT': 400, 'PG': 380,
                'KO': 260, 'PEP': 240, 'CAT': 150, 'BA': 120,
                'GS': 120, 'GE': 100,
                # ETFs (Assets Under Management in Billions)
                'SPY': 400, 'QQQ': 200, 'IWM': 60, 'VTI': 300,
                'VXUS': 100, 'BND': 90, 'TLT': 20, 'GLD': 60,
                'VNQ': 30, 'XLE': 25, 'XLF': 40, 'XLK': 50,
                'XLV': 30, 'XLI': 15, 'XLP': 15, 'XLY': 20,
                'XLU': 15, 'XLB': 10, 'XLRE': 8
            }

@dataclass
class OptimizationSettings:
    """Portfolio optimization configuration"""
    # Default constraints
    long_only: bool = True
    max_weight: float = 0.4
    min_weight: float = 0.0
    
    # Advanced constraints
    max_sector_weight: Optional[float] = None
    max_turnover: Optional[float] = None
    transaction_cost: float = 0.001
    
    # Solver preferences (in order of preference)
    preferred_solvers: List[str] = None
    
    def __post_init__(self):
        if self.preferred_solvers is None:
            self.preferred_solvers = ['CLARABEL', 'OSQP', 'SCS', 'SCIPY']

@dataclass
class ViewsConfiguration:
    """Views framework configuration"""
    # Default confidence levels and their uncertainty scaling
    confidence_scales: Dict[str, float] = None
    
    # View types
    view_types: List[str] = None
    
    def __post_init__(self):
        if self.confidence_scales is None:
            self.confidence_scales = {
                'low': 2.0,      # High uncertainty (less confident)
                'medium': 1.0,   # Moderate uncertainty
                'high': 0.5      # Low uncertainty (more confident)
            }
        
        if self.view_types is None:
            self.view_types = [
                'relative_performance',  # Asset A outperforms Asset B
                'absolute_return',       # Asset A returns X%
                'sector_view',          # Sector returns X%
                'factor_view'           # Factor exposure view
            ]

@dataclass
class BacktestConfiguration:
    """Backtesting configuration"""
    # Rebalancing settings
    rebalance_frequency: str = 'M'  # 'D', 'W', 'M', 'Q'
    transaction_cost: float = 0.001
    
    # Performance metrics
    risk_free_rate: float = 0.02
    benchmark_ticker: str = 'SPY'
    
    # Market regimes for testing
    market_regimes: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.market_regimes is None:
            self.market_regimes = {
                'bull_market': {'start': '2016-01-01', 'end': '2020-02-01'},
                'covid_crash': {'start': '2020-02-01', 'end': '2020-04-01'},
                'recovery': {'start': '2020-04-01', 'end': '2022-01-01'},
                'inflation_period': {'start': '2022-01-01', 'end': '2023-01-01'},
                'recent': {'start': '2023-01-01', 'end': '2024-12-31'}
            }

@dataclass
class DashboardConfiguration:
    """Streamlit dashboard configuration"""
    # UI settings
    page_title: str = "Black-Litterman Portfolio Optimizer"
    page_icon: str = "🎯"
    layout: str = "wide"
    
    # Cache settings
    data_cache_ttl: int = 3600  # 1 hour
    model_cache_ttl: int = 300  # 5 minutes
    
    # Visualization settings
    default_plot_height: int = 600
    color_scheme: List[str] = None
    
    # Export settings
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
        
        if self.export_formats is None:
            self.export_formats = ['CSV', 'Excel', 'JSON', 'PDF']

@dataclass
class SystemConfiguration:
    """System-wide configuration"""
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Numerical precision
    float_precision: int = 6
    matrix_condition_threshold: float = 1e12
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance
    use_multiprocessing: bool = True
    n_jobs: int = -1  # Use all available cores

# Global configuration instance
class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelParameters()
        self.data = DataConfiguration()
        self.optimization = OptimizationSettings()
        self.views = ViewsConfiguration()
        self.backtest = BacktestConfiguration()
        self.dashboard = DashboardConfiguration()
        self.system = SystemConfiguration()
    
    def update_from_dict(self, config_dict: Dict):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'optimization': self.optimization.__dict__,
            'views': self.views.__dict__,
            'backtest': self.backtest.__dict__,
            'dashboard': self.dashboard.__dict__,
            'system': self.system.__dict__
        }

# Create global config instance
config = Config()

# Set numpy random seed
np.random.seed(config.system.random_seed)
