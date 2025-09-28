"""
Backtesting Engine for Portfolio Strategies

This module provides comprehensive backtesting functionality for comparing
different portfolio optimization strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime


class BacktestEngine:
    """
    Portfolio backtesting engine with comprehensive performance metrics
    """
    
    def __init__(self, 
                 returns: pd.DataFrame,
                 rebalance_frequency: str = 'M',
                 transaction_cost: float = 0.001):
        """
        Initialize backtesting engine
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns data
        rebalance_frequency : str
            Rebalancing frequency ('D', 'W', 'M', 'Q')
        transaction_cost : float
            Transaction cost as fraction of trade value
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        
        # Create rebalancing dates
        self.rebalance_dates = self._get_rebalance_dates()
        
    def _get_rebalance_dates(self) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency"""
        if self.rebalance_frequency == 'M':
            # Monthly rebalancing - last business day of month
            return pd.date_range(
                start=self.returns.index[0],
                end=self.returns.index[-1],
                freq='M'
            ).intersection(self.returns.index).tolist()
        elif self.rebalance_frequency == 'Q':
            # Quarterly rebalancing
            return pd.date_range(
                start=self.returns.index[0],
                end=self.returns.index[-1],
                freq='Q'
            ).intersection(self.returns.index).tolist()
        elif self.rebalance_frequency == 'W':
            # Weekly rebalancing - Fridays
            return pd.date_range(
                start=self.returns.index[0],
                end=self.returns.index[-1],
                freq='W-FRI'
            ).intersection(self.returns.index).tolist()
        else:
            # Daily rebalancing
            return self.returns.index.tolist()
    
    def backtest_strategy(self, 
                         strategy_weights: pd.Series,
                         strategy_name: str = "Strategy") -> Dict:
        """
        Backtest a single strategy with fixed weights
        
        Parameters:
        -----------
        strategy_weights : pd.Series
            Portfolio weights (aligned with assets)
        strategy_name : str
            Name of the strategy
            
        Returns:
        --------
        results : dict
            Backtesting results and performance metrics
        """
        # Align weights with assets
        weights = strategy_weights.reindex(self.assets).fillna(0)
        weights = weights / weights.sum()  # Normalize
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Performance metrics
        metrics = self._calculate_performance_metrics(
            portfolio_returns, 
            cumulative_returns,
            strategy_name
        )
        
        return {
            'strategy_name': strategy_name,
            'weights': weights,
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'metrics': metrics
        }
    
    def backtest_dynamic_strategy(self,
                                 weight_function,
                                 strategy_name: str = "Dynamic Strategy",
                                 **kwargs) -> Dict:
        """
        Backtest a dynamic strategy that rebalances periodically
        
        Parameters:
        -----------
        weight_function : callable
            Function that returns portfolio weights given historical data
            Signature: weight_function(returns_history, **kwargs) -> pd.Series
        strategy_name : str
            Name of the strategy
        **kwargs : dict
            Additional arguments passed to weight_function
            
        Returns:
        --------
        results : dict
            Backtesting results and performance metrics
        """
        portfolio_returns = []
        cumulative_returns = []
        weights_history = []
        turnover_history = []
        
        current_weights = None
        
        for i, date in enumerate(self.returns.index):
            # Check if rebalancing is needed
            if date in self.rebalance_dates or current_weights is None:
                # Get historical data up to current date
                hist_returns = self.returns.loc[:date].iloc[:-1]  # Exclude current date
                
                if len(hist_returns) > 20:  # Minimum history requirement
                    try:
                        # Calculate new weights
                        new_weights = weight_function(hist_returns, **kwargs)
                        new_weights = new_weights.reindex(self.assets).fillna(0)
                        new_weights = new_weights / new_weights.sum()
                        
                        # Calculate turnover
                        if current_weights is not None:
                            turnover = np.abs(new_weights - current_weights).sum()
                        else:
                            turnover = 0.0
                        
                        turnover_history.append(turnover)
                        current_weights = new_weights.copy()
                        
                    except Exception as e:
                        warnings.warn(f"Failed to rebalance on {date}: {e}")
                        if current_weights is None:
                            current_weights = pd.Series(1/len(self.assets), index=self.assets)
            
            # Calculate portfolio return for current date
            if current_weights is not None:
                daily_return = (self.returns.loc[date] * current_weights).sum()
                
                # Apply transaction costs on rebalancing days
                if date in self.rebalance_dates and len(turnover_history) > 0:
                    daily_return -= self.transaction_cost * turnover_history[-1]
                
                portfolio_returns.append(daily_return)
                weights_history.append(current_weights.copy())
            else:
                portfolio_returns.append(0.0)
                weights_history.append(pd.Series(0, index=self.assets))
        
        # Convert to pandas Series
        portfolio_returns = pd.Series(portfolio_returns, index=self.returns.index)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Performance metrics
        metrics = self._calculate_performance_metrics(
            portfolio_returns,
            cumulative_returns, 
            strategy_name
        )
        
        # Add turnover metrics
        if turnover_history:
            metrics['avg_turnover'] = np.mean(turnover_history)
            metrics['total_transaction_costs'] = sum(t * self.transaction_cost for t in turnover_history)
        
        return {
            'strategy_name': strategy_name,
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'weights_history': weights_history,
            'turnover_history': turnover_history,
            'metrics': metrics
        }
    
    def _calculate_performance_metrics(self,
                                     returns: pd.Series,
                                     cumulative_returns: pd.Series,
                                     strategy_name: str) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Drawdown metrics
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate time to recovery from max drawdown
        max_dd_date = drawdown.idxmin()
        recovery_dates = cumulative_returns[cumulative_returns.index > max_dd_date]
        if len(recovery_dates) > 0:
            recovery_idx = recovery_dates[recovery_dates >= peak.loc[max_dd_date]].index
            if len(recovery_idx) > 0:
                recovery_days = (recovery_idx[0] - max_dd_date).days
            else:
                recovery_days = None  # Not recovered yet
        else:
            recovery_days = None
        
        # Risk metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Value at Risk (95% and 99%)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Hit ratio (percentage of positive returns)
        hit_ratio = (returns > 0).mean()
        
        # Calmar ratio (return/max drawdown)
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'recovery_days': recovery_days,
            'calmar_ratio': calmar_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'hit_ratio': hit_ratio,
            'start_date': returns.index[0],
            'end_date': returns.index[-1],
            'n_observations': len(returns)
        }
    
    def compare_strategies(self, strategies_results: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple strategies side by side
        
        Parameters:
        -----------
        strategies_results : list of dict
            List of strategy backtest results
            
        Returns:
        --------
        comparison : pd.DataFrame
            Comparison table of all strategies
        """
        comparison_data = {}
        
        for result in strategies_results:
            strategy_name = result['strategy_name']
            metrics = result['metrics']
            
            comparison_data[strategy_name] = {
                'Total Return (%)': metrics['total_return'] * 100,
                'Annual Return (%)': metrics['annualized_return'] * 100,
                'Annual Volatility (%)': metrics['annualized_volatility'] * 100,
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Sortino Ratio': metrics['sortino_ratio'],
                'Max Drawdown (%)': metrics['max_drawdown'] * 100,
                'Calmar Ratio': metrics['calmar_ratio'],
                'Hit Ratio (%)': metrics['hit_ratio'] * 100,
                'VaR 95% (%)': metrics['var_95'] * 100,
                'Skewness': metrics['skewness'],
                'Kurtosis': metrics['kurtosis']
            }
            
            # Add turnover if available
            if 'avg_turnover' in metrics:
                comparison_data[strategy_name]['Avg Turnover (%)'] = metrics['avg_turnover'] * 100
        
        return pd.DataFrame(comparison_data).T.round(2)
    
    def plot_performance(self, strategies_results: List[Dict], figsize=(15, 10)):
        """Plot comprehensive performance comparison"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Cumulative returns
        ax1 = axes[0, 0]
        for result in strategies_results:
            cumulative_returns = result['cumulative_returns']
            ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=result['strategy_name'], linewidth=2)
        ax1.set_title('📈 Cumulative Returns', fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[0, 1]
        for result in strategies_results:
            cumulative_returns = result['cumulative_returns']
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, 
                           alpha=0.3, label=result['strategy_name'])
        ax2.set_title('📉 Drawdown (%)', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio (252-day window)
        ax3 = axes[1, 0]
        for result in strategies_results:
            returns = result['returns']
            rolling_sharpe = (returns.rolling(252).mean() * 252) / (returns.rolling(252).std() * np.sqrt(252))
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values, 
                    label=result['strategy_name'], alpha=0.8)
        ax3.set_title('⚡ Rolling Sharpe Ratio (1Y)', fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Risk-Return scatter
        ax4 = axes[1, 1]
        for result in strategies_results:
            metrics = result['metrics']
            ax4.scatter(metrics['annualized_volatility'] * 100,
                       metrics['annualized_return'] * 100,
                       s=100, alpha=0.7, label=result['strategy_name'])
        ax4.set_xlabel('Volatility (%)')
        ax4.set_ylabel('Return (%)')
        ax4.set_title('📊 Risk-Return Profile', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
