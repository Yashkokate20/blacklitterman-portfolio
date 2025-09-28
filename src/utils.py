"""
Utility Functions for Black-Litterman Portfolio Optimization

Consolidated utility functions used across the project.
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", format_str: str = None):
    """Setup logging configuration"""
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()), format=format_str, force=True
    )


def validate_matrix_properties(
    matrix: np.ndarray,
    name: str = "Matrix",
    check_positive_definite: bool = True,
    condition_threshold: float = 1e12,
) -> Dict[str, bool]:
    """
    Validate matrix properties for numerical stability

    Parameters:
    -----------
    matrix : np.ndarray
        Matrix to validate
    name : str
        Name of matrix for logging
    check_positive_definite : bool
        Whether to check positive definiteness
    condition_threshold : float
        Threshold for condition number warning

    Returns:
    --------
    validation_results : dict
        Dictionary of validation results
    """
    results = {}

    # Check if square
    results["is_square"] = matrix.shape[0] == matrix.shape[1]
    if not results["is_square"]:
        logger.warning(f"{name} is not square: {matrix.shape}")

    # Check for NaN or inf
    results["has_nan"] = np.isnan(matrix).any()
    results["has_inf"] = np.isinf(matrix).any()

    if results["has_nan"]:
        logger.error(f"{name} contains NaN values")
    if results["has_inf"]:
        logger.error(f"{name} contains infinite values")

    # Check symmetry (for covariance matrices)
    if results["is_square"]:
        results["is_symmetric"] = np.allclose(matrix, matrix.T, rtol=1e-10)
        if not results["is_symmetric"]:
            logger.warning(f"{name} is not symmetric")

    # Check positive definiteness
    if check_positive_definite and results["is_square"] and results["is_symmetric"]:
        try:
            eigenvals = np.linalg.eigvals(matrix)
            results["is_positive_definite"] = np.all(eigenvals > 1e-8)
            results["min_eigenvalue"] = np.min(eigenvals)

            if not results["is_positive_definite"]:
                logger.warning(
                    f"{name} is not positive definite (min eigenvalue: {results['min_eigenvalue']:.2e})"
                )
        except:
            results["is_positive_definite"] = False
            logger.error(f"Failed to compute eigenvalues for {name}")

    # Check condition number
    if results["is_square"]:
        try:
            results["condition_number"] = np.linalg.cond(matrix)
            results["is_well_conditioned"] = (
                results["condition_number"] < condition_threshold
            )

            if not results["is_well_conditioned"]:
                logger.warning(
                    f"{name} is poorly conditioned (condition number: {results['condition_number']:.2e})"
                )
        except:
            results["condition_number"] = np.inf
            results["is_well_conditioned"] = False

    return results


def clean_returns_data(
    returns: pd.DataFrame, min_observations: int = 500, max_missing_pct: float = 0.05
) -> pd.DataFrame:
    """
    Clean returns data by removing assets with insufficient data

    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    min_observations : int
        Minimum number of observations required
    max_missing_pct : float
        Maximum percentage of missing values allowed

    Returns:
    --------
    cleaned_returns : pd.DataFrame
        Cleaned returns data
    """
    logger.info(
        f"Cleaning returns data: {returns.shape[0]} obs × {returns.shape[1]} assets"
    )

    # Remove assets with too few observations
    valid_assets = []
    for col in returns.columns:
        n_valid = returns[col].count()
        missing_pct = (len(returns) - n_valid) / len(returns)

        if n_valid >= min_observations and missing_pct <= max_missing_pct:
            valid_assets.append(col)
        else:
            logger.info(f"Removing {col}: {n_valid} obs ({missing_pct:.1%} missing)")

    cleaned_returns = returns[valid_assets].copy()

    # Forward fill small gaps (up to 5 days)
    cleaned_returns = cleaned_returns.fillna(method="ffill", limit=5)

    # Drop remaining rows with NaN
    cleaned_returns = cleaned_returns.dropna()

    logger.info(
        f"Cleaned data: {cleaned_returns.shape[0]} obs × {cleaned_returns.shape[1]} assets"
    )

    return cleaned_returns


def estimate_robust_covariance(
    returns: pd.DataFrame, method: str = "ledoit_wolf"
) -> Tuple[pd.DataFrame, float]:
    """
    Estimate robust covariance matrix

    Parameters:
    -----------
    returns : pd.DataFrame
        Returns data
    method : str
        Estimation method ('ledoit_wolf', 'sample')

    Returns:
    --------
    robust_cov : pd.DataFrame
        Robust covariance matrix
    shrinkage : float
        Shrinkage intensity (for Ledoit-Wolf)
    """
    if method == "ledoit_wolf":
        logger.info("Estimating robust covariance using Ledoit-Wolf shrinkage")

        lw_estimator = LedoitWolf()
        robust_cov_matrix, shrinkage = (
            lw_estimator.fit(returns.values).covariance_,
            lw_estimator.shrinkage_,
        )

        robust_cov = pd.DataFrame(
            robust_cov_matrix, index=returns.columns, columns=returns.columns
        )

        logger.info(f"Ledoit-Wolf shrinkage intensity: {shrinkage:.3f}")

    else:  # sample covariance
        logger.info("Using sample covariance matrix")
        robust_cov = returns.cov()
        shrinkage = 0.0

    # Validate the covariance matrix
    validation = validate_matrix_properties(robust_cov.values, "Robust Covariance")

    return robust_cov, shrinkage


def generate_synthetic_data(
    tickers: List[str],
    start_date: str = "2020-01-01",
    end_date: str = None,
    n_regimes: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Generate synthetic price and returns data with multiple market regimes

    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date for synthetic data
    end_date : str
        End date for synthetic data
    n_regimes : int
        Number of market regimes to simulate

    Returns:
    --------
    prices : pd.DataFrame
        Synthetic price data
    returns : pd.DataFrame
        Synthetic returns data
    market_caps : pd.Series
        Synthetic market capitalizations
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Generating synthetic data for {len(tickers)} assets")

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)
    n_assets = len(tickers)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate regime-switching parameters
    regime_length = n_days // n_regimes
    returns_data = []

    for regime in range(n_regimes):
        start_idx = regime * regime_length
        end_idx = (regime + 1) * regime_length if regime < n_regimes - 1 else n_days
        regime_days = end_idx - start_idx

        # Different market conditions for each regime
        if regime == 0:  # Bull market
            mean_returns = np.random.uniform(0.0008, 0.0015, n_assets)  # High returns
            volatility = np.random.uniform(0.15, 0.25, n_assets)  # Moderate vol
            correlation_level = 0.4  # Lower correlation
        elif regime == 1:  # Bear market / Crisis
            mean_returns = np.random.uniform(
                -0.0010, 0.0005, n_assets
            )  # Negative returns
            volatility = np.random.uniform(0.25, 0.40, n_assets)  # High volatility
            correlation_level = 0.7  # Higher correlation (contagion)
        else:  # Recovery / Normal market
            mean_returns = np.random.uniform(
                0.0005, 0.0010, n_assets
            )  # Moderate returns
            volatility = np.random.uniform(0.18, 0.30, n_assets)  # Moderate volatility
            correlation_level = 0.5  # Moderate correlation

        # Create correlation matrix
        correlation_matrix = np.full((n_assets, n_assets), correlation_level)
        np.fill_diagonal(correlation_matrix, 1.0)

        # Add some randomness to correlations
        random_corr = np.random.uniform(-0.1, 0.1, (n_assets, n_assets))
        random_corr = (random_corr + random_corr.T) / 2
        np.fill_diagonal(random_corr, 0)
        correlation_matrix += random_corr

        # Ensure valid correlation matrix
        correlation_matrix = np.clip(correlation_matrix, -0.99, 0.99)
        np.fill_diagonal(correlation_matrix, 1.0)

        # Create covariance matrix
        vol_matrix = np.outer(volatility, volatility)
        covariance_matrix = correlation_matrix * vol_matrix

        # Generate returns for this regime
        regime_returns = multivariate_normal.rvs(
            mean=mean_returns, cov=covariance_matrix, size=regime_days
        )

        returns_data.append(regime_returns)

        logger.info(
            f"Regime {regime + 1}: {regime_days} days, "
            f"avg return: {mean_returns.mean()*252:.1%}, "
            f"avg vol: {volatility.mean():.1%}"
        )

    # Combine all regimes
    all_returns = np.vstack(returns_data)

    # Create returns DataFrame
    returns = pd.DataFrame(
        all_returns, index=dates[: len(all_returns)], columns=tickers
    )

    # Generate prices from returns
    initial_prices = np.random.uniform(50, 300, n_assets)
    prices = pd.DataFrame(index=returns.index, columns=tickers)
    prices.iloc[0] = initial_prices

    for i in range(1, len(prices)):
        prices.iloc[i] = prices.iloc[i - 1] * (1 + returns.iloc[i])

    # Generate realistic market caps
    market_cap_base = np.random.lognormal(mean=np.log(100), sigma=1.5, size=n_assets)
    market_caps = pd.Series(market_cap_base, index=tickers)

    logger.info(f"Generated synthetic data: {len(prices)} days × {len(tickers)} assets")

    return prices, returns, market_caps


def load_market_data(
    tickers: List[str],
    start_date: str = "2019-01-01",
    end_date: str = None,
    fallback_market_caps: Dict[str, float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load market data with robust error handling and fallbacks

    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date for data
    end_date : str
        End date for data
    fallback_market_caps : dict
        Fallback market capitalizations

    Returns:
    --------
    prices : pd.DataFrame
        Price data
    returns : pd.DataFrame
        Returns data
    market_caps : pd.Series
        Market capitalizations
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Loading market data for {len(tickers)} tickers")

    try:
        # Download price data
        logger.info("Downloading price data from yfinance...")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        # Handle data structure
        if len(tickers) == 1:
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Adj Close"].to_frame()
                prices.columns = tickers
            elif "Adj Close" in data.columns:
                prices = data[["Adj Close"]].copy()
                prices.columns = tickers
            else:
                prices = data.to_frame()
                prices.columns = tickers
        else:
            if isinstance(data.columns, pd.MultiIndex):
                if "Adj Close" in data.columns.levels[0]:
                    prices = data["Adj Close"].copy()
                else:
                    # Take the first level if it's not Adj Close
                    prices = data.iloc[:, : len(tickers)].copy()
                    if isinstance(prices.columns, pd.MultiIndex):
                        prices.columns = tickers[: len(prices.columns)]
                    else:
                        prices.columns = tickers
            else:
                # Single level columns - assume it's already price data
                prices = data.copy()
                if len(prices.columns) != len(tickers):
                    # Take only the number of columns we need
                    prices = prices.iloc[:, : len(tickers)]
                    prices.columns = tickers

        # Clean price data
        prices = prices.dropna()

        # Calculate returns
        returns = np.log(prices / prices.shift(1)).dropna()

        # Get market caps
        market_caps = {}
        for ticker in tickers:
            if ticker in prices.columns:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    market_cap = info.get("marketCap", None)

                    if market_cap and market_cap > 0:
                        market_caps[ticker] = market_cap / 1e9
                    else:
                        # Use fallback
                        if fallback_market_caps and ticker in fallback_market_caps:
                            market_caps[ticker] = fallback_market_caps[ticker]
                        else:
                            market_caps[ticker] = 100  # Default 100B

                except Exception as e:
                    logger.warning(f"Could not get market cap for {ticker}: {e}")
                    if fallback_market_caps and ticker in fallback_market_caps:
                        market_caps[ticker] = fallback_market_caps[ticker]
                    else:
                        market_caps[ticker] = 100

        market_caps = pd.Series(market_caps)

        # Align data
        common_tickers = list(set(prices.columns) & set(market_caps.index))
        prices = prices[common_tickers]
        returns = returns[common_tickers]
        market_caps = market_caps[common_tickers]

        logger.info(f"Successfully loaded data for {len(common_tickers)} assets")
        logger.info(
            f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}"
        )

        return prices, returns, market_caps

    except Exception as e:
        logger.warning(f"Failed to load live data: {e}")
        logger.info("Generating synthetic data as fallback...")

        return generate_synthetic_data(tickers, start_date, end_date)


def calculate_performance_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics

    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
    risk_free_rate : float
        Risk-free rate (annual)

    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    # Convert to annual terms
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)

    # Basic metrics
    metrics = {
        "total_return": (1 + returns).prod() - 1,
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe_ratio": (
            (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        ),
    }

    # Drawdown metrics
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak

    metrics.update(
        {
            "max_drawdown": drawdown.min(),
            "current_drawdown": drawdown.iloc[-1],
        }
    )

    # Risk metrics
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = downside_returns.std() * np.sqrt(252)
        metrics["sortino_ratio"] = (
            (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        )
        metrics["downside_volatility"] = downside_vol
    else:
        metrics["sortino_ratio"] = float("inf")
        metrics["downside_volatility"] = 0

    # VaR metrics
    metrics.update(
        {
            "var_95": returns.quantile(0.05),
            "var_99": returns.quantile(0.01),
            "cvar_95": returns[returns <= returns.quantile(0.05)].mean(),
            "cvar_99": returns[returns <= returns.quantile(0.01)].mean(),
        }
    )

    # Additional metrics
    metrics.update(
        {
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "hit_ratio": (returns > 0).mean(),
            "calmar_ratio": (
                abs(annual_return / metrics["max_drawdown"])
                if metrics["max_drawdown"] != 0
                else 0
            ),
        }
    )

    # Benchmark comparison
    if benchmark_returns is not None:
        benchmark_annual_return = benchmark_returns.mean() * 252
        benchmark_annual_vol = benchmark_returns.std() * np.sqrt(252)

        # Tracking metrics
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = (
            (annual_return - benchmark_annual_return) / tracking_error
            if tracking_error > 0
            else 0
        )

        # Beta calculation
        covariance = np.cov(returns, benchmark_returns)[0, 1] * 252
        benchmark_variance = benchmark_returns.var() * 252
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Alpha calculation
        alpha = annual_return - (
            risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
        )

        metrics.update(
            {
                "alpha": alpha,
                "beta": beta,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "excess_return": annual_return - benchmark_annual_return,
            }
        )

    return metrics


def format_performance_report(metrics: Dict[str, float]) -> str:
    """
    Format performance metrics into a readable report

    Parameters:
    -----------
    metrics : dict
        Performance metrics dictionary

    Returns:
    --------
    report : str
        Formatted performance report
    """
    report = "📊 PERFORMANCE METRICS REPORT\n"
    report += "=" * 50 + "\n\n"

    # Return metrics
    report += "📈 RETURN METRICS:\n"
    report += f"Total Return:        {metrics.get('total_return', 0):.2%}\n"
    report += f"Annualized Return:   {metrics.get('annualized_return', 0):.2%}\n"
    if "excess_return" in metrics:
        report += f"Excess Return:       {metrics.get('excess_return', 0):.2%}\n"
    report += "\n"

    # Risk metrics
    report += "⚖️ RISK METRICS:\n"
    report += f"Annualized Vol:      {metrics.get('annualized_volatility', 0):.2%}\n"
    report += f"Max Drawdown:        {metrics.get('max_drawdown', 0):.2%}\n"
    report += f"Downside Vol:        {metrics.get('downside_volatility', 0):.2%}\n"
    report += f"VaR (95%):          {metrics.get('var_95', 0):.2%}\n"
    report += f"CVaR (95%):         {metrics.get('cvar_95', 0):.2%}\n"
    report += "\n"

    # Risk-adjusted metrics
    report += "⚡ RISK-ADJUSTED METRICS:\n"
    report += f"Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}\n"
    report += f"Sortino Ratio:       {metrics.get('sortino_ratio', 0):.3f}\n"
    report += f"Calmar Ratio:        {metrics.get('calmar_ratio', 0):.3f}\n"
    if "information_ratio" in metrics:
        report += f"Information Ratio:   {metrics.get('information_ratio', 0):.3f}\n"
    report += "\n"

    # Additional metrics
    report += "📊 ADDITIONAL METRICS:\n"
    report += f"Hit Ratio:           {metrics.get('hit_ratio', 0):.2%}\n"
    report += f"Skewness:            {metrics.get('skewness', 0):.3f}\n"
    report += f"Kurtosis:            {metrics.get('kurtosis', 0):.3f}\n"
    if "alpha" in metrics:
        report += f"Alpha:               {metrics.get('alpha', 0):.2%}\n"
        report += f"Beta:                {metrics.get('beta', 0):.3f}\n"

    return report
