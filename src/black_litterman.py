"""
Black-Litterman Model Implementation

This module contains the core Black-Litterman model implementation,
including implied equilibrium returns, posterior mean and covariance calculations.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg


class BlackLittermanModel:
    """
    Black-Litterman Portfolio Optimization Model

    This class implements the complete Black-Litterman framework for
    combining market equilibrium with investor views to generate
    improved expected returns and covariance estimates.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_caps: Optional[pd.Series] = None,
        risk_aversion: Optional[float] = None,
        tau: float = 0.05,
    ):
        """
        Initialize Black-Litterman model

        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns data (assets as columns)
        market_caps : pd.Series, optional
            Market capitalizations for computing market weights
        risk_aversion : float, optional
            Risk aversion parameter. If None, will be estimated
        tau : float, default=0.05
            Scaling factor for uncertainty in prior
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        self.tau = tau

        # Compute sample statistics
        self.sample_mean = returns.mean()
        self.sample_cov = returns.cov()

        # Set up market weights
        if market_caps is not None:
            self.market_weights = self._compute_market_weights(market_caps)
        else:
            # Equal weights as fallback
            self.market_weights = pd.Series(1.0 / self.n_assets, index=self.assets)

        # Estimate or set risk aversion
        if risk_aversion is not None:
            self.risk_aversion = risk_aversion
        else:
            self.risk_aversion = self._estimate_risk_aversion()

        # Compute implied equilibrium returns
        self.implied_returns = self._compute_implied_returns()

        # Initialize views (empty by default)
        self.P = None  # Picking matrix
        self.Q = None  # View values
        self.Omega = None  # View uncertainty

    def _compute_market_weights(self, market_caps: pd.Series) -> pd.Series:
        """Compute market capitalization weights"""
        # Align with assets and normalize
        aligned_caps = market_caps.reindex(self.assets).fillna(0)
        return aligned_caps / aligned_caps.sum()

    def _estimate_risk_aversion(self) -> float:
        """
        Estimate risk aversion using historical market data

        Uses the formula: δ = (μ_market - r_f) / σ²_market
        Assumes risk-free rate of 2% annually
        """
        # Convert to annual terms
        annual_returns = self.returns.mean() * 252
        annual_cov = self.sample_cov * 252

        # Market portfolio return and variance
        market_return = (annual_returns * self.market_weights).sum()
        market_variance = np.dot(
            self.market_weights, np.dot(annual_cov, self.market_weights)
        )

        # Risk-free rate assumption (2% annually)
        risk_free_rate = 0.02

        # Risk aversion estimate
        risk_aversion = (market_return - risk_free_rate) / market_variance

        # Reasonable bounds (typically 1-10)
        return max(1.0, min(10.0, risk_aversion))

    def _compute_implied_returns(self) -> pd.Series:
        """
        Compute implied equilibrium returns: π = δ Σ w_mkt
        """
        return pd.Series(
            self.risk_aversion * np.dot(self.sample_cov, self.market_weights),
            index=self.assets,
        )

    def set_views(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: Optional[np.ndarray] = None,
        confidence_level: str = "medium",
    ) -> None:
        """
        Set investor views for the Black-Litterman model

        Parameters:
        -----------
        P : np.ndarray, shape (n_views, n_assets)
            Picking matrix identifying which assets each view relates to
        Q : np.ndarray, shape (n_views,)
            View values (expected returns)
        Omega : np.ndarray, optional, shape (n_views, n_views)
            View uncertainty matrix. If None, will be computed based on confidence
        confidence_level : str, default='medium'
            Confidence level for views: 'low', 'medium', 'high'
        """
        self.P = P
        self.Q = Q

        if Omega is None:
            self.Omega = self._compute_view_uncertainty(P, confidence_level)
        else:
            self.Omega = Omega

    def _compute_view_uncertainty(
        self, P: np.ndarray, confidence_level: str
    ) -> np.ndarray:
        """
        Compute view uncertainty matrix based on confidence level

        Uses the formula: Ω = α * τ * P Σ P'
        where α is a scaling factor based on confidence
        """
        # Confidence scaling factors
        alpha_map = {
            "low": 2.0,  # High uncertainty
            "medium": 1.0,  # Moderate uncertainty
            "high": 0.5,  # Low uncertainty
        }

        alpha = alpha_map.get(confidence_level, 1.0)

        # Compute uncertainty matrix
        omega = alpha * self.tau * np.dot(P, np.dot(self.sample_cov, P.T))

        return omega

    def compute_posterior(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute Black-Litterman posterior mean and covariance

        Returns:
        --------
        posterior_mean : pd.Series
            Black-Litterman posterior expected returns
        posterior_cov : pd.DataFrame
            Black-Litterman posterior covariance matrix
        """
        if self.P is None or self.Q is None or self.Omega is None:
            warnings.warn("No views set. Using implied equilibrium returns.")
            return self.implied_returns, self.sample_cov

        # Convert to numpy arrays for computation
        Sigma = self.sample_cov.values
        pi = self.implied_returns.values

        # Compute precision matrices
        tau_Sigma_inv = linalg.inv(self.tau * Sigma)
        P_T_Omega_inv_P = np.dot(self.P.T, np.dot(linalg.inv(self.Omega), self.P))

        # Posterior precision matrix
        posterior_precision = tau_Sigma_inv + P_T_Omega_inv_P
        posterior_cov_matrix = linalg.inv(posterior_precision)

        # Posterior mean
        term1 = np.dot(tau_Sigma_inv, pi)
        term2 = np.dot(self.P.T, np.dot(linalg.inv(self.Omega), self.Q))
        posterior_mean_vector = np.dot(posterior_cov_matrix, term1 + term2)

        # Convert back to pandas
        posterior_mean = pd.Series(posterior_mean_vector, index=self.assets)
        posterior_cov = pd.DataFrame(
            posterior_cov_matrix, index=self.assets, columns=self.assets
        )

        return posterior_mean, posterior_cov

    def get_model_summary(self) -> dict:
        """Get summary of model parameters and computed values"""
        summary = {
            "n_assets": self.n_assets,
            "risk_aversion": self.risk_aversion,
            "tau": self.tau,
            "market_weights": self.market_weights.to_dict(),
            "implied_returns_annual": (self.implied_returns * 252).to_dict(),
            "has_views": self.P is not None,
        }

        if self.P is not None:
            summary["n_views"] = self.P.shape[0]
            summary["view_values"] = self.Q.tolist()

        return summary
