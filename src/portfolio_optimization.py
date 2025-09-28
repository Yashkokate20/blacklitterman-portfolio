"""
Portfolio Optimization Module

This module contains portfolio optimization functions including
unconstrained analytical solutions and constrained numerical optimization.
"""

import warnings
from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import linalg


class PortfolioOptimizer:
    """
    Portfolio optimization class supporting multiple optimization approaches
    """

    def __init__(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame):
        """
        Initialize portfolio optimizer

        Parameters:
        -----------
        expected_returns : pd.Series
            Expected returns for each asset
        covariance_matrix : pd.DataFrame
            Covariance matrix of asset returns
        """
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.assets = expected_returns.index.tolist()
        self.n_assets = len(self.assets)

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input data"""
        if not self.expected_returns.index.equals(self.covariance_matrix.index):
            raise ValueError(
                "Expected returns and covariance matrix indices must match"
            )

        if not self.covariance_matrix.index.equals(self.covariance_matrix.columns):
            raise ValueError("Covariance matrix must be square")

        # Check positive definiteness
        eigenvals = np.linalg.eigvals(self.covariance_matrix.values)
        if np.min(eigenvals) <= 1e-8:
            warnings.warn("Covariance matrix is not positive definite")

    def optimize_unconstrained(self, risk_aversion: float = 3.0) -> pd.Series:
        """
        Unconstrained mean-variance optimization

        Formula: w* = (1/δ) Σ⁻¹ μ

        Parameters:
        -----------
        risk_aversion : float
            Risk aversion parameter

        Returns:
        --------
        weights : pd.Series
            Optimal portfolio weights
        """
        try:
            # Compute optimal weights
            inv_cov = linalg.inv(self.covariance_matrix.values)
            optimal_weights = (1 / risk_aversion) * np.dot(
                inv_cov, self.expected_returns.values
            )

            return pd.Series(optimal_weights, index=self.assets)

        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            warnings.warn("Using pseudo-inverse due to singular covariance matrix")
            pinv_cov = linalg.pinv(self.covariance_matrix.values)
            optimal_weights = (1 / risk_aversion) * np.dot(
                pinv_cov, self.expected_returns.values
            )
            return pd.Series(optimal_weights, index=self.assets)

    def optimize_constrained(
        self,
        constraints: Dict = None,
        risk_aversion: float = 3.0,
        target_return: Optional[float] = None,
    ) -> Tuple[pd.Series, Dict]:
        """
        Constrained portfolio optimization using CVXPY

        Parameters:
        -----------
        constraints : dict
            Dictionary of constraints:
            - 'long_only': bool (default True)
            - 'max_weight': float (default 0.4)
            - 'min_weight': float (default 0.0)
            - 'turnover': float (optional turnover constraint)
            - 'sector_limits': dict (sector exposure limits)
        risk_aversion : float
            Risk aversion parameter
        target_return : float, optional
            Target portfolio return (for efficient frontier)

        Returns:
        --------
        weights : pd.Series
            Optimal portfolio weights
        info : dict
            Optimization information
        """
        # Default constraints
        if constraints is None:
            constraints = {"long_only": True, "max_weight": 0.4}

        # Decision variable
        w = cp.Variable(self.n_assets)

        # Objective function: maximize utility = μ'w - (δ/2) w'Σw
        portfolio_return = self.expected_returns.values.T @ w
        portfolio_risk = cp.quad_form(w, self.covariance_matrix.values)

        if target_return is not None:
            # Minimize risk for target return
            objective = cp.Minimize(portfolio_risk)
        else:
            # Maximize utility
            utility = portfolio_return - (risk_aversion / 2) * portfolio_risk
            objective = cp.Maximize(utility)

        # Constraints
        constraint_list = []

        # Budget constraint
        constraint_list.append(cp.sum(w) == 1.0)

        # Target return constraint
        if target_return is not None:
            constraint_list.append(portfolio_return >= target_return)

        # Long-only constraint
        if constraints.get("long_only", True):
            constraint_list.append(w >= 0)

        # Weight bounds
        if "max_weight" in constraints:
            constraint_list.append(w <= constraints["max_weight"])

        if "min_weight" in constraints:
            constraint_list.append(w >= constraints["min_weight"])

        # Additional constraints can be added here
        # (sector limits, turnover, etc.)

        # Solve optimization problem
        problem = cp.Problem(objective, constraint_list)

        try:
            # Try different solvers in order of preference
            solvers_to_try = [cp.CLARABEL, cp.OSQP, cp.SCS, cp.CVXOPT]

            for solver in solvers_to_try:
                try:
                    if solver.is_installed():
                        problem.solve(solver=solver, verbose=False)
                        if problem.status in ["optimal", "optimal_inaccurate"]:
                            break
                except:
                    continue
            else:
                # If no specific solver works, try default
                problem.solve(verbose=False)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Optimization failed with status: {problem.status}")

            optimal_weights = pd.Series(w.value, index=self.assets)

            # Compute portfolio statistics
            port_return = (optimal_weights * self.expected_returns).sum()
            port_risk = np.sqrt(
                np.dot(optimal_weights, np.dot(self.covariance_matrix, optimal_weights))
            )

            info = {
                "status": problem.status,
                "portfolio_return": port_return,
                "portfolio_risk": port_risk,
                "sharpe_ratio": port_return / port_risk if port_risk > 0 else 0,
                "objective_value": problem.value,
            }

            return optimal_weights, info

        except Exception as e:
            raise ValueError(f"Optimization failed: {e}")

    def efficient_frontier(
        self, n_points: int = 50, constraints: Dict = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute efficient frontier

        Parameters:
        -----------
        n_points : int
            Number of points on the frontier
        constraints : dict
            Portfolio constraints

        Returns:
        --------
        returns : np.ndarray
            Portfolio returns
        risks : np.ndarray
            Portfolio risks (volatilities)
        weights : np.ndarray
            Portfolio weights (n_points x n_assets)
        """
        # Get return range
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()

        target_returns = np.linspace(min_return, max_return, n_points)

        returns = []
        risks = []
        weights_list = []

        for target_ret in target_returns:
            try:
                weights, info = self.optimize_constrained(
                    constraints=constraints, target_return=target_ret
                )

                returns.append(info["portfolio_return"])
                risks.append(info["portfolio_risk"])
                weights_list.append(weights.values)

            except:
                # Skip infeasible points
                continue

        return np.array(returns), np.array(risks), np.array(weights_list)

    def compute_portfolio_stats(self, weights: pd.Series) -> Dict:
        """
        Compute portfolio statistics for given weights

        Parameters:
        -----------
        weights : pd.Series
            Portfolio weights

        Returns:
        --------
        stats : dict
            Portfolio statistics
        """
        # Align weights with assets
        aligned_weights = weights.reindex(self.assets).fillna(0)

        # Portfolio return and risk
        portfolio_return = (aligned_weights * self.expected_returns).sum()
        portfolio_variance = np.dot(
            aligned_weights, np.dot(self.covariance_matrix, aligned_weights)
        )
        portfolio_risk = np.sqrt(portfolio_variance)

        # Additional metrics
        stats = {
            "return": portfolio_return,
            "volatility": portfolio_risk,
            "sharpe_ratio": (
                portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            ),
            "weights_sum": aligned_weights.sum(),
            "max_weight": aligned_weights.max(),
            "min_weight": aligned_weights.min(),
            "n_nonzero": (aligned_weights.abs() > 1e-6).sum(),
            "concentration": (aligned_weights**2).sum(),  # Herfindahl index
        }

        return stats
