# 🤝 Contributing to Black-Litterman Portfolio Optimization

Thank you for your interest in contributing to this project! This guide will help you get started with contributing to our Black-Litterman portfolio optimization implementation.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Contribution Workflow](#contribution-workflow)
- [Types of Contributions](#types-of-contributions)
- [Documentation Standards](#documentation-standards)
- [Performance Considerations](#performance-considerations)

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of portfolio optimization and Python

### Quick Start

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/blacklitterman-portfolio.git
   cd blacklitterman-portfolio
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Verify setup**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Check code style
   flake8 src/ tests/
   black --check src/ tests/
   ```

## 🛠️ Development Setup

### Development Dependencies

Create `requirements-dev.txt`:
```
pytest>=7.4.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.0.0
pre-commit>=3.0.0
jupyter>=1.0.0
memory-profiler>=0.61.0
```

### Pre-commit Hooks

Set up pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually (optional)
pre-commit run --all-files
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=127, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

## 📝 Code Style Guidelines

### Python Code Style

We follow **PEP 8** with some modifications:

- **Line length**: 127 characters (to match GitHub's display width)
- **String quotes**: Use double quotes for strings, single quotes for string literals in code
- **Import organization**: Use `isort` with the following configuration

#### `.isort.cfg`:
```ini
[settings]
profile = black
multi_line_output = 3
line_length = 127
known_first_party = src
known_third_party = numpy,pandas,scipy,matplotlib,plotly,streamlit,yfinance,cvxpy
```

#### Example Code Style

```python
"""
Module docstring with clear description.

This module implements the Black-Litterman portfolio optimization model
with support for investor views and various constraint types.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union

from src.utils import validate_matrix_properties


class BlackLittermanModel:
    """
    Black-Litterman portfolio optimization model.
    
    This class implements the Black-Litterman model for portfolio optimization,
    combining market equilibrium assumptions with investor views.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns data with assets as columns
    market_weights : pd.Series
        Market capitalization weights
    risk_aversion : float, optional
        Risk aversion parameter (default: estimated from data)
    tau : float, optional
        Uncertainty scaling parameter (default: 0.05)
        
    Attributes
    ----------
    n_assets : int
        Number of assets in the portfolio
    implied_returns : pd.Series
        Market-implied equilibrium returns
        
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> returns = pd.DataFrame(np.random.randn(252, 3), columns=['A', 'B', 'C'])
    >>> market_caps = pd.Series([100, 200, 150], index=['A', 'B', 'C'])
    >>> bl_model = BlackLittermanModel(returns, market_caps)
    >>> bl_model.set_views(P_matrix, Q_vector)
    >>> posterior_returns, posterior_cov = bl_model.compute_posterior()
    """
    
    def __init__(
        self, 
        returns: pd.DataFrame,
        market_weights: pd.Series,
        risk_aversion: Optional[float] = None,
        tau: float = 0.05
    ) -> None:
        # Implementation here
        pass
    
    def compute_posterior(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute Black-Litterman posterior returns and covariance.
        
        Returns
        -------
        posterior_returns : pd.Series
            Updated expected returns incorporating views
        posterior_cov : pd.DataFrame
            Updated covariance matrix
            
        Raises
        ------
        ValueError
            If views are not properly set or matrix operations fail
        """
        # Implementation here
        pass
```

### Jupyter Notebook Style

- Use descriptive cell headers with markdown
- Include explanatory text between code cells
- Clear outputs before committing (use `jupyter nbconvert --clear-output`)
- Use consistent variable naming throughout

### Streamlit App Style

- Use semantic section headers
- Include helpful tooltips and explanations
- Implement proper error handling with user-friendly messages
- Cache expensive operations appropriately

## 🧪 Testing Requirements

### Test Coverage Standards

- **Minimum coverage**: 90% overall, 95% for core modules
- **Test types**: Unit tests, integration tests, property-based tests
- **Edge cases**: Handle boundary conditions and error cases

### Writing Tests

#### Unit Test Example

```python
import pytest
import numpy as np
import pandas as pd
from src.black_litterman import BlackLittermanModel


class TestBlackLittermanModel:
    
    @pytest.fixture
    def sample_data(self):
        """Create consistent test data."""
        np.random.seed(42)  # For reproducibility
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.02,
            columns=tickers
        )
        market_caps = pd.Series([1000, 800, 600], index=tickers)
        return returns, market_caps
    
    def test_initialization(self, sample_data):
        """Test proper model initialization."""
        returns, market_caps = sample_data
        
        bl_model = BlackLittermanModel(returns, market_caps)
        
        assert bl_model.n_assets == 3
        assert bl_model.assets == ['AAPL', 'MSFT', 'GOOGL']
        assert abs(bl_model.market_weights.sum() - 1.0) < 1e-6
    
    def test_views_validation(self, sample_data):
        """Test view matrix validation."""
        returns, market_caps = sample_data
        bl_model = BlackLittermanModel(returns, market_caps)
        
        # Valid views should work
        P = np.zeros((1, 3))
        P[0, 0] = 1
        P[0, 1] = -1
        Q = np.array([0.05])
        
        bl_model.set_views(P, Q)
        assert bl_model.P.shape == (1, 3)
        assert bl_model.Q.shape == (1,)
        
        # Invalid dimensions should raise error
        with pytest.raises(ValueError):
            bl_model.set_views(P, np.array([0.05, 0.03]))  # Wrong Q size
    
    @pytest.mark.parametrize("tau,expected_range", [
        (0.01, (0.5, 2.0)),
        (0.05, (0.8, 1.5)),
        (0.1, (0.9, 1.2))
    ])
    def test_tau_sensitivity(self, sample_data, tau, expected_range):
        """Test model sensitivity to tau parameter."""
        returns, market_caps = sample_data
        bl_model = BlackLittermanModel(returns, market_caps, tau=tau)
        
        # Set standard view
        P = np.zeros((1, 3))
        P[0, 0] = 1
        Q = np.array([0.1])
        bl_model.set_views(P, Q)
        
        posterior_returns, _ = bl_model.compute_posterior()
        
        # Check that results are within expected range
        annual_returns = posterior_returns * 252
        assert expected_range[0] <= annual_returns.max() <= expected_range[1]
```

#### Property-Based Testing Example

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp

@given(
    n_assets=st.integers(min_value=3, max_value=10),
    returns_data=hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=2, max_dims=2),
        elements=st.floats(min_value=-0.1, max_value=0.1)
    )
)
def test_portfolio_weights_properties(n_assets, returns_data):
    """Test that portfolio weights have expected properties regardless of input."""
    assume(returns_data.shape[1] == n_assets)
    assume(returns_data.shape[0] >= 50)  # Minimum observations
    
    # Create test data
    tickers = [f'ASSET_{i}' for i in range(n_assets)]
    returns = pd.DataFrame(returns_data, columns=tickers)
    market_caps = pd.Series(np.random.lognormal(5, 1, n_assets), index=tickers)
    
    # Test Black-Litterman model
    bl_model = BlackLittermanModel(returns, market_caps)
    optimizer = PortfolioOptimizer(
        bl_model.implied_returns * 252,
        bl_model.sample_cov * 252
    )
    
    weights = optimizer.optimize_unconstrained(risk_aversion=3.0)
    
    # Properties that should always hold
    assert len(weights) == n_assets
    assert not np.isnan(weights).any()
    assert not np.isinf(weights).any()
    # Note: unconstrained weights don't need to sum to 1 or be positive
```

### Performance Testing

```python
import time
import pytest


@pytest.mark.performance
def test_optimization_speed():
    """Test that optimization completes within reasonable time."""
    # Setup large portfolio
    n_assets = 50
    returns = generate_test_returns(n_assets, 252)
    market_caps = generate_test_market_caps(n_assets)
    
    # Time the computation
    start_time = time.time()
    
    bl_model = BlackLittermanModel(returns, market_caps)
    optimizer = PortfolioOptimizer(
        bl_model.implied_returns * 252,
        bl_model.sample_cov * 252
    )
    weights = optimizer.optimize_constrained(
        constraints={'long_only': True},
        risk_aversion=3.0
    )
    
    elapsed_time = time.time() - start_time
    
    # Performance requirements
    assert elapsed_time < 10.0, f"Optimization too slow: {elapsed_time:.2f}s"
    assert len(weights) == n_assets
```

## 🔄 Contribution Workflow

### 1. Issue Creation

Before starting work:

- **Check existing issues** to avoid duplication
- **Create detailed issue** with:
  - Clear description of problem/feature
  - Expected behavior
  - Current behavior (for bugs)
  - Steps to reproduce (for bugs)
  - Proposed solution (for enhancements)

### 2. Branch Strategy

```bash
# Create feature branch
git checkout -b feature/descriptive-name

# Or for bug fixes
git checkout -b fix/issue-number-description

# Or for documentation
git checkout -b docs/topic-description
```

### 3. Development Process

1. **Write tests first** (TDD approach)
2. **Implement feature/fix**
3. **Update documentation**
4. **Run full test suite**
5. **Check code style**

```bash
# Full development check
pytest tests/ --cov=src --cov-report=term-missing
flake8 src/ tests/
black --check src/ tests/
isort --check-only src/ tests/
```

### 4. Commit Guidelines

Use conventional commits:

```bash
# Feature commits
git commit -m "feat: add support for custom view confidence matrices"

# Bug fix commits
git commit -m "fix: handle singular covariance matrices in optimization"

# Documentation commits
git commit -m "docs: update API documentation for portfolio constraints"

# Test commits
git commit -m "test: add integration tests for backtesting engine"

# Refactor commits
git commit -m "refactor: simplify view matrix validation logic"
```

### 5. Pull Request Process

1. **Update your branch** with latest main
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create pull request** with:
   - **Clear title** and description
   - **Link to related issues**
   - **Screenshots** (for UI changes)
   - **Performance impact** (if applicable)
   - **Breaking changes** (if any)

3. **Address review feedback**
4. **Ensure CI passes**
5. **Squash commits** if requested

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)

## Screenshots (if applicable)

## Additional Notes
```

## 🎯 Types of Contributions

### 🐛 Bug Reports

Include:
- **Python version** and OS
- **Package versions** (`pip list`)
- **Minimal reproducible example**
- **Expected vs actual behavior**
- **Error messages** (full traceback)

### ✨ Feature Requests

Include:
- **Use case description**
- **Proposed API** (if applicable)
- **Alternative solutions considered**
- **Willingness to implement**

### 📚 Documentation Improvements

- **API documentation** (docstrings)
- **User guides** and tutorials
- **Code examples**
- **README updates**
- **Mathematical explanations**

### 🔧 Code Contributions

Priority areas:
- **Performance optimizations**
- **Additional optimization constraints**
- **New portfolio construction methods**
- **Enhanced risk models**
- **Better visualization options**
- **Additional data sources**

## 📖 Documentation Standards

### Docstring Format (NumPy Style)

```python
def optimize_constrained(
    self,
    constraints: Optional[Dict[str, Union[bool, float]]] = None,
    target_return: Optional[float] = None,
    risk_aversion: float = 3.0
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Optimize portfolio with constraints using convex optimization.
    
    Solves the mean-variance optimization problem subject to various
    constraints such as long-only, maximum weights, and target returns.
    
    Parameters
    ----------
    constraints : dict, optional
        Dictionary of constraints with keys:
        - 'long_only' : bool, default True
            If True, prevent short selling (all weights >= 0)
        - 'max_weight' : float, default None
            Maximum weight for any single asset
        - 'min_weight' : float, default None
            Minimum weight for any single asset
    target_return : float, optional
        Target portfolio return. If specified, minimizes risk subject
        to achieving this return level.
    risk_aversion : float, default 3.0
        Risk aversion parameter. Higher values prefer lower risk.
        Only used when target_return is None.
    
    Returns
    -------
    weights : pd.Series
        Optimal portfolio weights with asset names as index
    info : dict
        Dictionary containing optimization results:
        - 'status' : str
            Optimization status ('optimal', 'infeasible', etc.)
        - 'portfolio_return' : float
            Expected portfolio return (annualized)
        - 'portfolio_risk' : float
            Portfolio volatility (annualized)
        - 'sharpe_ratio' : float
            Risk-adjusted return measure
    
    Raises
    ------
    ValueError
        If constraints are infeasible or optimization fails
    LinAlgError
        If covariance matrix is singular or poorly conditioned
        
    Examples
    --------
    >>> optimizer = PortfolioOptimizer(expected_returns, cov_matrix)
    >>> weights, info = optimizer.optimize_constrained(
    ...     constraints={'long_only': True, 'max_weight': 0.3},
    ...     risk_aversion=2.5
    ... )
    >>> print(f"Optimal Sharpe ratio: {info['sharpe_ratio']:.3f}")
    
    Notes
    -----
    The optimization problem solved is:
    
    .. math::
        \\min_w \\frac{1}{2} w^T \\Sigma w - \\frac{1}{\\delta} \\mu^T w
        
    subject to the specified constraints, where :math:`w` are the portfolio
    weights, :math:`\\Sigma` is the covariance matrix, :math:`\\mu` are the
    expected returns, and :math:`\\delta` is the risk aversion parameter.
    """
```

### README Sections

Ensure README includes:
- **Clear project description**
- **Quick start guide**
- **Installation instructions**
- **Basic usage examples**
- **API documentation links**
- **Contributing guidelines**
- **License information**

## ⚡ Performance Considerations

### Optimization Guidelines

1. **Use vectorized operations** (NumPy/Pandas)
2. **Cache expensive computations**
3. **Profile before optimizing**
4. **Consider memory usage** for large portfolios
5. **Use appropriate data types** (float64 vs float32)

### Memory Management

```python
# Good: Use generators for large datasets
def generate_scenarios(n_scenarios: int):
    for i in range(n_scenarios):
        yield create_scenario(i)

# Bad: Load everything into memory
scenarios = [create_scenario(i) for i in range(n_scenarios)]
```

### Profiling Tools

```bash
# Time profiling
python -m cProfile -o profile.stats script.py

# Memory profiling
python -m memory_profiler script.py

# Line profiler
kernprof -l -v script.py
```

## 🔒 Security Considerations

- **No hardcoded secrets** or API keys
- **Validate all inputs** especially from external sources
- **Use secure random number generation** for sensitive operations
- **Be careful with pickle/eval** - prefer JSON for data serialization
- **Update dependencies** regularly for security patches

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: All contributions are reviewed by maintainers
- **Documentation**: Check existing docs before asking questions

## 🙏 Recognition

Contributors will be:
- **Listed in CONTRIBUTORS.md**
- **Mentioned in release notes**
- **Credited in academic citations** (if applicable)

Thank you for contributing to make this project better! 🚀
