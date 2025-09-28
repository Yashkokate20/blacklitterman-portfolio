#!/usr/bin/env python3
"""
Installation script for Black-Litterman Portfolio Optimization
Installs required packages and solvers with proper error handling
"""

import subprocess
import sys
import importlib

def install_package(package_name, pip_name=None):
    """Install a package using pip"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} already installed")
        return True
    except ImportError:
        print(f"📦 Installing {pip_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"✅ {pip_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {pip_name}: {e}")
            return False

def main():
    print("🎯 Black-Litterman Portfolio Optimization - Solver Installation")
    print("=" * 60)
    
    # Core packages
    packages = [
        ("numpy", "numpy>=1.24.0"),
        ("pandas", "pandas>=2.0.0"),
        ("scipy", "scipy>=1.10.0"),
        ("sklearn", "scikit-learn>=1.3.0"),
        ("cvxpy", "cvxpy>=1.3.0"),
        ("yfinance", "yfinance>=0.2.0"),
        ("plotly", "plotly>=5.15.0"),
        ("streamlit", "streamlit>=1.25.0"),
        ("matplotlib", "matplotlib>=3.7.0"),
        ("seaborn", "seaborn>=0.12.0"),
        ("tqdm", "tqdm>=4.65.0"),
    ]
    
    # Solver packages (optional but recommended)
    solvers = [
        ("clarabel", "clarabel>=0.5.0"),
    ]
    
    print("\n📦 Installing core packages...")
    failed_packages = []
    
    for package, pip_name in packages:
        if not install_package(package, pip_name):
            failed_packages.append(pip_name)
    
    print("\n🔧 Installing optimization solvers...")
    failed_solvers = []
    
    for package, pip_name in solvers:
        if not install_package(package, pip_name):
            failed_solvers.append(pip_name)
    
    # Test CVXPY solvers
    print("\n🧪 Testing available CVXPY solvers...")
    try:
        import cvxpy as cp
        
        available_solvers = []
        for solver_name in ['CLARABEL', 'OSQP', 'SCS', 'CVXOPT']:
            solver = getattr(cp, solver_name, None)
            if solver and solver.is_installed():
                available_solvers.append(solver_name)
                print(f"✅ {solver_name} solver available")
            else:
                print(f"⚠️  {solver_name} solver not available")
        
        if available_solvers:
            print(f"\n🎉 {len(available_solvers)} solver(s) available: {', '.join(available_solvers)}")
        else:
            print("\n❌ No CVXPY solvers available. Portfolio optimization may not work.")
            print("💡 Try: pip install cvxopt")
    
    except ImportError:
        print("❌ CVXPY not installed correctly")
    
    # Summary
    print("\n" + "=" * 60)
    if failed_packages or failed_solvers:
        print("⚠️  Installation completed with some issues:")
        if failed_packages:
            print(f"   Failed core packages: {', '.join(failed_packages)}")
        if failed_solvers:
            print(f"   Failed solvers: {', '.join(failed_solvers)}")
        print("\n💡 Try installing failed packages manually:")
        for pkg in failed_packages + failed_solvers:
            print(f"   pip install {pkg}")
    else:
        print("🎉 All packages installed successfully!")
    
    print("\n🚀 Ready to run:")
    print("   jupyter notebook black_litterman_notebook.ipynb")
    print("   streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
