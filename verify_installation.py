#!/usr/bin/env python3
"""
Quick verification script for Black-Litterman installation
"""

def check_installation():
    print("🎯 Black-Litterman Installation Verification")
    print("=" * 50)
    
    # Check core packages
    packages_to_check = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'cvxpy', 
        'yfinance', 'plotly', 'streamlit', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package in packages_to_check:
        try:
            if package == 'sklearn':
                import sklearn
                installed_packages.append(f"✅ scikit-learn {sklearn.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                installed_packages.append(f"✅ {package} {version}")
        except ImportError:
            missing_packages.append(f"❌ {package}")
    
    # Print results
    print("\n📦 Package Status:")
    for pkg in installed_packages:
        print(pkg)
    
    if missing_packages:
        print("\n⚠️  Missing packages:")
        for pkg in missing_packages:
            print(pkg)
        print("\n💡 To install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    
    # Check CVXPY solvers
    print("\n🔧 CVXPY Solver Status:")
    try:
        import cvxpy as cp
        available_solvers = []
        
        installed_solvers = cp.installed_solvers()
        for solver_name in ['CLARABEL', 'OSQP', 'SCS', 'CVXOPT', 'SCIPY']:
            if solver_name in installed_solvers:
                available_solvers.append(solver_name)
                print(f"✅ {solver_name} solver available")
            else:
                print(f"⚠️  {solver_name} solver not available")
        
        if not available_solvers:
            print("\n❌ No optimization solvers available!")
            print("💡 Install a solver: pip install clarabel")
            return False
        else:
            print(f"\n🎉 {len(available_solvers)} solver(s) ready!")
    
    except ImportError:
        print("❌ CVXPY not available")
        return False
    
    print("\n" + "=" * 50)
    print("🚀 Installation verified! Ready to run:")
    print("   streamlit run streamlit_app.py")
    print("   jupyter notebook black_litterman_notebook.ipynb")
    
    return True

if __name__ == "__main__":
    success = check_installation()
    exit(0 if success else 1)

