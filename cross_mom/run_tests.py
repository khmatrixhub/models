#!/usr/bin/env python3
"""
Test runner for cross_mom tests using pytest.

Runs all test_*.py files in the tests/ directory using pytest framework.
"""

import sys
import subprocess
from pathlib import Path


def main():
    # Get the project root (where pytest.ini is located)
    project_root = Path(__file__).parent.parent
    
    print("="*80)
    print("Running tests with pytest")
    print("="*80)
    
    # Run pytest from project root
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "cross_mom/tests/", "-v"],
        cwd=project_root
    )
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())