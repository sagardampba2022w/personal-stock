#!/usr/bin/env python3
"""
Simple test runner for prediction comparison
Save this as run_comparison.py and run from data-prep directory
"""

if __name__ == "__main__":
    # Import and run the comparison from your existing file
    from compare_predictions import main
    
    print("Starting prediction comparison...")
    main()
    print("Done!")