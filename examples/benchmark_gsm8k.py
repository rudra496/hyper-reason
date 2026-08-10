"""
GSM8K Benchmark Evaluation Script for HyperReason Engine (v2)
Executes automated test-time compute scaling benchmarks across GSM8K word problems.
"""

import sys
import os

# Delegate to the v2 benchmark runner in eval/gsm8k_mini.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.gsm8k_mini import main

if __name__ == "__main__":
    main()
