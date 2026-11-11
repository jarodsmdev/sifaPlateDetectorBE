#!/usr/bin/env python3
"""Alias histórico de ``test/evaluate_validation.py``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_validation import main

if __name__ == "__main__":
    raise SystemExit(main())
