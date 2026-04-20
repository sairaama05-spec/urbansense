"""
conftest.py — project root
Ensures the repo root is on sys.path so that
`from models.X import Y` and `from tracking.X import Y`
work in pytest regardless of how it is invoked.
"""
import sys
import os

# Insert repo root at the front of sys.path
sys.path.insert(0, os.path.dirname(__file__))
