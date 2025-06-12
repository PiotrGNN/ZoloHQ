# Root conftest.py for pytest to help with import resolution
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
