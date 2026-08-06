"""Put src/ and preprocessing/ on the import path for tests."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in ('src', 'preprocessing'):
    sys.path.insert(0, os.path.join(ROOT, folder))
