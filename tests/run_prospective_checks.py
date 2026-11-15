#!/usr/bin/env python3
"""CPU tests; tiny random local Transformers require optional GPU libraries, not weights."""
from pathlib import Path
import sys
import pytest

if __name__=='__main__':
    raise SystemExit(pytest.main([str(Path(__file__).parent/'prospective'),'-q',*sys.argv[1:]]))
