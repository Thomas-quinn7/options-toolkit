"""Repo-root conftest: lets ``pytest`` run against a plain clone.

With ``pip install -e .`` the packages are importable anyway; this only exists
so the test suite (and CI) also works straight out of ``git clone`` with no
install step.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
