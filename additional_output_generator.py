#!/usr/bin/env python3
"""
Additional Output Generator for QCentroid Solvers
Generates rich HTML visualizations and CSV exports in additional_output/ folder.
Platform picks up files from this folder and displays them in the job detail view.

All HTML is self-contained (inline CSS/SVG) — no external dependencies needed.
"""

import os
import json
import csv
import io
import math
from typing import Dict, List, Any


def _safe_get(obj, key, default=None):
    """Safely get a key from an object, returning default if obj is not a dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default
