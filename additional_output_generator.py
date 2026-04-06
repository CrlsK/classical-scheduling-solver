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
