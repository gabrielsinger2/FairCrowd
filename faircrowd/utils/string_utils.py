from typing import Any
import re


def title_case(s: Any) -> str:
    """
    Convert snake_case, camelCase, and PascalCase to Title Case
    """
    # Convert classes and functions to string
    s = name(s)
    # Don't convert abbreviations
    if s.upper() == s:
        return s
    return re.sub("([A-Z])", r" \1", s.replace("_", " ")).title().strip()


def name(obj: Any) -> str:
    """
    Get the name of an object or function
    """
    if isinstance(obj, str):
        return obj
    return obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
