"""
Utilities for AI Data Science.

This package provides utility functions for working with data, 
logging, text processing, and other common operations.
"""

from app.utils.regex import (
    format_recommended_steps, 
    add_comments_to_top, 
    format_agent_name, 
    get_generic_summary, 
    
    relocate_imports_inside_function,
    remove_consecutive_duplicates,
)

from app.utils.logging import log_ai_function
from app.utils.messages import get_tool_call_names
from app.utils.plotly import plotly_from_dict
from app.utils.html import open_html_file_in_browser
from app.utils.matplotlib import matplotlib_from_base64

__all__ = [
    "format_recommended_steps",
    "add_comments_to_top",
    "format_agent_name",
    "get_generic_summary",
    
    "relocate_imports_inside_function",
    "remove_consecutive_duplicates", 
    "log_ai_function",
    "get_tool_call_names",
    "plotly_from_dict",
    "open_html_file_in_browser",
    "matplotlib_from_base64"
] 