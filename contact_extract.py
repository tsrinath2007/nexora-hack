"""
Contact Information Extraction Module.

Provides utilities for extracting candidate contact details (such as email addresses)
from raw resume text using robust regular expressions.
"""

from typing import Optional
from resume_quality import extract_email, _EMAIL_PATTERN, _PHONE_PATTERN

__all__ = ["extract_email"]
