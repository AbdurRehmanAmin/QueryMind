"""Basic schema definitions for the DS-Agent system."""

import json
import datetime
from typing import List, Dict, Any, Optional

class LLMError(Exception):
    """
    Exception raised when an LLM API call fails.
    """
    pass

class Action:
    """
    Action to be executed by the agent.
    """
    def __init__(self, name, input_dict=None):
        self.name = name
        self.input_dict = input_dict or {}

    def __str__(self):
        return f"Action({self.name}, {self.input_dict})"
        
class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Extended JSON encoder that handles datetime objects.
    """
    def default(self, o):
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        return super().default(o)
