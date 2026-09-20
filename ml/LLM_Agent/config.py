"""
Configuration management for the AI Financial Intelligence Assistant.
Handles API credentials, model parameters, and runtime security settings.
"""

import os
from typing import Optional
from dataclasses import dataclass, field

def _load_env_file() -> None:
    """
    Discovers and loads .env files from the workspace root or local module directories.
    Uses python-dotenv if installed, with a reliable built-in fallback parser.
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        os.path.abspath(os.path.join(current_file_dir, "..", "..")),  # Workspace root
        os.path.abspath(os.path.join(current_file_dir, "..")),        # ml/ root
        current_file_dir,                                              # LLM_Agent/
        os.getcwd(),                                                   # Active working dir
    ]

    env_paths = [os.path.join(d, ".env") for d in candidate_dirs]

    # Try python-dotenv first if available
    try:
        from dotenv import load_dotenv
        for path in env_paths:
            if os.path.isfile(path):
                load_dotenv(dotenv_path=path, override=False)
    except ImportError:
        pass

    # Built-in fallback parser in case dotenv is not installed
    for path in env_paths:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip("'\"")
                            if key and key not in os.environ:
                                os.environ[key] = val
            except Exception:
                pass

# Ensure .env is loaded upon importing config
_load_env_file()

@dataclass
class AgentConfig:
    """Configuration settings for Groq LLM and AI Orchestrator."""
    groq_api_key: str = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY", "")
    )
    # Primary model: qwen/qwen3.8-27b (state-of-the-art fast reasoning)
    primary_model: str = "qwen/qwen3.8-27b"
    fallback_model: str = "openai/gpt-oss-120b"
    fast_model: str = "openai/gpt-oss-20b"
    
    # Inference parameters
    temperature: float = 0.2  # Low temperature for analytical rigor & low hallucination
    max_completion_tokens: int = 1500
    top_p: float = 0.9
    timeout_seconds: float = 30.0
    max_retries: int = 3
    
    # Currency and localization
    currency_symbol: str = "₹"
    currency_code: str = "INR"
    
    # Path settings
    ds_root_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "DataScience ")
    )
    models_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "DataScience ", "models")
    )
    data_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "DataScience ", "data")
    )

    def validate(self) -> bool:
        """Validates that credentials and essential directories are configured."""
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Please add GROQ_API_KEY to your .env file "
                "or export it as an environment variable."
            )
        if not self.groq_api_key.startswith("gsk_"):
            raise ValueError("Invalid GROQ API Key format configured (expected 'gsk_' prefix).")
        return True
