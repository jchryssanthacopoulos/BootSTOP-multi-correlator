import json
from pathlib import Path
from typing import Optional


def load_config_file(config_file: Path) -> Optional[dict]:
    """Load the configuration file containing the spin partition or optimisation config

    Args:
        config_file: Path to the configuration file

    Returns:
        The JSON-like dictionary for the spin partition, or None on error

    """
    if not config_file.is_file():
        print(f"Error: Config file '{config_file}' not found.")
        return None

    try:
        with config_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON in '{config_file}'.\nDetails: {e}")
    except Exception as e:
        print(f"Unexpected error while loading '{config_file}': {e}")
