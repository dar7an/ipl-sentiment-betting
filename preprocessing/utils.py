import json
import pandas as pd
from typing import Any, Dict, List

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The content of the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return {}

def load_reddit_data(file_path: str) -> pd.DataFrame:
    """
    Load reddit comments from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame with the reddit comments.
    """
    try:
        return pd.read_csv(file_path, parse_dates=["Timestamp"])
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

def save_chunks(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data chunks to a JSON file.

    Args:
        data (Dict[str, Any]): The data to save.
        file_path (str): The path to the output JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving file {file_path}: {e}") 