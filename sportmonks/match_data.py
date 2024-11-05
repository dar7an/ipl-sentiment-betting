import requests
import json
import os
import time
from pathlib import Path

API_TOKEN = os.getenv("SPORTMONKS_API_TOKEN")
if not API_TOKEN:
    raise ValueError("Please set SPORTMONKS_API_TOKEN environment variable")

BASE_URL = "https://cricket.sportmonks.com/api/v2.0"


def fetch_match_data(fixture_id):
    url = f"{BASE_URL}/fixtures/{fixture_id}"
    params = {"api_token": API_TOKEN, "include": "balls"}

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def main():
    # Read fixture IDs
    with open(
        "sportmonks/fixture_ids.txt",
        "r",
    ) as f:
        fixture_ids = [line.strip() for line in f if line.strip()]

    # Fetch and save data for each fixture
    for fixture_id in fixture_ids:
        output_file = Path(f"sportmonks/match_data/{fixture_id}.json")

        if output_file.exists():
            print(f"Skipping {fixture_id} - already exists")
            continue

        try:
            print(f"Fetching data for fixture {fixture_id}")
            data = fetch_match_data(fixture_id)

            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Saved data for fixture {fixture_id}")
            time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"Error fetching fixture {fixture_id}: {e}")


if __name__ == "__main__":
    main()
