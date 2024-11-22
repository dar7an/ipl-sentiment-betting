import json
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path


def load_json_file(filepath: str) -> Dict:
    with open(filepath, "r") as f:
        return json.load(f)


def save_chunks(chunks: Dict, output_path: str, match_id: str):
    output_file = os.path.join(output_path, f"{match_id}.json")
    with open(output_file, "w") as f:
        json.dump(chunks, f, indent=4)


def load_reddit_data(filepath: str) -> List[Dict]:
    """Load Reddit data from CSV file and convert to list of dicts"""
    df = pd.read_csv(filepath)
    comments = []
    for _, row in df.iterrows():
        comments.append(
            {
                "created_utc": row["Timestamp"],
                "body": row["Comment"],
                "score": row["Upvotes"],
            }
        )
    return comments


def create_chunks(
    match_id: str,
    odds_data: List[Dict],
    ball_by_ball: Dict,  # Changed from List to Dict
    reddit_comments: List[Dict],
) -> Dict:

    # Convert timestamps to datetime objects
    for odd in odds_data:
        odd["timestamp"] = datetime.strptime(odd["last_update"], "%Y-%m-%dT%H:%M:%SZ")

    match_start_time = datetime.strptime(
        ball_by_ball["summary"]["starting_at"].replace(".000000", ""),
        "%Y-%m-%dT%H:%M:%SZ",
    )

    chunks = {
        "match_id": match_id,
        "pre_game": {
            "comments": [],
            "ball_by_ball": [],
            "odds": {
                "timestamp": odds_data[0]["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "odds": odds_data[0]["odds"],
            },
        },
    }

    # Process pre-game data
    for comment in reddit_comments:
        # Handle timezone format in reddit timestamps
        comment_time = datetime.strptime(
            comment["created_utc"].replace("+00:00", "Z"), "%Y-%m-%dT%H:%M:%SZ"
        )
        if comment_time < match_start_time:
            chunks["pre_game"]["comments"].append(comment)

    # Create time-based chunks
    for i in range(1, len(odds_data)):
        chunk_start = odds_data[i - 1]["timestamp"]
        chunk_end = odds_data[i]["timestamp"]
        chunk_id = f"chunk_{i}"

        chunks[chunk_id] = {
            "start_time": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "comments": [],
            "ball_by_ball": [],
            "odds": {
                "timestamp": odds_data[i]["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "odds": odds_data[i]["odds"],
            },
            "forecast": None,  # Initialize forecast as None
        }

        # Add comments within this time window
        for comment in reddit_comments:
            # Handle timezone format in reddit timestamps
            comment_time = datetime.strptime(
                comment["created_utc"].replace("+00:00", "Z"), "%Y-%m-%dT%H:%M:%SZ"
            )
            if chunk_start <= comment_time < chunk_end:
                chunks[chunk_id]["comments"].append(comment)

        # Add ball by ball data within this time window
        for ball in ball_by_ball["balls"]:  # Access balls list from the dictionary
            if "updated_at" in ball:
                ball_time = datetime.strptime(
                    ball["updated_at"].replace(".000000", ""), "%Y-%m-%dT%H:%M:%SZ"
                )
                if chunk_start <= ball_time < chunk_end:
                    chunks[chunk_id]["ball_by_ball"].append(ball)
            elif "forecast_data" in ball:
                # Assign the timestamp of the last ball in the over to the forecast data
                if chunks[chunk_id]["ball_by_ball"]:
                    last_ball_time = chunks[chunk_id]["ball_by_ball"][-1]["updated_at"]
                    ball["forecast_data"]["timestamp"] = last_ball_time
                chunks[chunk_id]["ball_by_ball"].append(ball)
                chunks[chunk_id]["forecast"] = ball[
                    "forecast_data"
                ]  # Include forecast data

    return chunks


def main():
    base_dir = Path("/Users/darshan/Documents/GitHub/ipl-sentiment-betting")
    odds_dir = base_dir / "the_odds_api" / "2024_trimmed"  # Updated path
    ball_by_ball_dir = base_dir / "sportmonks" / "2024_enhanced"  # Updated path
    reddit_dir = base_dir / "reddit" / "2024"  # Updated path
    output_dir = base_dir / "sentiment_analysis" / "processed_data"  # Updated path

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each odds file sequentially in ascending order
    for odds_file in sorted(odds_dir.glob("*.json")):
        match_id = odds_file.stem

        # Load data
        odds_data = load_json_file(str(odds_file))
        ball_by_ball = load_json_file(str(ball_by_ball_dir / f"{match_id}.json"))
        reddit_comments = load_reddit_data(str(reddit_dir / f"{match_id}.csv"))

        # Create chunks
        chunks = create_chunks(match_id, odds_data, ball_by_ball, reddit_comments)

        # Save chunks
        save_chunks(chunks, str(output_dir), match_id)
        print(f"Processed match {match_id}")


if __name__ == "__main__":
    main()
