import json
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path


def create_chunks(
    match_id: str,
    odds_data: List[Dict],
    ball_by_ball: Dict,
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
        comment_time = datetime.strptime(
            comment["timestamp"].replace("+00:00", "Z"), "%Y-%m-%dT%H:%M:%SZ"
        )
        if comment_time < match_start_time:
            chunks["pre_game"]["comments"].append(comment)

    # Identify the end of the first innings and start of the second innings
    first_innings_end_time = None
    second_innings_start_time = None

    for ball in ball_by_ball["balls"]:
        if ball.get("innings") == 1 and ball.get("ball") == 6.0:  # Last ball of an over
            first_innings_end_time = datetime.strptime(
                ball["updated_at"].replace(".000000", ""), "%Y-%m-%dT%H:%M:%SZ"
            )
        if ball.get("innings") == 2 and not second_innings_start_time:
            second_innings_start_time = datetime.strptime(
                ball["updated_at"].replace(".000000", ""), "%Y-%m-%dT%H:%M:%SZ"
            )
            break

    # Add a break chunk if both times are found
    if first_innings_end_time and second_innings_start_time:
        chunks["break"] = {
            "start_time": first_innings_end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": second_innings_start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "comments": [],
            "ball_by_ball": [],
        }

        # Add comments during the break
        for comment in reddit_comments:
            comment_time = datetime.strptime(
                comment["timestamp"].replace("+00:00", "Z"), "%Y-%m-%dT%H:%M:%SZ"
            )
            if first_innings_end_time <= comment_time < second_innings_start_time:
                chunks["break"]["comments"].append(comment)

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
        }

        # Add comments within this time window
        for comment in reddit_comments:
            comment_time = datetime.strptime(
                comment["timestamp"].replace("+00:00", "Z"), "%Y-%m-%dT%H:%M:%SZ"
            )
            if chunk_start <= comment_time < chunk_end:
                chunks[chunk_id]["comments"].append(comment)

        # Add ball by ball data within this time window
        for ball in ball_by_ball["balls"]:
            if "updated_at" in ball:
                ball_time = datetime.strptime(
                    ball["updated_at"].replace(".000000", ""), "%Y-%m-%dT%H:%M:%SZ"
                )
                if chunk_start <= ball_time < chunk_end:
                    chunks[chunk_id]["ball_by_ball"].append(ball)
            elif "forecast_data" in ball:
                if chunks[chunk_id]["ball_by_ball"]:
                    last_ball_time = chunks[chunk_id]["ball_by_ball"][-1]["updated_at"]
                    ball["forecast_data"]["timestamp"] = last_ball_time
                chunks[chunk_id]["ball_by_ball"].append(ball)
                chunks[chunk_id]["forecast"] = ball["forecast_data"]

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
        break  # Remove this line to process all matches


if __name__ == "__main__":
    main()
