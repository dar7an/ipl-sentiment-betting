import json
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path

from preprocessing.utils import load_json_file, load_reddit_data, save_chunks

# Constants for datetime formats
TIMESTAMP_FORMAT_Z = "%Y-%m-%dT%H:%M:%SZ"
TIMESTAMP_FORMAT_NO_Z = "%Y-%m-%dT%H:%M:%S"


def _process_pre_game_data(chunks, reddit_comments, match_start_time):
    for comment in reddit_comments:
        comment_time = comment["Timestamp"].to_pydatetime().replace(tzinfo=None)
        if comment_time < match_start_time:
            chunks["pre_game"]["comments"].append(comment)


def _find_innings_break(ball_by_ball):
    first_innings_end_time = None
    second_innings_start_time = None
    for ball in ball_by_ball["balls"]:
        if ball.get("innings") == 1 and ball.get("ball") == 6.0:  # Last ball of an over
            first_innings_end_time = datetime.strptime(
                ball["updated_at"].replace(".000000", ""), TIMESTAMP_FORMAT_Z
            )
        if ball.get("innings") == 2 and not second_innings_start_time:
            second_innings_start_time = datetime.strptime(
                ball["updated_at"].replace(".000000", ""), TIMESTAMP_FORMAT_Z
            )
            break
    return first_innings_end_time, second_innings_start_time


def _process_break_data(chunks, reddit_comments, first_innings_end_time, second_innings_start_time):
    if first_innings_end_time and second_innings_start_time:
        chunks["break"] = {
            "start_time": first_innings_end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": second_innings_start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "comments": [],
            "ball_by_ball": [],
        }

        for comment in reddit_comments:
            comment_time = comment["Timestamp"].to_pydatetime().replace(tzinfo=None)
            if first_innings_end_time <= comment_time < second_innings_start_time:
                chunks["break"]["comments"].append(comment)


def _process_time_based_chunks(chunks, odds_data, reddit_comments, ball_by_ball):
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

        for comment in reddit_comments:
            comment_time = comment["Timestamp"].to_pydatetime().replace(tzinfo=None)
            if chunk_start <= comment_time < chunk_end:
                chunks[chunk_id]["comments"].append(comment)

        for ball in ball_by_ball["balls"]:
            if "updated_at" in ball:
                ball_time = datetime.strptime(
                    ball["updated_at"].replace(".000000", ""), TIMESTAMP_FORMAT_Z
                )
                if chunk_start <= ball_time < chunk_end:
                    chunks[chunk_id]["ball_by_ball"].append(ball)
            elif "forecast_data" in ball:
                if chunks[chunk_id]["ball_by_ball"]:
                    last_ball_time = chunks[chunk_id]["ball_by_ball"][-1]["updated_at"]
                    ball["forecast_data"]["timestamp"] = last_ball_time
                chunks[chunk_id]["ball_by_ball"].append(ball)
                chunks[chunk_id]["forecast"] = ball["forecast_data"]


def create_chunks(
    match_id: str,
    odds_data: List[Dict],
    ball_by_ball: Dict,
    reddit_comments: List[Dict],
) -> Dict:
    # Convert timestamps to datetime objects
    for odd in odds_data:
        odd["timestamp"] = datetime.strptime(odd["last_update"], TIMESTAMP_FORMAT_Z)

    match_start_time = datetime.strptime(
        ball_by_ball["summary"]["starting_at"].replace(".000000", ""),
        TIMESTAMP_FORMAT_Z,
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

    _process_pre_game_data(chunks, reddit_comments, match_start_time)

    first_innings_end_time, second_innings_start_time = _find_innings_break(
        ball_by_ball
    )

    _process_break_data(
        chunks, reddit_comments, first_innings_end_time, second_innings_start_time
    )

    _process_time_based_chunks(chunks, odds_data, reddit_comments, ball_by_ball)

    return chunks


def main():
    base_dir = Path(".")
    odds_dir = base_dir / "preprocessing" / "the_odds_api" / "2024_trimmed"
    ball_by_ball_dir = base_dir / "preprocessing" / "sportmonks" / "2024_enhanced"
    reddit_dir = base_dir / "preprocessing" / "reddit" / "2024"
    output_dir = base_dir / "preprocessing" / "sentiment_analysis" / "processed_data"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each odds file sequentially in ascending order
    for odds_file in sorted(odds_dir.glob("*.json")):
        match_id = odds_file.stem
        print(f"Processing match {match_id}...")

        # Load data
        odds_data = load_json_file(str(odds_file))
        ball_by_ball_file = ball_by_ball_dir / f"{match_id}.json"
        ball_by_ball = load_json_file(str(ball_by_ball_file))
        reddit_comments_file = reddit_dir / f"{match_id}.csv"
        reddit_comments_df = load_reddit_data(str(reddit_comments_file))

        if not odds_data or not ball_by_ball or reddit_comments_df.empty:
            print(f"Skipping match {match_id} due to missing data.")
            continue

        reddit_comments = reddit_comments_df.to_dict('records')

        # Create chunks
        chunks = create_chunks(match_id, odds_data, ball_by_ball, reddit_comments)

        # Save chunks
        output_path = output_dir / f"{match_id}.json"
        save_chunks(chunks, str(output_path))
        print(f"Finished processing match {match_id}")


if __name__ == "__main__":
    main()
