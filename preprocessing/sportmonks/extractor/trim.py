import json
import os
from typing import Dict


def extract_summary(data: Dict) -> Dict:
    """Extract summary information from match data."""
    return {
        "id": data["data"]["id"],
        "round": data["data"]["round"],
        "localteam_id": data["data"]["localteam_id"],
        "visitorteam_id": data["data"]["visitorteam_id"],
        "starting_at": data["data"]["starting_at"],
        "note": data["data"]["note"],
        "venue_id": data["data"]["venue_id"],
        "toss_won_team_id": data["data"]["toss_won_team_id"],
        "winner_team_id": data["data"]["winner_team_id"],
    }


def extract_ball_info(ball: Dict) -> Dict:
    """Extract information for a single ball."""
    return {
        "ball": ball["ball"],
        "updated_at": ball["updated_at"],
        "id": ball["team"]["id"],
        "name": ball["team"]["name"],
        "score": {
            "name": ball["score"]["name"],
            "runs": ball["score"]["runs"],
            "four": ball["score"]["four"],
            "six": ball["score"]["six"],
            "bye": ball["score"]["bye"],
            "leg_bye": ball["score"]["leg_bye"],
            "is_wicket": ball["score"]["is_wicket"],
            "ball": ball["score"]["ball"],
            "out": ball["score"]["out"],
        },
        "batsman": {
            "id": ball["batsman"]["id"],
            "fullname": ball["batsman"]["fullname"],
            "battingstyle": ball["batsman"]["battingstyle"],
        },
        "bowler": {
            "id": ball["bowler"]["id"],
            "fullname": ball["bowler"]["fullname"],
            "bowlingstyle": ball["bowler"]["bowlingstyle"],
        },
    }


def process_match_data(input_path: str, output_path: str) -> None:
    """Process a single match JSON file and save trimmed data."""
    with open(input_path, "r") as file:
        data = json.load(file)

    summary = extract_summary(data)
    balls_data = [extract_ball_info(ball) for ball in data["data"]["balls"]]

    output_data = {"summary": summary, "balls": balls_data}

    with open(output_path, "w") as outfile:
        json.dump(output_data, outfile, indent=4)


def main():
    """Main function to process all JSON files."""
    input_folder = (
        "/Users/darshan/Documents/GitHub/ipl-sentiment-trader/sportmonks/2024/"
    )
    output_folder = (
        "/Users/darshan/Documents/GitHub/ipl-sentiment-trader/sportmonks/trimmed/"
    )

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each JSON file
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_match_data(input_path, output_path)


if __name__ == "__main__":
    main()
