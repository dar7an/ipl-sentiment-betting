import datetime
import json
import csv
from pathlib import Path


def process_match(match_id):
    try:
        # Load all data
        with open(f"balls/{match_id}.json") as f:
            balls_data = json.load(f)

        comments = []
        with open(f"comments/{match_id}.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                comments.append(
                    {
                        "timestamp": datetime.datetime.strptime(
                            row["Timestamp"], "%Y-%m-%d %I:%M:%S %p IST"
                        ).strftime("%Y-%m-%d %I:%M:%S %p"),
                        "comment": row["Comment"],
                        "upvotes": int(row["Upvotes"]),
                    }
                )

        with open(f"odds/{match_id}.json") as f:
            odds_data = json.load(f)

        # Get first and last ball timestamps for each innings
        first_innings_balls = []
        second_innings_balls = []
        first_team_id = balls_data["balls"][0]["id"]

        for ball in balls_data["balls"]:
            ball_time = datetime.datetime.strptime(
                ball["updated_at"], "%Y-%m-%d %I:%M:%S %p IST"
            ).strftime("%Y-%m-%d %I:%M:%S %p")
            if ball["id"] == first_team_id:
                first_innings_balls.append(ball_time)
            else:
                second_innings_balls.append(ball_time)

        # Get innings transition points
        first_innings_end = max(first_innings_balls)
        second_innings_start = min(second_innings_balls)

        # Get unique odds times while preserving order
        odds_times = []
        seen_times = set()
        for odd in odds_data:
            odds_time = datetime.datetime.strptime(
                odd["last_update"], "%Y-%m-%d %I:%M:%S %p IST"
            ).strftime("%Y-%m-%d %I:%M:%S %p")
            if odds_time not in seen_times:
                odds_times.append(odds_time)
                seen_times.add(odds_time)
        odds_times.sort()

        # Get the last ball timestamp to know when to stop creating chunks
        last_ball_time = max(first_innings_balls + second_innings_balls)

        # Create two pregame chunks
        # First chunk - from first comment until first odds
        first_pregame_chunk = {
            "name": "chunk_1",
            "start_time": min([c["timestamp"] for c in comments]),
            "end_time": odds_times[0],
            "is_pregame": True,
            "comments": [c for c in comments if c["timestamp"] < odds_times[0]],
            "odds": [
                o
                for o in odds_data
                if datetime.datetime.strptime(
                    o["last_update"], "%Y-%m-%d %I:%M:%S %p IST"
                ).strftime("%Y-%m-%d %I:%M:%S %p")
                == odds_times[0]
            ],
        }

        # Second chunk - from first odds until second odds (game start)
        second_pregame_chunk = {
            "name": "chunk_2",
            "start_time": odds_times[0],
            "end_time": odds_times[1],
            "is_pregame": True,
            "comments": [
                c for c in comments if odds_times[0] <= c["timestamp"] < odds_times[1]
            ],
            "odds": [
                o
                for o in odds_data
                if datetime.datetime.strptime(
                    o["last_update"], "%Y-%m-%d %I:%M:%S %p IST"
                ).strftime("%Y-%m-%d %I:%M:%S %p")
                == odds_times[1]
            ],
        }

        chunks = [first_pregame_chunk, second_pregame_chunk]

        # Create regular game chunks
        chunk_start = odds_times[1]  # Start from second odds timestamp
        chunk_index = 3  # Start chunk index from 3
        while chunk_start < last_ball_time:
            # Find next unique odds time or add 5 minutes
            next_odds_time = next((t for t in odds_times if t > chunk_start), None)
            if next_odds_time:
                chunk_end = next_odds_time
            else:
                # Add 5 minutes to current time
                chunk_start_dt = datetime.datetime.strptime(
                    chunk_start, "%Y-%m-%d %I:%M:%S %p"
                )
                chunk_end = (chunk_start_dt + datetime.timedelta(minutes=5)).strftime(
                    "%Y-%m-%d %I:%M:%S %p"
                )

            # Rest of the code remains same
            is_innings_break = first_innings_end <= chunk_start <= second_innings_start

            if is_innings_break:
                chunk = {
                    "name": f"chunk_{chunk_index}",
                    "start_time": chunk_start,
                    "end_time": chunk_end,
                    "is_pregame": False,
                    "is_innings_break": True,
                    "innings_break": {
                        "first_innings_end": first_innings_end,
                        "second_innings_start": second_innings_start,
                    },
                    "comments": [
                        c for c in comments if chunk_start <= c["timestamp"] < chunk_end
                    ],
                    "odds": [
                        o
                        for o in odds_data
                        if datetime.datetime.strptime(
                            o["last_update"], "%Y-%m-%d %I:%M:%S %p IST"
                        ).strftime("%Y-%m-%d %I:%M:%S %p")
                        == chunk_start
                    ],
                }
            else:
                chunk = {
                    "name": f"chunk_{chunk_index}",
                    "start_time": chunk_start,
                    "end_time": chunk_end,
                    "is_pregame": False,
                    "is_innings_break": False,
                    "balls": [
                        b
                        for b in balls_data["balls"]
                        if chunk_start
                        <= datetime.datetime.strptime(
                            b["updated_at"], "%Y-%m-%d %I:%M:%S %p IST"
                        ).strftime("%Y-%m-%d %I:%M:%S %p")
                        < chunk_end
                    ],
                    "comments": [
                        c for c in comments if chunk_start <= c["timestamp"] < chunk_end
                    ],
                    "odds": [
                        o
                        for o in odds_data
                        if datetime.datetime.strptime(
                            o["last_update"], "%Y-%m-%d %I:%M:%S %p IST"
                        ).strftime("%Y-%m-%d %I:%M:%S %p")
                        == chunk_start
                    ],
                }

            chunks.append(chunk)
            chunk_start = chunk_end
            chunk_index += 1

        # Save chunks to file
        output_path = Path(f"chunks/{match_id}.json")
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "chunks": chunks,
                },
                f,
                indent=2,
                default=str,
            )

        print(f"Successfully processed match {match_id} - Created {len(chunks)} chunks")
        return True

    except FileNotFoundError as e:
        print(f"Error processing match {match_id}: Missing file - {e.filename}")
        return False
    except Exception as e:
        print(f"Error processing match {match_id}: {str(e)}")
        return False


def main():
    # Create list of match IDs excluding 70, 66, and 63
    match_ids = [i for i in range(1, 75) if i not in [70, 66, 63]]

    successful = 0
    failed = 0

    for match_id in match_ids:
        if process_match(match_id):
            successful += 1
        else:
            failed += 1

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful}")
    print(f"Failed to process: {failed}")


if __name__ == "__main__":
    main()
