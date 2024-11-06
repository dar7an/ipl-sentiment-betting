import os
import json
import csv

# Directory containing the JSON files
input_directory = "sportmonks/2024/match_data_copy"
output_file = "the_odds_api/timestamps.csv"

# Open the output CSV file
with open(output_file, mode="w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["match", "abandoned", "start", "end"])  # Headers

    # Get and sort the list of JSON files in the directory
    filenames = sorted(f for f in os.listdir(input_directory) if f.endswith(".json"))

    # Process each JSON file in the directory
    for filename in filenames:
        file_path = os.path.join(input_directory, filename)

        with open(file_path, "r") as file:
            data = json.load(file)

            # Check if the match is abandoned
            note = data["data"].get("note", "").lower()
            if "abandoned" in note:
                csv_writer.writerow([filename, True, None, None])
            else:
                match_start_time = data["data"]["starting_at"]
                balls = data["data"].get("balls", [])
                match_end_time = balls[-1]["updated_at"] if balls else None
                csv_writer.writerow(
                    [filename, False, match_start_time, match_end_time]
                )

print(f"Timestamps extracted and saved to {output_file}")
