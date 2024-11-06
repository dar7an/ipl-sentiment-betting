import json
import csv

# Read JSON file
with open("the_odds_api/fixtures.json", "r") as json_file:
    fixtures = json.load(json_file)

# Write to CSV
with open("the_odds_api/fixtures.csv", "w", newline="") as csv_file:
    # Define headers based on JSON structure
    fieldnames = [
        "id",
        "sport_key",
        "sport_title",
        "commence_time",
        "home_team",
        "away_team",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write headers
    writer.writeheader()

    # Write each fixture
    for fixture in fixtures:
        writer.writerow(fixture)

print("CSV file has been created successfully!")
