import os
import csv
import shutil

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

odds_dir = 'the_odds_api/odds'
print(f"Looking for files in: {os.path.abspath(odds_dir)}")

# Print list of files in odds directory
print("Files in odds directory:", os.listdir(odds_dir))

with open('the_odds_api/timestamps.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        old_filename = f"{row['the_odds_api_id']}.json"
        old_path = os.path.join(odds_dir, old_filename)
        
        print(f"Checking for file: {old_path}")
        if os.path.exists(old_path):
            new_filename = f"{row['sportmonks_id']}.json"
            new_path = os.path.join(odds_dir, new_filename)
            shutil.move(old_path, new_path)
            print(f"Renamed {old_filename} to {new_filename}")
