import os
import json

def extract_odds(directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                # Create output dictionary to store all entries
                all_entries = []
                
                for entry in data:
                    timestamp = entry['timestamp']
                    odds_data = entry['odds']['data']
                    
                    # Add error checking
                    if not odds_data.get('bookmakers') or len(odds_data['bookmakers']) == 0:
                        print(f"Warning: No bookmakers data found for timestamp {timestamp} in file {filename}")
                        continue
                        
                    last_update = odds_data['bookmakers'][0]['last_update']
                    outcomes = odds_data['bookmakers'][0]['markets'][0]['outcomes']
                    
                    game_data = {
                        "last_update": last_update,
                        "odds": outcomes
                    }
                    all_entries.append(game_data)
                
                # Write all entries to output file with same name as input
                output_filepath = os.path.join(output_directory, filename)
                with open(output_filepath, 'w') as output_file:
                    json.dump(all_entries, output_file, indent=4)

if __name__ == "__main__":
    directory = "/Users/darshan/Documents/GitHub/ipl-sentiment-betting/the_odds_api/2024"
    output_directory = "/Users/darshan/Documents/GitHub/ipl-sentiment-betting/the_odds_api/2024_trimmed"
    extract_odds(directory, output_directory)
    
# Warning: No bookmakers data found for timestamp 2024-03-29T16:00:00Z in file 10.json
# Warning: No bookmakers data found for timestamp 2024-05-04T16:35:00Z in file 52.json
# Warning: No bookmakers data found for timestamp 2024-05-04T17:00:00Z in file 52.json
# Warning: No bookmakers data found for timestamp 2024-05-12T17:35:00Z in file 62.json