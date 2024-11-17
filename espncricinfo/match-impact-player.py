import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def extract_player_stats(url):
    try:
        # Send GET request to the URL
        response = requests.get(url)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the table in the page
        table = soup.find(
            "table", class_="ds-w-full ds-table ds-table-md ds-table-auto"
        )

        # Initialize lists to store data
        data = {"Player": [], "Team": [], "TI": [], "B. Impact": [], "Bo. Impact": []}

        # Extract rows from table
        rows = table.find_all("tr")[1:]  # Skip header row

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 9:  # Ensure row has enough columns
                data["Player"].append(cols[0].text.strip())
                data["Team"].append(cols[1].text.strip())
                data["TI"].append(cols[2].text.strip())
                data["B. Impact"].append(cols[5].text.strip())
                data["Bo. Impact"].append(cols[8].text.strip())

        # Create DataFrame
        df = pd.DataFrame(data)
        return df

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def process_all_matches(csv_file, output_dir):
    matches = pd.read_csv(csv_file)
    for index, row in matches.iterrows():
        match_number = row['match_number']
        url = row['link']
        result = extract_player_stats(url)
        if result is not None:
            output_file = os.path.join(output_dir, f"{match_number}.csv")
            result.to_csv(output_file, index=False)
            print(f"Saved data for match {match_number} to {output_file}")

# Example usage:
csv_file = "/Users/darshan/Documents/GitHub/ipl-sentiment-betting/espncricinfo/2024_match-impact-player.csv"
output_dir = "/Users/darshan/Documents/GitHub/ipl-sentiment-betting/espncricinfo/impact"
process_all_matches(csv_file, output_dir)
