import os
import requests
from typing import List, Optional


class OddsAPI:
    BASE_URL = "https://api.the-odds-api.com/v4/historical"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_historical_odds(
        self,
        sport: str,
        event_id: str,
        date: str,
        regions: List[str],
        markets: List[str],
        bookmakers: List[str],
    ) -> Optional[dict]:
        """
        Fetch historical odds data for a specific event

        Args:
            sport (str): Sport key (e.g., 'basketball_nba')
            event_id (str): Event ID
            date (str): ISO format date (e.g., '2023-11-29T22:45:00Z')
            regions (List[str]): List of regions (default: ['us'])
            markets (List[str]): List of markets (default: ['h2h'])
            bookmakers (List[str]): List of bookmakers (default: ['unibet_eu'])

        Returns:
            dict: API response data or None if request fails
        """
        try:
            url = f"{self.BASE_URL}/sports/{sport}/events/{event_id}/odds"

            params = {
                "apiKey": self.api_key,
                "date": date,
                "regions": ",".join(regions),
                "markets": ",".join(markets),
                "bookmakers": ",".join(bookmakers),
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error fetching odds data: {e}")
            return None


# Example usage:
if __name__ == "__main__":
    client = OddsAPI(os.getenv("ODDS_API_KEY"))

    # Example parameters
    sport = "cricket_ipl"
    event_id = "45b7eacdd4a34d9ed1fd4013462999b0"
    date = "2024-03-22T16:00:00Z"
    regions = ["us"]
    markets = ["h2h"]
    bookmakers = ["fanduel"]

    result = client.get_historical_odds(
        sport=sport, event_id=event_id, date=date, regions=regions, markets=markets, bookmakers=bookmakers
    )

    if result:
        print(result)
