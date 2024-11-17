from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time


class WinProbabilityScraper:
    def __init__(self):
        # Add Chrome options for better performance
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.probabilities = []
        self.over_1_count = 0
        # Cache selectors
        self.SELECTORS = {
            "graph": "div.ds-w-full.ds-bg-fill-content-prime.ds-overflow-hidden",
            "slider": "div.ds-absolute.ds-w-0\\.5.ds-h-\\[200px\\].ds-top-0.ds-z-10.ds-cursor-move.ds-bg-clip-padding.ds-box-content.ds-select-none.ds-bg-fill-contrast",
            "win_prob": "div.ds-text-tight-s.ds-font-bold",
            "over_info": "p.ds-text-compact-xs.ds-font-medium",
            "forecast_container": "div.ds-px-4.ds-pb-3.ds-select-none",
            "forecast": "span.ds-text-compact-xs",
            "separator": "line.ds-stroke-line-default-translucent",
        }
        # Reduce default wait time
        self.wait = WebDriverWait(self.driver, 2)
        self.short_wait = WebDriverWait(self.driver, 0.1)

    def setup_driver(self, url):
        self.driver.get(url)

    def find_graph_container(self):
        try:
            return self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.SELECTORS["graph"])
                )
            )
        except:
            return None

    def get_slider(self):
        try:
            return self.short_wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.SELECTORS["slider"])
                )
            )
        except:
            return None

    def wait_for_over_update(self, target_over):
        max_attempts = 10
        attempts = 0
        while attempts < max_attempts:
            over_info = self.get_over_info()
            if over_info:
                current_over = float(over_info.split()[1])
                if abs(current_over - target_over) < 0.1:
                    return True
            time.sleep(0.2)
            attempts += 1
        return False

    def get_current_over_number(self):
        over_info = self.get_over_info()
        if over_info:
            try:
                return float(over_info.split()[1])
            except (IndexError, ValueError):
                return None
        return None

    def move_slider_incrementally(self, start_position, target_position, container):
        try:
            slider = self.get_slider()
            actions = ActionChains(self.driver)
            distance = target_position - start_position
            increment = 250 if distance > 0 else -1
            steps = int(abs(distance) // abs(increment))
            last_over = self.get_current_over_number()
            actions.move_to_element(slider).click_and_hold()

            for step in range(steps):
                actions.move_by_offset(increment, 0).perform()
                actions = ActionChains(self.driver).move_by_offset(0, 0)
                current_over_num = self.get_current_over_number()
                if current_over_num and current_over_num != last_over:
                    if current_over_num == 1:
                        self.over_1_count += 1
                        if self.over_1_count >= 2:
                            self.capture_final_state(current_over_num)
                            return False
                    self.scrape_data(current_over_num)
                    last_over = current_over_num

            actions.release().perform()
            return True

        except Exception as e:
            print(f"Error moving slider: {e}")
            raise

    def capture_final_state(self, current_over_num):
        win_prob = self.get_win_probability()
        over_info = self.get_over_info()
        forecast = self.get_forecast()
        if all(x is not None for x in [win_prob, over_info, forecast]):
            self.probabilities.insert(
                0,
                {
                    "over": current_over_num,
                    "win_probability": win_prob,
                    "over_info": over_info,
                    "forecast": forecast,
                },
            )
            print("Final state captured before stopping")

    def scrape_data(self, current_over_num):
        try:
            data = {
                "win_probability": self.get_win_probability(),
                "over_info": self.get_over_info(),
                "forecast": self.get_forecast(),
            }

            if all(data.values()):
                slider_x_position = self.get_current_slider_position()

                self.probabilities.insert(
                    0, {"over": current_over_num, **data}
                )
                print(f"Detected over change: {data['over_info']}")
        except Exception as e:
            print(f"Error scraping data: {e}")

    def get_current_slider_position(self):
        return self.get_slider().rect["x"]

    def calculate_target_position(self, container, over_number):
        container_rect = container.rect
        total_width = container_rect["width"]
        start_x = container_rect["x"]
        relative_position = over_number / 40
        target_x = start_x + (total_width * relative_position)
        print(f"Over {over_number}: Moving to position {target_x}")
        return target_x

    def get_win_probability(self):
        try:
            return self.short_wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.SELECTORS["win_prob"])
                )
            ).text
        except:
            return None

    def get_over_info(self):
        try:
            over_element = self.short_wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.SELECTORS["over_info"])
                )
            )
            over_info = over_element.text
            return over_info.split('•')[1].strip() if '•' in over_info else over_info
        except:
            return None

    def get_forecast(self):
        try:
            container_element = self.short_wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.SELECTORS["forecast_container"])
                )
            )
            forecast_element = container_element.find_element(
                By.CSS_SELECTOR, self.SELECTORS["forecast"]
            )
            return forecast_element.text
        except:
            return None

    def scrape_current_state(self):
        win_prob = self.get_win_probability()
        over_info = self.get_over_info()
        forecast = self.get_forecast()
        if all(x is not None for x in [win_prob, over_info, forecast]):
            current_over = float(over_info.split()[1])
            self.probabilities.append(
                {
                    "over": current_over,
                    "win_probability": win_prob,
                    "over_info": over_info,
                    "forecast": forecast,
                }
            )
            print(f"Initial state captured: {over_info}")

    def get_over_range(self, current_over, innings):
        if innings == 2:
            return range(int(current_over), 0, -1)
        else:
            max_over = min(20, int(current_over))
            return range(max_over, 0, -1)

    def scrape_innings(self, container, start_position, current_over):
        print(f"Starting scrape from over {current_over}")
        for target_over in range(int(current_over), 0, -1):
            target_position = self.calculate_target_position(container, target_over)
            current_position = self.get_current_slider_position()
            if not self.move_slider_incrementally(
                current_position, target_position, container
            ):
                return

    def scrape_match(self, url):
        try:
            self.setup_driver(url)
            self.separator_x_position = self.get_innings_separator_position()

            # Pre-allocate list size for better memory management
            self.probabilities = []
            self.scrape_current_state()

            if self.probabilities:
                current_state = self.probabilities[0]
                container = self.find_graph_container()
                if container:
                    self.scrape_innings(
                        container,
                        self.get_current_slider_position(),
                        current_state["over"],
                    )

            # Convert to DataFrame only once at the end
            return pd.DataFrame(self.probabilities)
        except Exception as e:
            print(f"Error scraping match: {e}")
            return pd.DataFrame()
        finally:
            self.driver.quit()

    def save_data(self, df, match_number):
        if not df.empty:
            output_dir = "/Users/darshan/Documents/GitHub/ipl-sentiment-betting/espncricinfo"
            filename = f"{output_dir}/{match_number}.csv"
            df.to_csv(filename, index=False, compression=None)
            print(f"Data saved to {filename}")

    def get_innings_separator_position(self):
        try:
            separator_line = self.driver.find_element(
                By.CSS_SELECTOR, self.SELECTORS["separator"]
            )
            return float(separator_line.get_attribute("x1"))
        except Exception as e:
            print(f"Could not find innings separator: {e}")
            return None

    def get_current_innings(self, slider_x_position, separator_x_position):
        return None


def main():
    import os
    import pandas as pd

    # Read the matches CSV
    matches_df = pd.read_csv("/Users/darshan/Documents/GitHub/ipl-sentiment-betting/espncricinfo/2024.csv")
    output_dir = "/Users/darshan/Documents/GitHub/ipl-sentiment-betting/espncricinfo"

    for _, row in matches_df.iterrows():
        match_number = row['match_number']
        match_url = row['link']
        
        # Check if file already exists
        output_file = f"{output_dir}/{match_number}.csv"
        if os.path.exists(output_file):
            print(f"Match {match_number} already processed, skipping...")
            continue

        print(f"Processing match {match_number}...")
        scraper = WinProbabilityScraper()
        df = scraper.scrape_match(match_url)
        
        if not df.empty:
            scraper.save_data(df, match_number)
            print(f"Successfully processed match {match_number}")
        else:
            print(f"Failed to scrape data for match {match_number}")

if __name__ == "__main__":
    main()
