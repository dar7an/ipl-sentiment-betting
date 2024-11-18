import os
import pandas as pd

def remove_username_column(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            if 'Username' in df.columns:
                df.drop(columns=['Username'], inplace=True)
                df.to_csv(filepath, index=False)
                print(f"Processed {filename}")

if __name__ == "__main__":
    directory = "/Users/darshan/Documents/GitHub/ipl-sentiment-betting/reddit/2024"
    remove_username_column(directory)
