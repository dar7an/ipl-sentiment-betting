import pandas as pd
import os

def enhance_csv(file_path):
    df = pd.read_csv(file_path)
    if 'over_info' in df.columns:
        df['over_info'] = df['over_info'].apply(lambda x: x.split('•')[1].strip() if '•' in x else x)
        df.to_csv(file_path, index=False)

if __name__ == "__main__":
    directory = '/Users/darshan/Documents/GitHub/ipl-sentiment-betting/espncricinfo'
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            enhance_csv(file_path)
