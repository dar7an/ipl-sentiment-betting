import pandas as pd
import praw
import os
import csv
import time
from prawcore.exceptions import RequestException, ResponseException

# Create a dictonary of threads
threads_df = pd.read_csv('reddit_threads.csv', header=None, names=['number', 'match', 'thread'])
thread_dict = {i+1: [row['number'], row['match'], row['thread']] for i, row in threads_df.iterrows()}

# Read-only Reddit instance
reddit = praw.Reddit(
    client_id='',
    client_secret='',
    user_agent='',
)

# Function to retrieve comments from Reddit
def get_comments(submission_url, retries=6, initial_wait=30):
    submission = reddit.submission(url=submission_url)
    submission.comment_sort = 'old'
    
    wait_time = initial_wait
    
    # Retry logic with exponential backoff
    for attempt in range(retries):
        try:
            submission.comments.replace_more(limit=None)
            for top_level_comment in submission.comments:
                yield top_level_comment.body, top_level_comment.score
            break
        
        except (RequestException, ResponseException) as e:
            if '429' in str(e):
                print(f"Rate limited. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                raise
    else:
        raise Exception(f"Failed to fetch comments from {submission_url} after {retries} attempts.")

# Function to save comments to CSV
TARGET_DIR = ''

def save_to_csv(comments, filename):
    os.makedirs(TARGET_DIR, exist_ok=True)
    filepath = os.path.join(TARGET_DIR, filename)
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Comment', 'Score'])
        writer.writerows(comments)

# Main loop to process all threads in the thread_dict
for current_thread_number, (match_number, match_teams, thread_url) in thread_dict.items():
    print(f"Processing Match {match_number}: {match_teams}")

    comments = []
    try:
        for comment, score in get_comments(thread_url):
            comments.append((comment, score))

    except Exception as e:
        print(f"Error for Match {match_number}: {match_teams}")
        print(e)

    finally:
        if comments:
            filename = f'{match_number}.csv'
            save_to_csv(comments, filename)
            print(f"Saved {len(comments)} comments for Match {match_number}: {match_teams}")

        else:
            print(f"Exiting with no comments for Match {match_number}: {match_teams}")
