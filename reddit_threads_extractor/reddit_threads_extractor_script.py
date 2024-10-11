import pandas as pd
import praw
import os
import csv
import time
from prawcore.exceptions import RequestException, ResponseException

# Load threads data from CSV into a dictionary
threads_df = pd.read_csv(
    "old_reddit_threads.csv", header=None, names=["number", "match", "thread"]
)
thread_dict = {
    i + 1: [row["number"], row["match"], row["thread"]]
    for i, row in threads_df.iterrows()
}

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id="xESeT0xUGodn0tYLJ-QNJA",
    client_secret="55vjXKRQsjrDuOB8z1SpzyB6_u0L-A",
    user_agent="Script by u/minimalisticiam",
)


def get_comments(submission_url, retries=6, initial_wait=30):
    submission = reddit.submission(url=submission_url)
    submission.comment_sort = "old"

    wait_time = initial_wait

    # Retry logic with exponential backoff
    for attempt in range(retries):
        try:
            submission.comments.replace_more(limit=None)
            for top_level_comment in submission.comments:
                yield top_level_comment.body, top_level_comment.score
            break

        except (RequestException, ResponseException) as e:
            if "429" in str(e):
                print(
                    f"Rate limited. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})"
                )
                time.sleep(wait_time)
                wait_time *= 2
            else:
                raise
    else:
        raise Exception(
            f"Failed to fetch comments from {submission_url} after {retries} attempts."
        )


TARGET_DIR = "reddit_threads"


def save_to_csv(comments, filename):
    os.makedirs(TARGET_DIR, exist_ok=True)
    filepath = os.path.join(TARGET_DIR, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Comment", "Upvotes"])
        writer.writerows(comments)


# Start processing from the first thread
starting_thread_number = 1

for current_thread_number in range(starting_thread_number, len(thread_dict) + 1):
    match_number, match_teams, thread_url = thread_dict[current_thread_number]
    print(f"Processing Match {match_number}: {match_teams}")

    comments = []
    try:
        for comment, score in get_comments(thread_url):
            comments.append((comment, score))

    except Exception as e:
        print(f"Error {match_number}: {match_teams}")
        print(e)

    finally:
        if comments:
            filename = f"{match_number}.csv"
            save_to_csv(comments, filename)
            print(
                f"Saved {len(comments)} comments for Match {match_number}: {match_teams}"
            )

        else:
            print(f"Exit with no comments for Match {match_number}: {match_teams}")
