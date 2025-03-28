# -*- coding: utf-8 -*-
"""
12th Man Insights - IPL Betting Analysis Script (v3)

This script analyzes IPL match data chunks (comments, odds, ball-by-ball)
to generate sentiment analysis and betting opportunity insights using a
causal language model (Google Gemma).

Processes ALL comments per chunk for sentiment.
Uses the correct 'balls' key for ball-by-ball data.

Dependencies:
pip install torch transformers pandas openpyxl accelerate
# Add 'bitsandbytes' if using quantization
"""

import os
import gc
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---

# Set MPS high watermark ratio (adjust if needed, 0.0 means no limit)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

MODEL_NAME = "google/gemma-3-4b-it"
DATA_FILE_PATH = "/Users/darshan/Documents/GitHub/ipl-sentiment-betting/model/chunks/2.json"  # Make sure this path is correct

# Token Limits (Adjust based on model and hardware capacity)
MAX_INPUT_TOKENS = 128000
MAX_NEW_TOKENS = 512

# Processing Controls
DEVICE_MAP = (
    "auto"  # Use "auto" for Transformers to handle device placement (CPU/GPU/MPS)
)

# --- Model and Tokenizer Loading ---

print(f"Loading model: {MODEL_NAME}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",  # Use bfloat16 if available and supported
        device_map=DEVICE_MAP,
        # Add quantization config here if needed, e.g.:
        # load_in_8bit=True # Requires bitsandbytes
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model name is correct and dependencies are installed.")
    exit()

# --- Helper Functions ---


def clear_memory():
    """Clears GPU cache and runs garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def generate_model_response(system_prompt, user_prompt):
    """
    Generates a response from the loaded causal LM.

    Args:
        system_prompt (str): The system prompt for the model.
        user_prompt (str): The user prompt for the model.

    Returns:
        str: The generated response text, lowercased and stripped.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Prepare input text
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return "Error: Could not format prompt."

    # Tokenize and check length BEFORE sending to model
    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,  # Truncate if exceeds limit
        truncation=True,
    )
    input_token_count = model_inputs.input_ids.shape[1]

    if input_token_count >= MAX_INPUT_TOKENS:
        print(
            f"Warning: Input length ({input_token_count}) was truncated to MAX_INPUT_TOKENS ({MAX_INPUT_TOKENS})."
        )

    model_inputs = model_inputs.to(model.device)

    # Generate response
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                # Use greedy decoding (do_sample=False) for deterministic output
                do_sample=False,
            )

        # Decode only the newly generated tokens
        generated_ids_only = generated_ids[:, model_inputs.input_ids.shape[1] :]
        response = tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)
        final_response = response[0].strip().lower()

    except Exception as e:
        print(f"Error during model generation: {e}")
        final_response = "Error: Model generation failed."
    finally:
        # Clean up tensors and memory
        del model_inputs, generated_ids, generated_ids_only
        clear_memory()

    return final_response


def analyze_sentiment(comment_text):
    """
    Analyzes the sentiment of a single comment.

    Args:
        comment_text (str): The text of the comment.

    Returns:
        str: 'positive', 'negative', or 'neutral'. Defaults to 'neutral' on error/unexpected output.
    """
    if not comment_text or not isinstance(comment_text, str):
        return "neutral"  # Handle empty or invalid input

    system_prompt = (
        "You are a sentiment analysis expert focused on T20 cricket, specifically the IPL."
        "Analyze the sentiment of the following comment from a live match thread."
        "Classify the comment's sentiment towards one of the teams involved or the general state of the match."
        "Answer ONLY with one word: 'positive', 'negative', or 'neutral'."
    )
    user_prompt = f'Comment: "{comment_text}"\nSentiment:'

    sentiment = generate_model_response(system_prompt, user_prompt)

    # Validate output
    valid_sentiments = ["positive", "negative", "neutral"]
    # Handle cases where the model might add punctuation or extra words
    cleaned_sentiment = sentiment.split()[0].strip(".,!?") if sentiment else ""

    if cleaned_sentiment not in valid_sentiments:
        print(
            f"Warning: Unexpected sentiment '{sentiment}' for comment: {comment_text[:50]}... Defaulting to neutral."
        )
        return "neutral"
    return cleaned_sentiment


# --- Data Summarization Functions ---


def summarize_comments(comments):
    """
    Placeholder: Summarizes a list of comments.
    Replace with actual summarization logic (e.g., sentiment counts, key topics).
    """
    if not comments:
        return "No comments in this interval."
    num_comments = len(comments)
    # Example: Just show the first few comments (inefficient for LLM)
    preview = " | ".join([c.get("comment", "")[:50] + "..." for c in comments[:3]])
    return f"~{num_comments} comments. Start: {preview}"


def format_odds(odds_data):
    """
    Formats odds data from the chunk structure.
    """
    if not odds_data or not isinstance(odds_data, list) or not odds_data[0].get("odds"):
        return "No odds data available."
    # Example: Extract latest odds prices
    try:
        latest_odds_entry = odds_data[0][
            "odds"
        ]  # Assuming the first entry is the latest
        odds_str = ", ".join([f"{o['name']}: {o['price']}" for o in latest_odds_entry])
        update_time = odds_data[0].get("last_update", "unknown time")
        return f"Odds ({update_time}): {odds_str}"
    except (KeyError, IndexError, TypeError) as e:
        print(f"Warning: Could not parse odds data: {e} - Data: {odds_data}")
        return "Could not parse odds data."


def summarize_ball_by_ball(balls_data):
    """
    Summarizes key events from the ball-by-ball data list ('balls' key) for the interval.
    """
    if not balls_data or not isinstance(balls_data, list):
        return "No balls recorded in this interval."

    total_runs = 0
    wickets = 0
    fours = 0
    sixes = 0
    wides = 0
    no_balls = 0  # Assuming 'noball' might be a score name or flag if present
    dots = 0
    valid_balls_count = 0
    batting_team = "Unknown"
    bowler = "Unknown"
    batsmen = set()

    for ball_info in balls_data:
        try:
            score_info = ball_info.get("score", {})
            runs = score_info.get("runs", 0)
            is_wicket = score_info.get("is_wicket", False)
            is_four = score_info.get("four", False)
            is_six = score_info.get("six", False)
            is_valid_ball = score_info.get(
                "ball", False
            )  # Counts towards over completion
            score_name = score_info.get("name", "").lower()

            total_runs += runs
            if is_wicket:
                wickets += 1
            if is_four:
                fours += 1
            if is_six:
                sixes += 1
            if "wide" in score_name:
                wides += 1
            # Add check for no balls if data includes it
            # if 'noball' in score_name: no_balls += 1
            if is_valid_ball and runs == 0 and not is_wicket:
                dots += 1  # Approx dot balls

            if is_valid_ball:
                valid_balls_count += 1

            # Capture team, bowler, batsmen involved
            if ball_info.get("name"):
                batting_team = ball_info["name"]
            if ball_info.get("bowler", {}).get("fullname"):
                bowler = ball_info["bowler"]["fullname"]
            if ball_info.get("batsman", {}).get("fullname"):
                batsmen.add(ball_info["batsman"]["fullname"])

        except Exception as e:
            print(f"Warning: Error processing ball data: {e} - Data: {ball_info}")
            continue  # Skip malformed ball data

    batsmen_str = ", ".join(list(batsmen)) if batsmen else "Unknown"
    summary = (
        f"{valid_balls_count} balls bowled by {bowler} to {batsmen_str} ({batting_team} batting). "
        f"Runs: {total_runs}, Wickets: {wickets}, Fours: {fours}, Sixes: {sixes}, "
        f"Wides: {wides}, Dots: {dots}."
    )

    # Add forecast if available in the last ball data
    if balls_data and balls_data[-1].get("forecast_data"):
        forecast = balls_data[-1]["forecast_data"]
        win_prob = forecast.get("win_probability", "")
        over_info = forecast.get("over_info", "")
        forecast_score = forecast.get("forecast", "")
        summary += (
            f" End Interval Forecast: {win_prob} | {over_info} | {forecast_score}"
        )

    return summary


# --- Betting Analysis Function ---


def analyze_betting_opportunity(chunk, team1_info, team2_info):
    """
    Analyzes a data chunk to identify potential betting opportunities.

    Args:
        chunk (dict): The data chunk containing comments, odds, balls.
        team1_info (dict): Dictionary with team 1 name and playing XI list.
        team2_info (dict): Dictionary with team 2 name and playing XI list.

    Returns:
        str: The model's analysis of the betting opportunity.
    """
    team1_name = team1_info.get("name", "Team 1")
    team1_xi_str = ", ".join(team1_info.get("xi", ["Not Available"]))
    team2_name = team2_info.get("name", "Team 2")
    team2_xi_str = ", ".join(team2_info.get("xi", ["Not Available"]))

    system_prompt = (
        f"You are a professional T20 cricket bettor analyzing an ongoing IPL match between {team1_name} and {team2_name}.\n"
        f"{team1_name} Playing XI: {team1_xi_str}\n"
        f"{team2_name} Playing XI: {team2_xi_str}\n"
        "Your goal is to identify profitable betting opportunities based on real-time data feeds within a short interval."
        "Analyze the provided sentiment summary, odds, and key match events summary from the ball-by-ball data."
        "Think step-by-step: 1. Assess fan sentiment summary. 2. Evaluate key match events summary. 3. Consider the current odds. 4. Conclude if there's a value bet."
        f"If you recommend a bet, clearly state the team ({team1_name} or {team2_name}) and provide concise reasoning based ONLY on the data provided."
        "If no clear opportunity exists, state 'No bet recommended' and briefly explain why (e.g., odds too low, situation unclear)."
        "Keep your final reasoning and conclusion under 100 words."
    )

    # Prepare summaries
    comments_summary = summarize_comments(chunk.get("comments", []))
    odds_summary = format_odds(chunk.get("odds", []))
    # *** Use the correct key 'balls' here ***
    bbb_summary = summarize_ball_by_ball(chunk.get("balls", []))

    user_prompt = f"""Data for this interval:
    Sentiment Summary: {comments_summary}
    Current Odds: {odds_summary}
    Recent Match Events (Ball-by-ball summary): {bbb_summary}

    Analysis and Betting Recommendation:"""

    print("Analyzing betting opportunity...")
    betting_opportunity = generate_model_response(system_prompt, user_prompt)
    print(f"Betting analysis result: {betting_opportunity}")
    return betting_opportunity


# --- Main Processing Function ---


def process_match_data(chunks, team1_info, team2_info):
    """
    Processes all data chunks for a match. Analyzes sentiment for ALL comments.

    Args:
        chunks (list): A list of data chunk dictionaries.
        team1_info (dict): Information for team 1.
        team2_info (dict): Information for team 2.

    Returns:
        tuple: (pandas.DataFrame, pandas.DataFrame) containing sentiments and betting opportunities.
    """
    all_sentiments = []
    all_betting_opportunities = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Processing Chunk {i+1}/{total_chunks} ---")

        # 1. Analyze Sentiment for ALL comments in the chunk
        comments_in_chunk = chunk.get("comments", [])
        print(f"Analyzing sentiment for {len(comments_in_chunk)} comments...")
        # Iterate through ALL comments, no slicing
        for comment_data in comments_in_chunk:
            comment_text = comment_data.get("comment", "")
            if comment_text and comment_text != "[deleted]":  # Skip deleted comments
                sentiment = analyze_sentiment(comment_text)
                all_sentiments.append(
                    {
                        "chunk_id": i,
                        "comment": comment_text,
                        "sentiment": sentiment,
                        "upvotes": comment_data.get("upvotes", 0),  # Store upvotes too
                    }
                )

        # 2. Analyze Betting Opportunity for the chunk
        # Ensure team1_info and team2_info are passed correctly
        betting_opportunity = analyze_betting_opportunity(chunk, team1_info, team2_info)
        all_betting_opportunities.append(
            {
                "chunk_id": i,
                "start_time": chunk.get("start_time", "N/A"),
                "end_time": chunk.get("end_time", "N/A"),
                "betting_opportunity_analysis": betting_opportunity,
            }
        )

        # Optional: Clear memory more aggressively if running into issues
        # print("Clearing memory after chunk...")
        # clear_memory()

    sentiments_df = pd.DataFrame(all_sentiments)
    betting_opportunities_df = pd.DataFrame(all_betting_opportunities)

    return sentiments_df, betting_opportunities_df


# --- Main Execution Block ---

if __name__ == "__main__":
    print("Starting 12th Man Insights Analysis...")

    # --- Load Data ---
    try:
        df_raw = pd.read_json(DATA_FILE_PATH)
        if "chunks" not in df_raw.columns:
            raise ValueError("JSON file must contain a 'chunks' column/key.")
        match_chunks = df_raw["chunks"].tolist()
        print(f"Loaded {len(match_chunks)} data chunks from {DATA_FILE_PATH}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        exit()
    except Exception as e:
        print(f"Error reading or parsing JSON file: {e}")
        exit()

    # --- Define Team Info (Replace/Enhance with dynamic loading) ---
    # !! IMPORTANT: This is hardcoded for the PBKS vs DC match example. !!
    pbks_info = {
        "name": "Punjab Kings",
        "xi": [
            "S Dhawan",
            "JM Bairstow",
            "LS Livingstone",
            "JM Sharma",
            "SM Curran",
            "HV Patel",
            "Harpreet Brar",
            "Shashank Singh",
            "K Rabada",
            "RD Chahar",
            "Arshdeep Singh",
        ],
    }
    dc_info = {
        "name": "Delhi Capitals",
        "xi": [
            "MR Marsh",
            "DA Warner",
            "SD Hope",
            "RR Pant",
            "T Stubbs",
            "AR Patel",
            "Sumit Kumar",
            "Kuldeep Yadav",
            "RK Bhui",
            "KK Ahmed",
            "I Sharma",
        ],
    }

    # --- Assign Teams (Hardcoded for this example) ---
    # Assuming for 2.json, PBKS is Team 1 and DC is Team 2.
    # TODO: Add logic here to determine team1/team2 based on file metadata or content if needed.
    team1_info = pbks_info
    team2_info = dc_info
    print(f"Assigned Team 1: {team1_info['name']}, Team 2: {team2_info['name']}")

    # --- Process Data ---
    print("\nStarting chunk processing...")
    # Pass the assigned team info dictionaries
    sentiments_df, betting_opportunities_df = process_match_data(
        match_chunks, team1_info, team2_info
    )
    print("\nFinished processing all chunks.")

    # --- Display Results ---
    if not sentiments_df.empty:
        print("\n--- Sentiment Analysis Results ---")
        print("Overall Sentiment Distribution:")
        print(sentiments_df["sentiment"].value_counts())
        print("\nOverall Sentiment Distribution (%):")
        print((sentiments_df["sentiment"].value_counts(normalize=True) * 100).round(2))
        # Optional: Save sentiments
        # sentiments_df.to_csv("sentiment_results.csv", index=False)
        # print("\nSentiment results saved to sentiment_results.csv")
    else:
        print("\nNo sentiment results generated.")

    if not betting_opportunities_df.empty:
        print("\n--- Betting Opportunity Analysis Results ---")
        # Display the analysis for each chunk
        for index, row in betting_opportunities_df.iterrows():
            print(
                f"\nChunk {row['chunk_id']} ({row['start_time']} - {row['end_time']}):"
            )
            print(f"  Analysis: {row['betting_opportunity_analysis']}")
        # Optional: Save betting opportunities
        # betting_opportunities_df.to_csv("betting_results.csv", index=False)
        # print("\nBetting results saved to betting_results.csv")
    else:
        print("\nNo betting opportunity results generated.")

    print("\nAnalysis complete.")
