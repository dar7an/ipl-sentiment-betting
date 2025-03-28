# -*- coding: utf-8 -*-
"""
12th Man Insights - IPL Betting Analysis Script (v5 - Stricter Formatting)

This script analyzes IPL match data chunks (comments, odds, ball-by-ball)
to generate sentiment analysis and stateful betting opportunity insights
using a causal language model (Google Gemma).

Features:
- Stateful betting: Tracks active bets (Enter/Hold/Exit).
- Conditional prompting based on active bet state.
- Enhanced ball-by-ball summarization with player-team context.
- Refined reasoning prompts with consistency checks.
- **Stricter output format instructions for the LLM.**
- Structured action output parsing (ACTION: ..., REASON: ...).
- Processes ALL comments per chunk for sentiment.
- Corrected generation parameters.

Dependencies:
pip install torch transformers pandas openpyxl accelerate regex
# Add 'bitsandbytes' if using quantization
"""

import os
import gc
import re  # Import regex for parsing
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

MODEL_NAME = "google/gemma-3-4b-it"
DATA_FILE_PATH = (
    "/Users/darshan/Documents/GitHub/ipl-sentiment-betting/model/chunks/2.json"
)

MAX_INPUT_TOKENS = 128000
MAX_NEW_TOKENS = 350
DEVICE_MAP = "auto"

# --- Model and Tokenizer Loading ---
print(f"Loading model: {MODEL_NAME}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map=DEVICE_MAP,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
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
    """Generates a response from the loaded causal LM."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return "Error: Could not format prompt."

    model_inputs = tokenizer(
        [text], return_tensors="pt", max_length=MAX_INPUT_TOKENS, truncation=True
    )
    input_token_count = model_inputs.input_ids.shape[1]
    if input_token_count >= MAX_INPUT_TOKENS:
        print(f"Warning: Input length ({input_token_count}) truncated.")

    model_inputs = model_inputs.to(model.device)
    final_response = "Error: Model generation failed."  # Default

    # --- Corrected Generation Arguments ---
    generation_args = {
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,  # Pass attention mask explicitly
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": False,  # Keep greedy for format consistency
        # No top_k or top_p when do_sample=False
    }

    try:
        with torch.no_grad():
            generated_ids = model.generate(**generation_args)

        generated_ids_only = generated_ids[:, model_inputs.input_ids.shape[1] :]
        response = tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)
        final_response = response[0].strip()
    except Exception as e:
        print(f"Error during model generation: {e}")
    finally:
        del model_inputs, generated_ids, generated_ids_only
        clear_memory()
    return final_response


def analyze_sentiment(comment_text):
    """Analyzes the sentiment of a single comment."""
    if not comment_text or not isinstance(comment_text, str):
        return "neutral"
    system_prompt = (
        "You are a sentiment analysis expert focused on T20 cricket, specifically the IPL."
        "Analyze the sentiment of the following comment from a live match thread."
        "Classify the comment's sentiment towards one of the teams involved or the general state of the match."
        "Answer ONLY with one word: 'positive', 'negative', or 'neutral'."
    )
    user_prompt = f'Comment: "{comment_text}"\nSentiment:'
    sentiment = generate_model_response(system_prompt, user_prompt).lower()
    valid_sentiments = ["positive", "negative", "neutral"]
    cleaned_sentiment = sentiment.split()[0].strip(".,!?") if sentiment else ""
    if cleaned_sentiment not in valid_sentiments:
        # print(f"Warning: Unexpected sentiment '{sentiment}' for comment: {comment_text[:50]}... Defaulting to neutral.")
        return "neutral"  # Reduce noise by not printing warning every time
    return cleaned_sentiment


# --- Data Summarization Functions ---
# (Keep summarize_comments, format_odds, summarize_ball_by_ball as in v4)
def summarize_comments(comments):
    """Placeholder: Summarizes a list of comments."""
    if not comments:
        return "No comments in this interval."
    num_comments = len(comments)
    preview = " | ".join([c.get("comment", "")[:50] + "..." for c in comments[:3]])
    return f"~{num_comments} comments. Start: {preview}"


def format_odds(odds_data):
    """Formats odds data from the chunk structure."""
    if not odds_data or not isinstance(odds_data, list) or not odds_data[0].get("odds"):
        return "No odds data available."
    try:
        latest_odds_entry = odds_data[0]["odds"]
        odds_str = ", ".join([f"{o['name']}: {o['price']}" for o in latest_odds_entry])
        update_time = odds_data[0].get("last_update", "unknown time")
        return f"Odds ({update_time}): {odds_str}"
    except (KeyError, IndexError, TypeError) as e:
        print(f"Warning: Could not parse odds data: {e} - Data: {odds_data}")
        return "Could not parse odds data."


def summarize_ball_by_ball(balls_data, team1_info, team2_info):
    """
    Summarizes key events from the ball-by-ball data list ('balls' key),
    including player-team associations.
    """
    if not balls_data or not isinstance(balls_data, list):
        return "No balls recorded in this interval."
    player_to_team = {}
    for player in team1_info.get("xi", []):
        player_to_team[player] = team1_info["name"]
    for player in team2_info.get("xi", []):
        player_to_team[player] = team2_info["name"]
    event_summary = []
    total_runs, wickets, fours, sixes, wides, dots, valid_balls_count = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    batting_team_name = "Unknown"
    for ball_info in balls_data:
        try:
            score_info = ball_info.get("score", {})
            ball_num = ball_info.get("ball", "?")
            runs = score_info.get("runs", 0)
            is_wicket = score_info.get("is_wicket", False)
            is_four = score_info.get("four", False)
            is_six = score_info.get("six", False)
            is_valid_ball = score_info.get("ball", False)
            score_name = score_info.get("name", "").lower()
            batsman_name = ball_info.get("batsman", {}).get(
                "fullname", "Unknown Batsman"
            )
            bowler_name = ball_info.get("bowler", {}).get("fullname", "Unknown Bowler")
            batting_team_name = ball_info.get("name", batting_team_name)
            batsman_team = player_to_team.get(batsman_name, "")
            bowler_team = player_to_team.get(bowler_name, "")
            batsman_str = (
                f"{batsman_name} ({batsman_team})" if batsman_team else batsman_name
            )
            bowler_str = (
                f"{bowler_name} ({bowler_team})" if bowler_team else bowler_name
            )
            total_runs += runs
            if is_valid_ball:
                valid_balls_count += 1
            if "wide" in score_name:
                wides += 1
            event_desc = None
            if is_wicket:
                wickets += 1
                event_desc = f"WICKET at {ball_num}! {batsman_str} out b {bowler_str}."
            elif is_six:
                sixes += 1
                event_desc = f"SIX at {ball_num}! by {batsman_str} off {bowler_str}."
            elif is_four:
                fours += 1
                event_desc = f"FOUR at {ball_num}! by {batsman_str} off {bowler_str}."
            elif is_valid_ball and runs == 0:
                dots += 1
            if event_desc:
                event_summary.append(event_desc)
        except Exception as e:
            print(f"Warning: Error processing ball data: {e} - Data: {ball_info}")
            continue
    run_rate = (total_runs / (valid_balls_count / 6)) if valid_balls_count > 0 else 0
    overall_summary = (
        f"Interval Summary ({batting_team_name} batting): {total_runs} runs off {valid_balls_count} balls "
        f"(RR: {run_rate:.2f}). W: {wickets}, 4s: {fours}, 6s: {sixes}, Wd: {wides}, Dots: {dots}."
    )
    full_summary = overall_summary
    if event_summary:
        full_summary += " Key events: " + " | ".join(event_summary)
    if balls_data and balls_data[-1].get("forecast_data"):
        forecast = balls_data[-1]["forecast_data"]
        win_prob = forecast.get("win_probability", "")
        over_info = forecast.get("over_info", "")
        forecast_score = forecast.get("forecast", "")
        full_summary += (
            f" | End Interval Forecast: {win_prob} / {over_info} / {forecast_score}"
        )
    return full_summary


# --- Betting Analysis Function (Stateful) ---


def analyze_betting_opportunity(
    chunk, team1_info, team2_info, is_bet_active, active_bet_team, entry_details
):
    """
    Analyzes a data chunk statefully to recommend betting actions (Enter, Hold, Exit).
    Uses stricter prompt formatting.
    """
    team1_name = team1_info.get("name", "Team 1")
    team2_name = team2_info.get("name", "Team 2")
    team1_xi_str = ", ".join(team1_info.get("xi", ["Not Available"]))
    team2_xi_str = ", ".join(team2_info.get("xi", ["Not Available"]))

    comments_summary = summarize_comments(chunk.get("comments", []))
    odds_summary = format_odds(chunk.get("odds", []))
    bbb_summary = summarize_ball_by_ball(chunk.get("balls", []), team1_info, team2_info)

    if not is_bet_active:
        # --- Stricter Prompt to ENTER / NO_ACTION ---
        system_prompt = (
            f"You are a professional T20 cricket bettor analyzing an ongoing IPL match between {team1_name} and {team2_name}.\n"
            f"{team1_name} Playing XI: {team1_xi_str}\n"
            f"{team2_name} Playing XI: {team2_xi_str}\n"
            "You currently have NO active bet. Your goal is to identify if there is a *strong value opportunity to ENTER a bet* now.\n"
            "Analyze the provided sentiment summary, odds, and the detailed ball-by-ball event summary for this interval.\n"
            "Think step-by-step:\n"
            "1. Assess fan sentiment towards **each team**.\n"
            "2. Evaluate the key match events summary, explicitly noting which team benefited or suffered (e.g., '{team1_name} bowler struggled', '{team2_name} batsmen scored freely').\n"
            "3. Consider the current odds for **both teams**.\n"
            "4. **Synthesize these factors:** Is there a clear mismatch between performance/sentiment and odds suggesting a value bet opportunity on either {team1_name} or {team2_name}?\n"
            "5. Conclude with a recommendation.\n"
            "CRITICAL: Your final recommendation *must* be logically consistent with your analysis. Do not recommend betting *on* a team if your analysis shows they performed poorly in this interval, unless there's a very strong counter-argument (e.g., odds shifted dramatically favorable).\n"
            "**VERY IMPORTANT: Format your response EXACTLY as follows, using ONLY these specific action words on the FIRST line:**\n"
            "ACTION: ENTER {Team Name}\n"  # Replace {Team Name} with the actual team name if recommending a bet
            "REASON: [Your concise reasoning (under 100 words) based on step 4 & 5].\n"
            "**OR, if no bet is recommended:**\n"
            "ACTION: NO_ACTION\n"
            "REASON: [Your concise reasoning (under 100 words) explaining why no bet is recommended].\n"
            "**Your response MUST start with 'ACTION: ' followed by 'ENTER {Team Name}' or 'NO_ACTION'. NO OTHER FORMAT IS ALLOWED for the ACTION line.**"
        )
        user_prompt = f"""Current State: No active bet.
        Data for this interval:
        Sentiment Summary: {comments_summary}
        Current Odds: {odds_summary}
        Recent Match Events Summary: {bbb_summary}

        Analysis and Betting Recommendation (MUST start with ACTION: ENTER TeamName or ACTION: NO_ACTION):"""

    else:  # A bet IS active
        # --- Stricter Prompt to HOLD / EXIT ---
        system_prompt = (
            f"You are a professional T20 cricket bettor analyzing an ongoing IPL match between {team1_name} and {team2_name}.\n"
            f"{team1_name} Playing XI: {team1_xi_str}\n"
            f"{team2_name} Playing XI: {team2_xi_str}\n"
            f"You currently HAVE AN ACTIVE BET on **{active_bet_team}** (entered around {entry_details}). Your goal is to decide whether to **HOLD** this bet or **EXIT** it now.\n"
            "Analyze the *latest* data: sentiment summary, odds changes, and the detailed ball-by-ball event summary for this interval.\n"
            "Think step-by-step:\n"
            "1. Assess recent fan sentiment towards **{active_bet_team}** and their opponent.\n"
            "2. Evaluate the latest key match events summary. Did events in this interval favor or hurt your bet on **{active_bet_team}**?\n"
            "3. Consider the current odds. Have they moved significantly for or against **{active_bet_team}** since entry?\n"
            "4. **Synthesize these factors:** Based on recent performance, sentiment, and odds movement, is the original reason for the bet still valid? Is it better to secure profit/cut losses now, or is holding likely to be more profitable?\n"
            "5. Conclude with a recommendation.\n"
            "CRITICAL: Your final recommendation *must* be logically consistent with your analysis. For example, if {active_bet_team} performed very poorly in this interval and odds moved against them, strongly consider EXIT unless sentiment/other factors provide a compelling reason to HOLD.\n"
            "**VERY IMPORTANT: Format your response EXACTLY as follows, using ONLY these specific action words on the FIRST line:**\n"
            "ACTION: HOLD\n"
            "REASON: [Your concise reasoning (under 100 words) based on step 4 & 5].\n"
            "**OR:**\n"
            "ACTION: EXIT\n"
            "REASON: [Your concise reasoning (under 100 words) based on step 4 & 5].\n"
            "**Your response MUST start with 'ACTION: ' followed by 'HOLD' or 'EXIT'. NO OTHER FORMAT IS ALLOWED for the ACTION line.**"
        )
        user_prompt = f"""Current State: Active bet on {active_bet_team} (entered {entry_details}).
        Data for this interval:
        Sentiment Summary: {comments_summary}
        Current Odds: {odds_summary}
        Recent Match Events Summary: {bbb_summary}

        Analysis and Recommendation (Hold or Exit - MUST start with ACTION: HOLD or ACTION: EXIT):"""

    print(
        f"Analyzing betting opportunity (State: {'Active Bet on ' + active_bet_team if is_bet_active else 'No Active Bet'})..."
    )
    raw_response = generate_model_response(system_prompt, user_prompt)
    print(f"Raw LLM Response: {raw_response}")

    # --- Parse the Response ---
    action = "NO_ACTION"
    reasoning = "Error: Could not parse LLM response."
    team_to_enter = None

    # Use regex that handles potential variations in whitespace and case, focusing on the start of the string
    action_match = re.match(r"ACTION:\s*([^\n]+)", raw_response, re.IGNORECASE)
    # Reason starts after the first newline following ACTION: line
    reason_match = re.search(
        r"ACTION:[^\n]+\nREASON:\s*(.*)", raw_response, re.IGNORECASE | re.DOTALL
    )

    if action_match:
        action_text = action_match.group(1).strip().upper()
        if not is_bet_active:
            if action_text.startswith("ENTER"):
                action = "ENTER"
                team_match = re.search(
                    r"ENTER\s+(.+)", action_text
                )  # Extract team name after ENTER
                if team_match:
                    team_name_raw = team_match.group(1).strip()
                    # Normalize and validate team name
                    if team_name_raw == team1_name.upper():
                        team_to_enter = team1_name
                    elif team_name_raw == team2_name.upper():
                        team_to_enter = team2_name
                    else:
                        print(
                            f"Warning: LLM suggested entering invalid team '{team_name_raw}'. Defaulting to NO_ACTION."
                        )
                        action = "NO_ACTION"
                else:
                    print(
                        "Warning: LLM action was 'ENTER' but team name was missing/invalid. Defaulting to NO_ACTION."
                    )
                    action = "NO_ACTION"
            elif action_text == "NO_ACTION":
                action = "NO_ACTION"
            else:  # Invalid action for this state
                print(
                    f"Warning: LLM provided invalid action '{action_text}' when no bet was active. Defaulting to NO_ACTION."
                )
                action = "NO_ACTION"
        else:  # Bet is active
            if action_text == "HOLD":
                action = "HOLD"
            elif action_text == "EXIT":
                action = "EXIT"
            else:  # Invalid action for this state
                print(
                    f"Warning: LLM provided invalid action '{action_text}' when bet was active. Defaulting to HOLD."
                )
                action = "HOLD"  # Default to HOLD if active and action is invalid

    if reason_match:
        reasoning = reason_match.group(1).strip()
    elif (
        action_match
    ):  # If action was parsed but reason wasn't (maybe missing REASON: line)
        reasoning = "Reason not found in expected format."

    if action == "ENTER":
        print(f"Parsed Action: {action} {team_to_enter}, Reason: {reasoning[:100]}...")
        return action, team_to_enter, reasoning
    else:
        print(f"Parsed Action: {action}, Reason: {reasoning[:100]}...")
        return action, None, reasoning


# --- Main Processing Function (Stateful) ---
# (Keep process_match_data function exactly as in v4)
def process_match_data(chunks, team1_info, team2_info):
    """Processes all data chunks statefully for a match."""
    all_sentiments = []
    betting_actions_log = []
    total_chunks = len(chunks)
    is_bet_active = False
    active_bet_team = None
    entry_details = None
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("name", f"chunk_{i}")
        start_time = chunk.get("start_time", "N/A")
        end_time = chunk.get("end_time", "N/A")
        print(f"\n--- Processing {chunk_id} ({start_time} - {end_time}) ---")
        print(
            f"Current Betting State: {'Active bet on ' + active_bet_team if is_bet_active else 'No Active Bet'}"
        )
        comments_in_chunk = chunk.get("comments", [])
        print(f"Analyzing sentiment for {len(comments_in_chunk)} comments...")
        for comment_data in comments_in_chunk:
            comment_text = comment_data.get("comment", "")
            if comment_text and comment_text != "[deleted]":
                sentiment = analyze_sentiment(comment_text)
                all_sentiments.append(
                    {
                        "chunk_id": chunk_id,
                        "comment": comment_text,
                        "sentiment": sentiment,
                        "upvotes": comment_data.get("upvotes", 0),
                    }
                )
        action, team_involved, reasoning = analyze_betting_opportunity(
            chunk, team1_info, team2_info, is_bet_active, active_bet_team, entry_details
        )
        state_before_action = (
            f"{'Active on ' + active_bet_team if is_bet_active else 'Inactive'}"
        )
        if action == "ENTER":
            if not is_bet_active:
                is_bet_active = True
                active_bet_team = team_involved
                entry_details = f"Chunk {chunk_id} ({start_time})"
                print(f"STATE CHANGE: Entered bet on {active_bet_team}")
            else:
                print(
                    f"INFO: Received ENTER action for {team_involved}, but bet already active on {active_bet_team}. Holding."
                )
                action = "HOLD"
                team_involved = active_bet_team
                reasoning += " (Note: Overridden to HOLD as bet already active)"
        elif action == "EXIT":
            if is_bet_active:
                print(f"STATE CHANGE: Exited bet on {active_bet_team}")
                is_bet_active = False
                active_bet_team = None
                entry_details = None
            else:
                print(
                    "INFO: Received EXIT action, but no bet is active. No action taken."
                )
                action = "NO_ACTION"
                reasoning += " (Note: Overridden to NO_ACTION as no bet was active)"
        betting_actions_log.append(
            {
                "chunk_id": chunk_id,
                "start_time": start_time,
                "end_time": end_time,
                "action_taken": action,
                "team_involved": (
                    active_bet_team
                    if is_bet_active
                    else (team_involved if action == "ENTER" else None)
                ),
                "reasoning": reasoning,
                "state_before": state_before_action,
                "state_after": f"{'Active on ' + active_bet_team if is_bet_active else 'Inactive'}",
            }
        )
        clear_memory()
    sentiments_df = pd.DataFrame(all_sentiments)
    betting_actions_df = pd.DataFrame(betting_actions_log)
    return sentiments_df, betting_actions_df


# --- Main Execution Block ---
# (Keep __main__ block exactly as in v4)
if __name__ == "__main__":
    print("Starting 12th Man Insights Analysis (Stateful)...")
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
    team1_info = pbks_info
    team2_info = dc_info
    print(f"Assigned Team 1: {team1_info['name']}, Team 2: {team2_info['name']}")
    print("\nStarting stateful chunk processing...")
    sentiments_df, betting_actions_df = process_match_data(
        match_chunks, team1_info, team2_info
    )
    print("\nFinished processing all chunks.")
    if not sentiments_df.empty:
        print("\n--- Sentiment Analysis Results ---")
        print("Overall Sentiment Distribution:")
        print(sentiments_df["sentiment"].value_counts())
        # sentiments_df.to_csv("sentiment_results_stateful.csv", index=False)
    else:
        print("\nNo sentiment results generated.")
    if not betting_actions_df.empty:
        pd.set_option("display.max_colwidth", 200)
        print("\n--- Betting Actions Log ---")
        print(
            betting_actions_df[
                [
                    "chunk_id",
                    "start_time",
                    "action_taken",
                    "team_involved",
                    "reasoning",
                    "state_after",
                ]
            ].to_string(index=False)
        )
        # betting_actions_df.to_csv("betting_actions_log.csv", index=False)
    else:
        print("\nNo betting actions log generated.")
    print("\nAnalysis complete.")
