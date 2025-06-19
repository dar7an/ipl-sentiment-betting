# -*- coding: utf-8 -*-
"""
12th Man Insights - IPL Match Analysis Tool (v9 - Full API)

This script provides high-quality, data-driven analysis of IPL matches for sports traders.
It synthesizes on-field events, odds movements, and fan sentiment into objective summaries
by leveraging the power of Google's Gemini models via the AI Studio API.

This version moves all analysis to the API to resolve dependency issues:
- Full API Stack: Removes the local sentiment model. All analysis, including sentiment,
  is now performed by the powerful `gemini-2.5-flash-preview-05-20` model.
- Simplified Dependencies: Eliminates torch, transformers, and other complex local AI
  libraries, fixing all environment-specific build errors.
- Secure API Key Handling: Loads the Google AI API key from a `.env` file.
- Professional Persona Prompting: The model is strictly instructed to act as a
  professional sports data analyst, filter out noise, and focus on trader-centric signals.

Dependencies:
pip install pandas tqdm praw python-dotenv requests nbformat nltk accelerate psutil google-generativeai
"""

import os
import pandas as pd
import argparse
import json
from dotenv import load_dotenv
import google.generativeai as genai
import time

# --- Configuration ---
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
load_dotenv()

class MatchAnalyzer:
    """
    A class to encapsulate the entire IPL match analysis process, using the
    Google AI API for all generative tasks.
    """

    def __init__(self):
        """
        Initializes the analyzer, configuring the Google AI API client.
        """
        # --- Google AI API Setup ---
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found. Please create a .env file with your key.")
        genai.configure(api_key=api_key)

        system_prompt = (
            "You are a professional sports-data analyst for a high-frequency trading firm. Your audience is expert cricket traders who need to cut through noise to find actionable signals. Your task is to provide objective, data-driven summaries of IPL match intervals.\n\n"
            "**CRITICAL RULES:**\n"
            "1. **ANALYZE ALL DATA:** You will be given on-field action, odds, and raw comments. Your summary MUST synthesize all three. Perform sentiment analysis on the comments as part of your task.\n"
            "2. **ADHERE TO DATA:** Your analysis must be strictly based on the data provided. Do NOT invent information. If a data source is unavailable, you MUST state that.\n"
            "3. **THINK LIKE A TRADER:** Focus on what matters: momentum shifts, significant player actions, odds movements, and the *substantive* meaning behind fan sentiment.\n"
            "4. **FILTER SENTIMENT NOISE:** Acknowledge low-signal comments (memes, trolls), but base your sentiment analysis on the genuine reactions.\n"
            "5. **MAINTAIN T20 CONTEXT:** Always interpret the data within the broader context of a T20 match (powerplay, middle overs, death overs).\n"
            "6. **BE OBJECTIVE & CONCISE:** Use neutral, analytical language. Avoid hype. Use markdown for clarity."
        )
        
        print("Initializing Google AI Generative Model...")
        self.generative_model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-preview-05-20',
            system_instruction=system_prompt
        )
        print("Google AI Model initialized successfully.")


    def generate_api_response(self, user_prompt):
        """Generates a response from the Google AI API."""
        try:
            response = self.generative_model.generate_content(user_prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Google AI API: {e}")
            return "Error: Could not generate a summary from the AI model."

    def format_odds(self, odds_data):
        """Formats odds data from the chunk structure."""
        if not odds_data or not isinstance(odds_data, list) or not odds_data[0].get("odds"):
            return "No odds data available for this interval."
        try:
            latest_odds_entry = odds_data[0]["odds"]
            odds_str = ", ".join([f"{o['name']}: {o['price']}" for o in latest_odds_entry])
            update_time = odds_data[0].get("last_update", "unknown time")
            return f"Latest odds ({update_time}): {odds_str}"
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Could not parse odds data: {e} - Data: {odds_data}")
            return "Could not parse odds data."

    def summarize_ball_by_ball(self, balls_data, team1_info, team2_info):
        """
        Summarizes key events from the ball-by-ball data list ('balls' key),
        including player-team associations.
        """
        if not balls_data or not isinstance(balls_data, list):
            return "No balls recorded in this interval."
        
        player_to_team = {}
        for player in team1_info.get("xi", []): player_to_team[player] = team1_info["name"]
        for player in team2_info.get("xi", []): player_to_team[player] = team2_info["name"]

        event_summary = []
        total_runs, wickets, fours, sixes, wides, dots, valid_balls_count = (0, 0, 0, 0, 0, 0, 0)
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
                
                batsman_name = ball_info.get("batsman", {}).get("fullname", "Unknown Batsman")
                bowler_name = ball_info.get("bowler", {}).get("fullname", "Unknown Bowler")
                batting_team_name = ball_info.get("name", batting_team_name)

                batsman_team = player_to_team.get(batsman_name, "")
                bowler_team = player_to_team.get(bowler_name, "")

                batsman_str = f"{batsman_name} ({batsman_team})" if batsman_team else batsman_name
                bowler_str = f"{bowler_name} ({bowler_team})" if bowler_team else bowler_name

                total_runs += runs
                if is_valid_ball: valid_balls_count += 1
                if "wide" in score_name: wides += 1
                
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
                
                if event_desc: event_summary.append(event_desc)
            except Exception as e:
                print(f"Warning: Error processing ball data: {e} - Data: {ball_info}")
                continue
        
        run_rate = (total_runs / (valid_balls_count / 6)) if valid_balls_count > 0 else 0
        overall_summary = (
            f"Summary for {batting_team_name}: {total_runs} runs from {valid_balls_count} balls "
            f"(RR: {run_rate:.2f}). Wickets: {wickets}, Fours: {fours}, Sixes: {sixes}, Wides: {wides}, Dots: {dots}."
        )
        
        full_summary = overall_summary
        if event_summary:
            full_summary += " Key events: " + " | ".join(event_summary)
        
        return full_summary

    def generate_match_update(self, ball_summary, odds_summary, comments, team1_name, team2_name):
        """Generates a professional, data-driven summary of a match interval using the Google AI API."""
        
        data_points = []
        if "No balls recorded" not in ball_summary:
            data_points.append(f"### On-Field Action\n{ball_summary}")
        
        if "No odds data available" not in odds_summary:
            data_points.append(f"### Odds Analysis\n{odds_summary}")

        if comments:
            comment_texts = json.dumps([c.get("comment", "") for c in comments if c.get("comment") and c.get("comment") != "[deleted]"], indent=2)
            data_points.append(f"### Raw Fan Comments (for sentiment analysis)\n{comment_texts}")
        
        if not data_points:
            return "No new data available in this interval to generate an update."

        user_prompt_content = "\n\n".join(data_points)
        user_prompt = f"""
        **Match Interval Report**
        **Teams:** {team1_name} vs {team2_name}

        {user_prompt_content}

        **Your Task:**
        Distill the provided data into a concise summary for a professional cricket trader. Analyze the on-field action in the context of the T20 game state, connect it to any odds movements, and perform sentiment analysis on the raw comments to summarize the *substantive* fan sentiment, filtering out the noise. Adhere strictly to your role and rules.
        """
        
        response = self.generate_api_response(user_prompt)
        
        return response.strip()

    def process_match_data(self, match_data, team1_info, team2_info):
        """Processes all data chunks for a match."""
        all_match_updates = []

        chunks = match_data.get('chunks', {})
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("name", f"chunk_{i+1}")
            print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ({chunk_id}) ---")

            # Prepare data for the API
            comments_in_chunk = chunk.get("comments", [])
            odds_summary = self.format_odds(chunk.get("odds"))
            ball_summary = self.summarize_ball_by_ball(chunk.get("balls"), team1_info, team2_info)
            
            # Match Update Generation
            update_text = self.generate_match_update(
                ball_summary, odds_summary, comments_in_chunk, team1_info['name'], team2_info['name']
            )
            print(f"  - Model Update: {update_text.replace(chr(10), ' ')[0:100]}...")

            all_match_updates.append({
                "chunk_id": chunk_id,
                "ball_by_ball_summary": ball_summary,
                "odds_summary": odds_summary,
                "analysis_update": update_text,
            })
            
            # Add a delay to respect API rate limits
            time.sleep(1)

        updates_df = pd.DataFrame(all_match_updates)
        return updates_df

def save_results_as_markdown(updates_df, output_path, team1_name, team2_name):
    """Saves the analysis results to a Markdown file."""
    print("\n--- Saving Results ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Match Analysis: {team1_name} vs {team2_name}\n\n")
            
            for _, row in updates_df.iterrows():
                f.write(f"## Interval: {row['chunk_id']}\n\n")

                f.write("### Ball-by-Ball Summary\n")
                f.write(f"{row['ball_by_ball_summary']}\n\n")

                f.write("### Odds Summary\n")
                f.write(f"{row['odds_summary']}\n\n")

                f.write("### AI-Generated Analysis\n")
                f.write(f"{row['analysis_update']}\n\n")

                f.write("---\n\n")
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to Markdown file: {e}")

def main(input_path, output_path):
    """Main function to run the enhanced analysis."""
    try:
        analyzer = MatchAnalyzer()
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return

    print(f"Processing {input_path}...")
    try:
        with open(input_path, 'r') as f:
            match_data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing JSON file: {e}")
        return

    # Attempt to dynamically load team info from the JSON structure
    match_info = match_data.get("match_info", {})
    team1_info = match_info.get("team1", {"name": "Team 1", "xi": []})
    team2_info = match_info.get("team2", {"name": "Team 2", "xi": []})
    print(f"Loaded team info: {team1_info['name']} vs {team2_info['name']}")
    
    updates_df = analyzer.process_match_data(
        match_data, team1_info, team2_info
    )

    save_results_as_markdown(updates_df, output_path, team1_info['name'], team2_info['name'])

    print("\nAnalysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IPL Match Analysis (v9 - Full API).")
    parser.add_argument("input_path", type=str, help="Path to the input JSON chunk file.")
    parser.add_argument("output_path", type=str, help="Path to save the output analysis Markdown file.")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
