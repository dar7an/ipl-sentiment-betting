# Sentiment Analysis on IPL Betting Odds

## Introduction

This project delves into the potential of sentiment analysis to enhance betting strategies for Indian Premier League (IPL) cricket matches. By analyzing the sentiment expressed in tweets and other social media posts, I seek to uncover patterns and correlations that could provide valuable insights for informed betting decisions.


## Datasets

I would like to thank [The Odds API](https://the-odds-api.com/) for providing historical odds data and [Sportmonks](https://www.sportmonks.com/) for providing ball-by-ball match data. The project would not have been possible without their support ❤️

## Project Structure

- [`.env`](.env) — Configuration file for environment variables.
- [`.gitignore`](.gitignore) — Specifies keys, files, and directories to be ignored by Git
- [`model/`](model/)
  - `model.ipynb` — Jupyter Notebook containing the data modeling process
  - `balls/` — Contains JSON files (`1.json` to `36.json`, etc.) used in the modeling
  - `chunks/` — Directory for data chunks used in processing
  - `comments/` — Includes user comments data for analysis
  - `odds/` — Contains data related to betting odds
- [`preprocessing/`](preprocessing/)
  - `reddit/` — Scripts and data for preprocessing Reddit data
  - `sentiment_analysis/` — Tools for performing sentiment analysis
  - `sportmonks/` — Code for interacting with the SportMonks API
  - `the_odds_api/` — Code for fetching data from The Odds API
- [`README.md`](README.md) — Project documentation

## Installation

To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/dar7an/ipl-sentiment-betting.git
cd ipl-sentiment-betting
pip install -r requirements.txt
