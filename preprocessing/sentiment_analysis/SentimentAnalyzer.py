from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


def get_sentiment_score(comment: str) -> float:
    """
    Calculates the compound sentiment score for a given comment using VADER.

    Args:
        comment: The text comment to analyze.

    Returns:
        The compound sentiment score, a float between -1 and 1.
        Returns 0.0 if the input is not a string.
    """
    if not isinstance(comment, str):
        return 0.0

    return sid.polarity_scores(comment)["compound"]