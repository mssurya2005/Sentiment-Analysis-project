import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Sample movie reviews
reviews = ["This movie was amazing! The storyline was gripping and the characters were well-developed.",
    "I hated this movie. It was too long and the plot was boring.",
    "The movie was okay, some parts were good but it was too predictable.",
    "Fantastic film! Brilliant acting and a very moving story.",
    "Not my cup of tea. The pace was too slow and I couldn't relate to the characters."
]

# Analyze the sentiment for each review
for review in reviews:
    sentiment_scores = sid.polarity_scores(review)
    print(f"Review: {review}")
    print(f"Sentiment Scores: {sentiment_scores}")
    
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    print(f"Overall Sentiment: {sentiment}\n")