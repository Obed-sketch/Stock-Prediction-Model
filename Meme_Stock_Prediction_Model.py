# -*- coding: utf-8 -*-
"""
Meme Stock Predictor using Reddit/Twitter Hype and Technical Indicators
"""
# %% [1] Install Required Libraries
!pip install yfinance praw snscrape pandas-ta numpy tensorflow matplotlib textblob

# %% [2] Import Libraries
import yfinance as yf #Fetch stock data from Yahoo Finance
import pandas as pd
import numpy as np
import praw #Access Reddit API
import snscrape.modules.twitter as sntwitter #Scrape Twitter data
from textblob import TextBlob
from pandas_ta import rsi, macd #Technical analysis indicators
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout #LSTM neural network
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# %% [3] Data Collection Functions
def get_stock_data(ticker, start_date, end_date): #Fetches historical price data
    """
    Fetch historical stock data from Yahoo Finance
    Args:
        ticker (str): Stock symbol (e.g., 'GME')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    Returns:
        pd.DataFrame: Historical stock data
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def get_reddit_posts(subreddit, limit=1000): #Retrieves Reddit posts using PRAW
    """
    Scrape Reddit posts using PRAW
    Args:
        subreddit (str): Subreddit name (e.g., 'wallstreetbets')
        limit (int): Number of posts to retrieve
    Returns:
        pd.DataFrame: Reddit posts data
    """
    reddit = praw.Reddit(
        client_id='YOUR_CLIENT_ID',
        client_secret='YOUR_CLIENT_SECRET',
        user_agent='YOUR_USER_AGENT'
    )
    
    posts = []
    for post in reddit.subreddit(subreddit).hot(limit=limit):
        posts.append([
            post.title, post.score, post.id,
            post.subreddit, post.url, post.num_comments,
            post.selftext, post.created_utc
        ])
    
    return pd.DataFrame(posts, columns=[
        'title', 'score', 'id', 'subreddit', 
        'url', 'num_comments', 'body', 'created'
    ])

def get_tweets(query, limit=1000): #Collects Twitter data without API using SNScrape
    """
    Scrape Twitter data using SNScrape
    Args:
        query (str): Search query (e.g., '$GME')
        limit (int): Number of tweets to retrieve
    Returns:
        pd.DataFrame: Twitter data
    """
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i > limit:
            break
        tweets.append([
            tweet.date, tweet.content, tweet.user.username,
            tweet.likeCount, tweet.retweetCount
        ])
    
    return pd.DataFrame(tweets, columns=[
        'date', 'content', 'user', 'likes', 'retweets'
    ])

# %% [4] Sentiment Analysis Functions
def analyze_sentiment(text): #Uses TextBlob for polarity scores
    """
    Perform sentiment analysis using TextBlob
    Args:
        text (str): Input text
    Returns:
        float: Sentiment polarity (-1 to 1)
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def process_social_data(df, platform): #Aggregates daily sentiment scores
    """
    Process social media data and add sentiment scores
    Args:
        df (pd.DataFrame): Social media data
        platform (str): 'reddit' or 'twitter'
    Returns:
        pd.DataFrame: Processed data with sentiment
    """
    if platform == 'reddit':
        text_col = 'title'
        date_col = 'created'
    else:
        text_col = 'content'
        date_col = 'date'
    
    df['sentiment'] = df[text_col].apply(analyze_sentiment)
    df['date'] = pd.to_datetime(df[date_col]).dt.date
    return df.groupby('date')['sentiment'].mean().reset_index()

# %% [5] Feature Engineering
def create_features(stock_data, reddit_sentiment, twitter_sentiment):
    """
    Combine stock data with social media sentiment
    Args:
        stock_data (pd.DataFrame): Historical stock data
        reddit_sentiment (pd.DataFrame): Reddit sentiment data
        twitter_sentiment (pd.DataFrame): Twitter sentiment data
    Returns:
        pd.DataFrame: Combined feature set
    """
    # Calculate technical indicators
    stock_data['RSI'] = rsi(stock_data['Close'])
    stock_data[['MACD', 'MACD_signal']] = macd(stock_data['Close'])
    
    # Merge social sentiment
    combined = stock_data.copy()
    combined = pd.merge(combined, reddit_sentiment, 
                       left_index=True, right_on='date', how='left')
    combined = pd.merge(combined, twitter_sentiment,
                       left_index=True, right_on='date', how='left')
    
    # Fill missing values
    combined[['sentiment_x', 'sentiment_y']] = combined[
        ['sentiment_x', 'sentiment_y']].fillna(0)
    combined['combined_sentiment'] = (
        combined['sentiment_x'] * 0.7 + combined['sentiment_y'] * 0.3
    )
    
    return combined[['Close', 'RSI', 'MACD', 'MACD_signal', 'combined_sentiment']]

# %% [6] LSTM Model
def create_lstm_model(input_shape):
    """
    Create LSTM model architecture
    Args:
        input_shape (tuple): Input shape for the model
    Returns:
        tf.keras.Model: Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# %% [7] Main Execution
if __name__ == "__main__":
    # Configuration
    ticker = 'GME'
    start_date = '2020-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Get data
    stock_data = get_stock_data(ticker, start_date, end_date)
    reddit_data = get_reddit_posts('wallstreetbets', 5000)
    twitter_data = get_tweets(f'${ticker}', 5000)
    
    # Process social data
    reddit_sentiment = process_social_data(reddit_data, 'reddit')
    twitter_sentiment = process_social_data(twitter_data, 'twitter')
    
    # Create feature set
    features = create_features(stock_data, reddit_sentiment, twitter_sentiment)
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    
    # Create sequences
    time_steps = 7
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build and train model
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, 
                       epochs=50, batch_size=32,
                       validation_data=(X_test, y_test))
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse scaling
    test_predictions = predictions.copy()
    test_actual = y_test.copy()
    
    dummy_array = np.zeros((len(test_predictions), features.shape[1]))
    dummy_array[:, 0] = test_predictions.flatten()
    test_predictions = scaler.inverse_transform(dummy_array)[:, 0]
    
    dummy_array[:, 0] = test_actual
    test_actual = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(test_actual, label='Actual Price')
    plt.plot(test_predictions, label='Predicted Price')
    plt.title(f'{ticker} Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()




