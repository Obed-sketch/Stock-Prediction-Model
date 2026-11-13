# -*- coding: utf-8 -*-
"""
Meme Stock Predictor using Reddit/Twitter Hype and Technical Indicators
"""
# %% [1] Install Required Libraries
# (Was a Colab-style "!pip install ..." line, which is a SyntaxError in a
# plain .py file. Install the dependencies with:)
#     pip install -r requirements.txt

# %% [2] Import Libraries
import os

import yfinance as yf #Fetch stock data from Yahoo Finance
import pandas as pd
import numpy as np
# Optional data-source libraries (Nov 2026): imported tolerantly so the
# DEMO mode below runs without them. The live Reddit path needs praw;
# snscrape's Twitter scraping has been broken by upstream Twitter/X
# changes for a long time, so the live path treats tweets as optional.
try:
    import praw #Access Reddit API
except ImportError:
    praw = None
try:
    import snscrape.modules.twitter as sntwitter #Scrape Twitter data
except Exception:
    sntwitter = None
from textblob import TextBlob
# pandas_ta is optional: create_features() falls back to equivalent
# built-in RSI/MACD implementations when it is not installed.
try:
    from pandas_ta import rsi, macd #Technical analysis indicators
except ImportError:
    rsi = macd = None
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout #LSTM neural network
import matplotlib
matplotlib.use('Agg')  # headless: the plot is saved to a file, not shown
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
    if praw is None:
        raise RuntimeError("praw is not installed; the live Reddit path "
                           "requires it (or run with --demo).")
    # Credentials come from the environment; the placeholder defaults
    # keep the original values visible but cannot authenticate. Set
    # REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT.
    reddit = praw.Reddit(
        client_id=os.environ.get('REDDIT_CLIENT_ID', 'YOUR_CLIENT_ID'),
        client_secret=os.environ.get('REDDIT_CLIENT_SECRET', 'YOUR_CLIENT_SECRET'),
        user_agent=os.environ.get('REDDIT_USER_AGENT', 'YOUR_USER_AGENT')
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
    if sntwitter is None:
        raise RuntimeError("snscrape is unavailable; Twitter/X scraping "
                           "no longer works reliably -- see README.")
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
    # REPAIRED (Nov 2026), two date bugs: (1) the original used .dt.date
    # (datetime.date objects), which never match the price frame's
    # DatetimeIndex in create_features' merge -- sentiment silently came
    # through as all-NaN/0; normalized timestamps merge correctly.
    # (2) Reddit's 'created' values are epoch SECONDS; pd.to_datetime
    # on raw numbers assumes nanoseconds and lands in 1970, so numeric
    # timestamps are converted with unit='s'.
    raw_dates = df[date_col]
    if pd.api.types.is_numeric_dtype(raw_dates):
        df['date'] = pd.to_datetime(raw_dates, unit='s').dt.normalize()
    else:
        df['date'] = pd.to_datetime(raw_dates).dt.normalize()
    return df.groupby('date')['sentiment'].mean().reset_index()

# %% [5] Feature Engineering
def _rsi_manual(close, length=14):
    """Wilder's RSI, equivalent to pandas_ta.rsi (fallback, Nov 2026)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_manual(close, fast=12, slow=26, signal=9):
    """MACD line + signal line, equivalent to the two columns of
    pandas_ta.macd that create_features uses (fallback, Nov 2026).
    (pandas_ta.macd actually returns three columns including the
    histogram, so the original two-column assignment would have failed
    even with pandas_ta installed.)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({'MACD': macd_line, 'MACD_signal': signal_line},
                        index=close.index)


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
    # Newer yfinance returns MultiIndex columns even for one ticker.
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    # Calculate technical indicators (pandas_ta if available, else the
    # built-in equivalents above)
    if rsi is not None and macd is not None:
        stock_data['RSI'] = rsi(stock_data['Close'])
        macd_df = macd(stock_data['Close'])
        stock_data['MACD'] = macd_df.iloc[:, 0]
        stock_data['MACD_signal'] = macd_df.iloc[:, 2]
    else:
        stock_data['RSI'] = _rsi_manual(stock_data['Close'])
        stock_data[['MACD', 'MACD_signal']] = _macd_manual(stock_data['Close'])
    
    # Merge social sentiment on a plain 'date' column (Nov 2026 fix:
    # the original merged the price DatetimeIndex against the sentiment
    # frames' 'date' column; with current pandas that corrupts the
    # result's index and the second merge then fails outright).
    combined = stock_data.copy()
    combined['date'] = combined.index.normalize()
    combined = pd.merge(combined, reddit_sentiment, on='date', how='left')
    combined = pd.merge(combined, twitter_sentiment, on='date', how='left')
    combined = combined.set_index('date')
    
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
def run_pipeline(stock_data, reddit_data, twitter_data, ticker='GME',
                 epochs=50, plot_path='prediction_plot.png'):
    """Shared pipeline for both modes: sentiment -> features -> LSTM ->
    evaluation. Returns a dict of test-set metrics."""
    # Process social data
    reddit_sentiment = process_social_data(reddit_data, 'reddit')
    twitter_sentiment = process_social_data(twitter_data, 'twitter')

    # Create feature set. Rows with NaN indicators (the first ~26 days,
    # before RSI/MACD warm up) must be dropped -- the original script
    # scaled them straight through and trained on NaNs.
    features = create_features(stock_data, reddit_sentiment, twitter_sentiment)
    features = features.dropna()
    print(f"Feature frame: {features.shape[0]} rows x {features.shape[1]} cols "
          f"({features.index.min().date()} to {features.index.max().date()})")

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
    model.fit(X_train, y_train,
              epochs=epochs, batch_size=32,
              validation_data=(X_test, y_test), verbose=0)

    # Make predictions
    predictions = model.predict(X_test, verbose=0)

    # Inverse scaling
    dummy_array = np.zeros((len(predictions), features.shape[1]))
    dummy_array[:, 0] = predictions.flatten()
    test_predictions = scaler.inverse_transform(dummy_array)[:, 0]

    dummy_array[:, 0] = y_test
    test_actual = scaler.inverse_transform(dummy_array)[:, 0]

    rmse = float(np.sqrt(np.mean((test_predictions - test_actual) ** 2)))
    mae = float(np.mean(np.abs(test_predictions - test_actual)))
    print(f"Test set: {len(y_test)} days | RMSE ${rmse:.2f} | MAE ${mae:.2f}")
    print(f"Next-day prediction from last window: "
          f"${test_predictions[-1]:.2f} (last actual close ${test_actual[-1]:.2f})")

    # Plot results (saved to a file; headless environments have no display)
    plt.figure(figsize=(12, 6))
    plt.plot(test_actual, label='Actual Price')
    plt.plot(test_predictions, label='Predicted Price')
    plt.title(f'{ticker} Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    return {'rmse': rmse, 'mae': mae, 'n_test': int(len(y_test))}


def _load_demo_data(here):
    """DEMO data (Nov 2026): real GME prices (cached from Yahoo Finance,
    see sample_data/README note in the repo README) + SYNTHETIC sample
    social posts bundled in sample_data/."""
    sample = os.path.join(here, 'sample_data')
    cache = os.path.join(sample, 'gme_prices.csv')
    if os.path.exists(cache):
        stock_data = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"DEMO: loaded cached GME prices from sample_data/gme_prices.csv "
              f"({len(stock_data)} rows)")
    else:
        stock_data = get_stock_data('GME', '2021-01-01', '2023-12-31')
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        stock_data.to_csv(cache)
        print(f"DEMO: downloaded GME prices from Yahoo Finance and cached "
              f"them to sample_data/gme_prices.csv ({len(stock_data)} rows)")
    reddit_data = pd.read_csv(os.path.join(sample, 'reddit_posts.csv'))
    twitter_data = pd.read_csv(os.path.join(sample, 'twitter_posts.csv'))
    print(f"DEMO: {len(reddit_data)} synthetic Reddit posts, "
          f"{len(twitter_data)} synthetic tweets (sample_data/)")
    return stock_data, reddit_data, twitter_data


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Meme stock LSTM predictor")
    parser.add_argument('--demo', action='store_true',
                        help="run on bundled sample data (synthetic social "
                             "posts + cached real GME prices); no API "
                             "credentials needed")
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ticker = 'GME'

    if args.demo:
        print("=== DEMO MODE ===")
        print("Social-media data is SYNTHETIC sample data; prices are real "
              "GME history. Results demonstrate the pipeline only.")
        stock_data, reddit_data, twitter_data = _load_demo_data(here)
        run_pipeline(stock_data, reddit_data, twitter_data,
                     ticker=ticker, epochs=args.epochs,
                     plot_path=os.path.join(here, 'prediction_plot.png'))
    else:
        # LIVE mode: needs real Reddit API credentials in the environment
        # (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT)
        # and live Yahoo Finance access. Twitter scraping via snscrape is
        # effectively defunct; if it fails we continue Reddit-only.
        if os.environ.get('REDDIT_CLIENT_ID', 'YOUR_CLIENT_ID') == 'YOUR_CLIENT_ID':
            print("Live mode needs real Reddit API credentials.\n"
                  "Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET and "
                  "REDDIT_USER_AGENT, or run in demo mode:\n"
                  "    python Meme_Stock_Prediction_Model.py --demo")
            sys.exit(2)
        start_date = '2020-01-01'
        end_date = datetime.today().strftime('%Y-%m-%d')
        stock_data = get_stock_data(ticker, start_date, end_date)
        reddit_data = get_reddit_posts('wallstreetbets', 5000)
        try:
            twitter_data = get_tweets(f'${ticker}', 5000)
        except Exception as exc:
            print(f"WARNING: tweet collection failed ({exc}); "
                  "continuing with Reddit sentiment only.")
            twitter_data = pd.DataFrame(
                columns=['date', 'content', 'user', 'likes', 'retweets'])
        run_pipeline(stock_data, reddit_data, twitter_data,
                     ticker=ticker, epochs=args.epochs,
                     plot_path=os.path.join(here, 'prediction_plot.png'))




