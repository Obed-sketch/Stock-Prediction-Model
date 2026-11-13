# Stock-Prediction-Model

An LSTM model that combines Yahoo Finance price data with Reddit/Twitter
sentiment to predict a meme-stock ticker (default: GME).

## Files

- `Meme_Stock_Prediction_Model.py` — the main script: data collection
  (yfinance, PRAW, snscrape), TextBlob sentiment, RSI/MACD features, an LSTM
  trained for 50 epochs, and a results plot.
- `enhanced meme stock prediction model.py` — a **continuation fragment**,
  not a standalone script: cross-validation, a backtester and a Flask
  deployment section that build on names and saved artifacts from the main
  file. See the header comment in the file.

## How to run

```bash
pip install -r requirements.txt
python Meme_Stock_Prediction_Model.py
```

Requirements in practice: live internet access (Yahoo Finance, Twitter
scraping), **real Reddit API credentials** in `get_reddit_posts` (the values
in the file are placeholders), and TensorFlow. A full run trains an LSTM on
several years of daily data and takes a long time; neither file is a quick
demo.
