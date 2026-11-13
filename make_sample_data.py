"""Generate the SYNTHETIC sample social-media data for --demo mode.

Writes sample_data/reddit_posts.csv and sample_data/twitter_posts.csv:
made-up GME-flavored posts/tweets spread over 2021-2023, in the same
column shapes the live collectors (get_reddit_posts / get_tweets)
return. They are synthetic -- written from title templates by this
script -- and exist so the demo pipeline (sentiment -> features -> LSTM)
can run without Reddit API credentials. The demo's price data is real
GME history (downloaded once from Yahoo Finance and cached to
sample_data/gme_prices.csv by the main script on first run).

Run:  python make_sample_data.py
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample_data")
rng = np.random.default_rng(7)

POS = ["GME to the moon, diamond hands forever",
       "Bullish on GME, apes together strong",
       "GME earnings beat was amazing, holding happily",
       "Love the GME turnaround story, buying more",
       "GME short squeeze round two incoming, glorious"]
NEG = ["GME is a terrible investment, bagholders beware",
       "Bearish GME puts printing, stock keeps falling sadly",
       "GME dilution fears make this a risky mess",
       "Selling my GME after another disappointing quarter",
       "GME hype is dead, worst trade of my year"]
NEU = ["GME daily discussion thread",
       "What is everyone doing with GME today",
       "GME options chain for this Friday",
       "Watching GME volume closely this week",
       "GME chart patterns and levels to watch"]

TWEETS_POS = ["$GME lets gooo, amazing run today",
              "$GME bullish breakout, love to see it",
              "Holding $GME strong, best community ever"]
TWEETS_NEG = ["$GME dumping hard, painful to watch",
              "$GME overvalued meme junk, sad",
              "Shorting $GME here, easy money hopefully"]
TWEETS_NEU = ["$GME volume picking up", "Watching $GME into close",
              "$GME earnings next week"]


def pick(pool_p, pool_n, pool_neu):
    r = rng.random()
    if r < 0.45:
        return str(rng.choice(pool_p))
    if r < 0.75:
        return str(rng.choice(pool_n))
    return str(rng.choice(pool_neu))


def main():
    os.makedirs(OUT, exist_ok=True)
    days = pd.bdate_range("2021-01-04", "2023-12-29")

    posts = []
    for i in range(300):
        day = days[int(rng.integers(0, len(days)))]
        ts = day + pd.Timedelta(hours=int(rng.integers(0, 24)),
                                minutes=int(rng.integers(0, 60)))
        posts.append([
            pick(POS, NEG, NEU), int(rng.integers(0, 5000)), f"demo{i:04d}",
            "wallstreetbets", "https://example.invalid/demo",
            int(rng.integers(0, 900)), "", ts.timestamp(),
        ])
    reddit = pd.DataFrame(posts, columns=["title", "score", "id", "subreddit",
                                          "url", "num_comments", "body", "created"])
    reddit.to_csv(os.path.join(OUT, "reddit_posts.csv"), index=False)

    tweets = []
    for i in range(300):
        day = days[int(rng.integers(0, len(days)))]
        ts = day + pd.Timedelta(hours=int(rng.integers(0, 24)),
                                minutes=int(rng.integers(0, 60)))
        tweets.append([ts.isoformat(), pick(TWEETS_POS, TWEETS_NEG, TWEETS_NEU),
                       f"demo_user_{i % 37}", int(rng.integers(0, 2000)),
                       int(rng.integers(0, 500))])
    twitter = pd.DataFrame(tweets, columns=["date", "content", "user", "likes", "retweets"])
    twitter.to_csv(os.path.join(OUT, "twitter_posts.csv"), index=False)

    print(f"Wrote sample_data/reddit_posts.csv : {len(reddit)} synthetic posts")
    print(f"Wrote sample_data/twitter_posts.csv: {len(twitter)} synthetic tweets")


if __name__ == "__main__":
    main()
