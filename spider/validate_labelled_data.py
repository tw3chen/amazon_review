import pandas as pd


df = pd.read_csv("data/amazon_consumer_affairs_review/consumer_affairs_amazon_review_shuffled_labelled.csv")


def sentiment_proportion(sentiment, df):
    return df[df.dex_sentiment == sentiment].shape[0]/df.shape[0]


def print_sentiment_proportions(df):
    for sentiment in ['o', 'n', 'p']:
        print(sentiment_proportion(sentiment, df))
    print("\n")


def rating_proportion(rating, df):
    return df[df.review_rating == rating].shape[0]/df.shape[0]


def print_rating_proportions(df):
    for i in range(1,6):
        print(rating_proportion(i, df))
    print("\n")


labelled_df = df[~df.dex_sentiment.isnull()]
print("Total", df.shape[0])
print("Num labelled", labelled_df.shape[0])
print("Unique sentiments", labelled_df.dex_sentiment.unique())
print("Sentiment proportions for labelled df")
print_sentiment_proportions(labelled_df)
print("Rating proportions for labelled df")
print_rating_proportions(labelled_df)
print("Rating proportions for df")
print_rating_proportions(df)
