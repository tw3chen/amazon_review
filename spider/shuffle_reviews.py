import pandas as pd


df = pd.read_csv("consumer_affairs_amazon_review.csv")


def rating_proportion(rating, df):
    return df[df.review_rating == rating].shape[0]/df.shape[0]


def print_rating_proportions(df):
    print("\n")
    for i in range(1,6):
        print(rating_proportion(i, df))
    print("\n")


print_rating_proportions(df)
print_rating_proportions(df.head(n=450))
# ensure that the class distribution for the labelled data is roughly the same as that of the unlabelled data
shuffled_df = df.sample(frac=1).reset_index(drop=True)
print_rating_proportions(shuffled_df.head(n=450))


shuffled_df.to_csv("consumer_affairs_amazon_review_shuffled.csv", index=False)
