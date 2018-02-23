import numpy as np
import pandas as pd
from review.model import LogisticRegressionClassifierWrapper
from sklearn.model_selection import train_test_split

TEXT_COLUMN = 'text'
Y_COLUMN = 'dex_sentiment'

df = pd.read_csv('data/amazon_consumer_affairs_review/consumer_affairs_amazon_review_shuffled_labelled.csv')
df = df[['review_text', 'dex_sentiment']]
df.columns = [TEXT_COLUMN, Y_COLUMN]

df_labelled = df[~df.dex_sentiment.isnull()]
df_labelled_train, df_labelled_others = train_test_split(df_labelled,
                                                         test_size=100,
                                                         random_state=1,
                                                         stratify=df_labelled[Y_COLUMN])
df_labelled_train_augment, df_labelled_test = train_test_split(df_labelled_others,
                                                               test_size=100,
                                                               random_state=1,
                                                               stratify=df_labelled_others[Y_COLUMN])

# df_labelled_augment, df_labelled_
# df_unlabelled = df[df.dex_sentiment.isnull()]
#
# classifier = LogisticRegressionClassifierWrapper()
#
# X_labelled_train = df_labelled_train[['text']]
# y_labelled_train = df_labelled_train['dex_sentiment']
# classifier.fit(X_labelled_train, y_labelled_train)
# score_train = classifier.score(X_labelled_train, y_labelled_train)

# X_unlabelled = df_unlabelled[['text']]
# prob_unlabelled_pred = classifier.predict_proba(X_unlabelled)
# max_prob_unlabelled_pred = prob_unlabelled_pred.max(axis=1)
# sorted_max_prob_unlabelled_pred = np.argsort(max_prob_unlabelled_pred)
# print(max_prob_unlabelled_pred[sorted_max_prob_unlabelled_pred])
#
# df_to_label = df_unlabelled.iloc[sorted_max_prob_unlabelled_pred[:50]]
