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
                                                         test_size=50,
                                                         random_state=0,
                                                         stratify=df_labelled[Y_COLUMN])
df_labelled_train_augment, df_labelled_test = train_test_split(df_labelled_others,
                                                               test_size=25,
                                                               random_state=0,
                                                               stratify=df_labelled_others[Y_COLUMN])
df_unlabelled = df[df.dex_sentiment.isnull()]

X_labelled_test = df_labelled_test[[TEXT_COLUMN]]
y_labelled_test = df_labelled_test[Y_COLUMN]

X_labelled_train = df_labelled_train[[TEXT_COLUMN]]
y_labelled_train = df_labelled_train[Y_COLUMN]
classifier = LogisticRegressionClassifierWrapper()
classifier.fit(X_labelled_train, y_labelled_train)
score_labelled_train = classifier.score(X_labelled_train, y_labelled_train)
score_labelled_train_test = classifier.score(X_labelled_test, y_labelled_test)
print("Accuracy for base dataset trained on itself:", score_labelled_train)
print("Accuracy for base dataset trained on test dataset:", score_labelled_train_test)

X_labelled_train_augment = df_labelled_train_augment[[TEXT_COLUMN]]
y_labelled_train_augment = df_labelled_train_augment[Y_COLUMN]
X_labelled_train_augmented = pd.concat([X_labelled_train, X_labelled_train_augment], ignore_index=True)
y_labelled_train_augmented = pd.concat([y_labelled_train, y_labelled_train_augment], ignore_index=True)
classifier = LogisticRegressionClassifierWrapper()
classifier.fit(X_labelled_train_augmented, y_labelled_train_augmented)
score_labelled_train_augmented = classifier.score(X_labelled_train_augmented, y_labelled_train_augmented)
score_labelled_train_augmented_test = classifier.score(X_labelled_test, y_labelled_test)
print("Accuracy for base + augment dataset trained on itself:", score_labelled_train)
print("Accuracy for base + augment dataset trained on test dataset:", score_labelled_train_test)

X_unlabelled = df_unlabelled[[TEXT_COLUMN]]
prob_unlabelled_pred = classifier.predict_proba(X_unlabelled)
max_prob_unlabelled_pred = prob_unlabelled_pred.max(axis=1)
sorted_max_prob_unlabelled_pred = np.argsort(max_prob_unlabelled_pred)
df_to_label = df_unlabelled.iloc[sorted_max_prob_unlabelled_pred[:50]]
