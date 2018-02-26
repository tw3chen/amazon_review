from nltk import corpus
import re
from review.dataset import TEXT_COLUMN_NAME


stopwords = set(corpus.stopwords.words('english'))


def clean_str(s):
    s = re.sub('[^a-zA-Z]+', ' ', s)
    s = s.lower()
    s = ' '.join([w for w in s.split(' ') if w != '' and w not in stopwords])
    return s


def clean_df(df, clean_str=False):
    df[TEXT_COLUMN_NAME].fillna("unknown", inplace=True)
    # commented out because cleaning in this manner actually gives worse results
    if clean_str:
        df[TEXT_COLUMN_NAME] = df[TEXT_COLUMN_NAME].map(lambda s: clean_str(s))
    return df


def clean_dataset(dataset, clean_str=False):
    dataset.X_train = clean_df(dataset.X_train, clean_str)
    dataset.X_validation = clean_df(dataset.X_validation, clean_str)
    dataset.X_test = clean_df(dataset.X_test, clean_str)
    return dataset
