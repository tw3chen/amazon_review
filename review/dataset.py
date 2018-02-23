import pandas as pd
from sklearn.model_selection import train_test_split


TEXT_COLUMN_NAME = 'text'


class Dataset():
    def __init__(self,
                 X_train, y_train,
                 X_validation, y_validation,
                 X_test, y_test,
                 auxiliay_dict={}):
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.X_test = X_test
        self.y_test = y_test
        self.auxiliary_dict = auxiliay_dict


def train_validation_test_split(X, y, validation_proportion, test_proportion):
    X_train_validation, X_test, \
    y_train_validation, y_test = train_test_split(X, y,
                                      test_size=test_proportion,
                                      random_state=1)
    X_train, X_validation, \
    y_train, y_validation = train_test_split(X_train_validation, y_train_validation,
                                                  test_size=validation_proportion/(1-test_proportion),
                                                  random_state=1)
    for df in [X_train, y_train, X_validation, y_validation, X_test, y_test]:
        df.reset_index(inplace=True, drop=True)
    return X_train, y_train, \
           X_validation, y_validation, \
           X_test, y_test


def prepare_amazon_fine_food_review_dataset(validation_proportion=0.15, test_proportion=0.15):
    raw_df = pd.read_csv('data/amazon_fine_food_review/Reviews.csv')
    raw_df = raw_df.sample(frac=0.05)
    X = raw_df[['Text']]
    X.columns = [TEXT_COLUMN_NAME]
    y = raw_df['Score']
    X_train, y_train, \
    X_validation, y_validation, \
    X_test, y_test = train_validation_test_split(X, y, validation_proportion, test_proportion)
    dataset = Dataset(X_train, y_train,
                      X_validation, y_validation,
                      X_test, y_test)
    return dataset
