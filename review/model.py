from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from review.dataset import TEXT_COLUMN_NAME
from scipy.sparse import hstack
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Dropout
from keras.preprocessing import text, sequence
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle


def save_model(model, model_path):
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


class LogisticRegressionClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, tfidf_ngram_range=(1,1), tfidf_min_df=3, tfidf_max_df=0.9, tfidf_sublinear_tf=True,
                 lr_inverse_reg=4):
        self.tfidf_vectorizer = None
        self.classifier = None
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_sublinear_tf = tfidf_sublinear_tf
        self.lr_inverse_reg = lr_inverse_reg

    def _construct_X_features(self, X):
        X_tfidf = self.tfidf_vectorizer.transform(X[TEXT_COLUMN_NAME])
        other_column_names = list(X.columns)
        other_column_names.remove(TEXT_COLUMN_NAME)
        if other_column_names:
            X_features = hstack([X_tfidf, X[other_column_names].values])
        else:
            X_features = X_tfidf
        return X_features

    def fit(self, X, y):
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=self.tfidf_ngram_range,
                                                min_df=self.tfidf_min_df, max_df=self.tfidf_max_df,
                                                strip_accents='unicode',
                                                use_idf=1, smooth_idf=1, sublinear_tf=self.tfidf_sublinear_tf)
        self.tfidf_vectorizer.fit(X[TEXT_COLUMN_NAME])
        X_features = self._construct_X_features(X)
        classifier = LogisticRegression(C=self.lr_inverse_reg, dual=True)
        classifier.fit(X_features, y)
        self.classifier = classifier
        return self

    def predict_proba(self, X):
        X_features = self._construct_X_features(X)
        return self.classifier.predict_proba(X_features)

    def predict(self, X):
        prob = self.predict_proba(X)
        vfunc = np.vectorize(lambda x: self.classifier.classes_[x])
        return vfunc(prob.argmax(axis=1))

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class XGBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, tfidf_ngram_range=(1,1), tfidf_min_df=3, tfidf_max_df=0.9, tfidf_sublinear_tf=True,
                 svd_num_dim=100,
                 xgb_num_tree=100, xgb_learning_rate=0.1, xgb_min_child_weight=1,
                 xgb_max_depth=3, xgb_reg_alpha=0, xgb_reg_lambda=1):
        self.tfidf_vectorizer = None
        self.dimension_reducer = None
        self.classifier = None
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_sublinear_tf = tfidf_sublinear_tf
        self.svd_num_dim = svd_num_dim
        self.xgb_num_tree = xgb_num_tree
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_min_child_weight = xgb_min_child_weight
        self.xgb_max_depth = xgb_max_depth
        self.xgb_reg_alpha = xgb_reg_alpha
        self.xgb_reg_lambda = xgb_reg_lambda

    def _construct_X_features(self, X):
        X_tfidf = self.tfidf_vectorizer.transform(X[TEXT_COLUMN_NAME])
        #X_reduced = X_tfidf
        X_reduced = self.dimension_reducer.transform(X_tfidf)
        other_column_names = list(X.columns)
        other_column_names.remove(TEXT_COLUMN_NAME)
        if other_column_names:
            X_features = hstack([X_reduced, X[other_column_names].values])
        else:
            X_features = X_reduced
        return X_features

    def fit(self, X, y):
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=self.tfidf_ngram_range,
                                                min_df=self.tfidf_min_df, max_df=self.tfidf_max_df,
                                                strip_accents='unicode',
                                                use_idf=1, smooth_idf=1, sublinear_tf=self.tfidf_sublinear_tf,
                                                max_features=100000, stop_words='english')
        self.dimension_reducer = TruncatedSVD(n_components=self.svd_num_dim)
        X_tfidf = self.tfidf_vectorizer.fit_transform(X[TEXT_COLUMN_NAME])
        #self.tfidf_vectorizer.fit(X[TEXT_COLUMN_NAME])
        self.dimension_reducer.fit(X_tfidf)
        X_features = self._construct_X_features(X)
        classifier = XGBClassifier(n_estimators=self.xgb_num_tree,
                                   min_child_weight=100,
                                   learning_rate=0.05,
                                   max_depth=5,
                                   reg_alpha=1,
                                   reg_lambda=4)
        classifier.fit(X_features, y)
        self.classifier = classifier
        return self

    def predict_proba(self, X):
        X_features = self._construct_X_features(X)
        return self.classifier.predict_proba(X_features)

    def predict(self, X):
        prob = self.predict_proba(X)
        vfunc = np.vectorize(lambda x: self.classifier.classes_[x])
        return vfunc(prob.argmax(axis=1))

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def define_bidirectional_lstm_attention_nn(max_sequence_length,
                                           num_classes,
                                           embedding_size, embedding_matrix):
    inputs = Input(shape=(max_sequence_length, ))
    embeddings = Embedding(len(embedding_matrix), embedding_size, weights=[embedding_matrix],
                           trainable=True)(inputs)
    bidirectional_lstm = Bidirectional(LSTM(embedding_size, return_sequences=True, dropout=0.25,
                                            recurrent_dropout=0.25))(embeddings)
    attention = Attention(max_sequence_length)(bidirectional_lstm)
    dense = Dense(256, activation='relu')(attention)
    dropout = Dropout(0.25)(dense)
    outputs = Dense(num_classes, activation='sigmoid')(dropout)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def make_glovevec(glovepath, max_features, embed_size, word_index):
    embeddings_index = {}
    f = open(glovepath)
    for line in f:
        values = line.split()
        word = ' '.join(values[:-embed_size])
        coefs = np.asarray(values[-embed_size:], dtype='float32')
        embeddings_index[word] = coefs.reshape(-1)
    f.close()
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words+1, embed_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def binarize_y(y, label_binarizer=None):
    if label_binarizer is None:
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(y)
    binarized_y = label_binarizer.transform(y)
    binarized_y = pd.DataFrame(binarized_y, columns=list(map(str, label_binarizer.classes_)))
    return binarized_y, label_binarizer


class DeepLearningClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, max_features_for_text=100000, embedding_size=300, max_text_length=300,
                 batch_size=256, epochs=10, validation_proportion=0.05,
                 keras_model_path="model/keras_deep_learning.model",
                 embedding_path="data/glove/glove.840B.300d.txt"):
        self.tokenizer = None
        self.embedding_matrix = None
        self.label_binarizer = None
        self.keras_model = None
        self.max_features_for_text = max_features_for_text
        self.embedding_size = embedding_size
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_proportion = validation_proportion
        self.keras_model_path = keras_model_path
        self.embedding_path = embedding_path

    def _construct_X_features(self, X):
        X_tokenized = self.tokenizer.texts_to_sequences(X[TEXT_COLUMN_NAME].values)
        X_padded = sequence.pad_sequences(X_tokenized, maxlen=self.max_text_length, padding='post', truncating='post')
        return X_padded

    def fit(self, X, y):
        tokenizer = text.Tokenizer(num_words=self.max_features_for_text)
        tokenizer.fit_on_texts(list(X[TEXT_COLUMN_NAME].values))
        self.tokenizer = tokenizer
        self.embedding_matrix = make_glovevec(self.embedding_path,
                                              self.max_features_for_text, self.embedding_size, tokenizer.word_index)
        binarized_y, self.label_binarizer = binarize_y(y, None)
        X_features = self._construct_X_features(X)
        keras_model = define_bidirectional_lstm_attention_nn(self.max_text_length,
                                                             binarized_y.shape[1],
                                                             self.embedding_size,
                                                             self.embedding_matrix)
        keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        ckpt = ModelCheckpoint(self.keras_model_path, monitor="val_loss", verbose=1,
                               save_best_only=True, mode="min")
        early = EarlyStopping(monitor="val_loss", mode="min", patience=1)
        keras_model.fit(X_features, binarized_y.values, batch_size=self.batch_size, epochs=self.epochs,
                        validation_split=self.validation_proportion,
                        callbacks=[ckpt, early])
        keras_model.load_weights(self.keras_model_path)
        self.keras_model = keras_model
        return self

    def predict_proba(self, X):
        X_features = self._construct_X_features(X)
        return self.keras_model.predict(X_features)

    def predict(self, X):
        prob = self.predict_proba(X)
        vfunc = np.vectorize(lambda x: self.label_binarizer.classes_[x])
        return vfunc(prob.argmax(axis=1))

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
