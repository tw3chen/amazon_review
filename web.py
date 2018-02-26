from flask import Flask, render_template, request
from review.model import load_model
from review.clean import clean_dataset
import pandas as pd
from review.dataset import TEXT_COLUMN_NAME, Dataset, prepare_amazon_fine_food_review_dataset
from review.model import DeepLearningClassifierWrapper


app = Flask(__name__)


dex_model = load_model('model/dex/classifier.pickle')
rating_model = DeepLearningClassifierWrapper(keras_model_path="model/rating/keras_deep_learning.model",
                                             load_keras_model=True)
dataset = prepare_amazon_fine_food_review_dataset(validation_proportion=0.15,
                                                  test_proportion=0.15)
dataset = clean_dataset(dataset)
rating_model.fit(dataset.X_train, dataset.y_train)


def prepare_dataset(review_text):
    df = pd.DataFrame([{TEXT_COLUMN_NAME: review_text}])
    X = df[[TEXT_COLUMN_NAME]]
    X_test = X
    y_train = y_validation = y_test = None
    X_train = X_validation = X.sample(frac=0)
    dataset = Dataset(X_train, y_train,
                      X_validation, y_validation,
                      X_test, y_test)
    return dataset


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/classify/", methods=['POST'])
def classify():
    data = dict(request.form)
    review_text = data['review-text'][0]
    print(review_text)
    dataset = prepare_dataset(review_text)
    dataset = clean_dataset(dataset)
    print(dex_model.predict_proba(dataset.X_test))
    predictions = dex_model.predict(dataset.X_test)
    rating_predictions = rating_model.predict(dataset.X_test)
    print(dataset.X_test)
    raw_pred = predictions[0]
    print(predictions)
    raw_rating_pred = rating_predictions[0]
    print(raw_rating_pred)
    dex_raw_pred_to_pred = {'n': 'Negative delivery experience',
                            'p': 'Positive delivery experience',
                            'o': 'Neutral delivery experience'}
    review_rating = raw_rating_pred
    result = {'review_rating': 'Rating: {0}'.format(review_rating),
              'review_dex': dex_raw_pred_to_pred[raw_pred],
              'review_text': review_text}
    return render_template('index.html', result=result)


app.run(
    host="0.0.0.0",
    port='50005',
    debug=False,
    threaded=False,
    use_reloader=False # under debug mode, app.py is initialized twice without this parameter set to False
)
