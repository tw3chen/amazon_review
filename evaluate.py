from review.dataset import prepare_amazon_fine_food_review_dataset, \
    prepare_amazon_consumer_affairs_review_dataset
from review.feature import add_features_to_dataset
from review.clean import clean_dataset
from review.model import LogisticRegressionClassifierWrapper, \
    XGBoostClassifierWrapper, \
    DeepLearningClassifierWrapper, \
    save_model, load_model
from sklearn.model_selection import GridSearchCV
from time import time


model_path = 'model/dex/classifier.pickle'


start = time()
# dataset = prepare_amazon_fine_food_review_dataset(validation_proportion=0.15,
#                                                   test_proportion=0.15)
dataset = prepare_amazon_consumer_affairs_review_dataset(validation_proportion=0.1,
                                                         test_proportion=0.1)
dataset = clean_dataset(dataset)
# dataset = add_features_to_dataset(dataset)
print('Preparing dataset, cleaning dataset, and adding features took {0} seconds.'.
      format(time()-start))


start = time()
classifier = LogisticRegressionClassifierWrapper(tfidf_ngram_range=(1,1), lr_inverse_reg=2)
classifier.fit(dataset.X_train, dataset.y_train)
# load classifier from file instead of training
# classifier = load_model(model_path)
# score_validation = classifier.score(dataset.X_validation, dataset.y_validation)
# print('Validation Accuracy: ', score_validation)
# score_test = classifier.score(dataset.X_test, dataset.y_test)
# print('Test Accuracy: ', score_test)
# print('Evaluation took {0} seconds.'.format(time()-start))


start = time()
save_model(classifier, model_path)
print('Saving model took {0} seconds.'.format(time()-start))
