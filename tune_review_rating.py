from review.dataset import prepare_amazon_fine_food_review_dataset
from review.clean import clean_dataset
from review.model import LogisticRegressionClassifierWrapper, \
    XGBoostClassifierWrapper, \
    DeepLearningClassifierWrapper, \
    save_model, load_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from time import time


start = time()
dataset = prepare_amazon_fine_food_review_dataset(validation_proportion=0.15,
                                                  test_proportion=0.15)
dataset = clean_dataset(dataset)
print('Preparing dataset and cleaning dataset took {0} seconds.'.
      format(time()-start))


params_to_tune = {'lr_inverse_reg': [2, 4],
                  'lr_dual': [True, False],
                  'lr_fit_intercept': [True, False]}
classifier = LogisticRegressionClassifierWrapper()
# params_to_tune = {'xgb_num_tree': [10, 20],
#                   'xgb_max_depth': [3, 4]}
# classifier = XGBoostClassifierWrapper()


start = time()
classifier_with_grid_search = GridSearchCV(classifier,
                                           params_to_tune, cv=2, verbose=3)
classifier_with_grid_search.fit(dataset.X_train, dataset.y_train)
print('Found best params:', classifier_with_grid_search.best_params_)
print('Best Params:', classifier_with_grid_search.best_params_)
score_test = classifier_with_grid_search.score(dataset.X_test, dataset.y_test)
print('Test Accuracy: ', score_test)
y_test_pred = classifier_with_grid_search.predict(dataset.X_test)
mae_test = mean_absolute_error(dataset.y_test, y_test_pred)
print('Test Mean Absolute Error: ', mae_test)
print('Tuning and evaluation took {0} seconds.'.format(time()-start))


# classifier = DeepLearningClassifierWrapper(keras_model_path="model/rating/keras_deep_learning.model",
#                                            load_keras_model=True)
# classifier.fit(dataset.X_train, dataset.y_train)
# score_test = classifier.score(dataset.X_test, dataset.y_test)
# print('Test Accuracy: ', score_test)
# y_test_pred = classifier.predict(dataset.X_test)
# mae_test = mean_absolute_error(dataset.y_test, y_test_pred)
# print('Test Mean Absolute Error: ', mae_test)


start = time()
# save_model(classifier_with_grid_search, 'model/rating/classifier_with_grid_search_linear_regression.pickle')
# save_model(classifier_with_grid_search, 'model/rating/classifier_with_grid_search_xgboost.pickle')
# command below does not work... need to investigate pickle and keras
# save_model(classifier, 'model/rating/classifier_deep_learning.pickle')
print('Saving model took {0} seconds.'.format(time()-start))


# Linear Regression
# Best Params: {'lr_dual': True, 'lr_fit_intercept': True, 'lr_inverse_reg': 4}
# Test Accuracy:  0.763935310605
# Test Mean Absolute Error:  0.385356929247
# Tuning and evaluation took 975.6741416454315 seconds.

# XGBoost
# Best Params: {'xgb_max_depth': 4, 'xgb_num_tree': 20}
# Test Accuracy:  0.641862810635
# Test Mean Absolute Error:  0.808230423718
# Tuning and evaluation took 1833.391235589981 seconds.

# Deep Learning
# Test Accuracy:  0.811983252999
# Test Mean Absolute Error:  0.260223527894
