from review.dataset import prepare_amazon_consumer_affairs_review_dataset
from review.clean import clean_dataset
from review.model import LogisticRegressionClassifierWrapper, \
    XGBoostClassifierWrapper, \
    DeepLearningClassifierWrapper, \
    save_model, load_model
from sklearn.model_selection import GridSearchCV
from time import time


start = time()
dataset = prepare_amazon_consumer_affairs_review_dataset(validation_proportion=0,
                                                         test_proportion=0.1)
dataset = clean_dataset(dataset)
print('Preparing dataset and cleaning dataset took {0} seconds.'.
      format(time()-start))


params_to_tune = {'lr_inverse_reg': [2, 4],
                  'lr_dual': [True, False],
                  'lr_fit_intercept': [True, False],
                  'tfidf_ngram_range': [(1,1), (1,2)],
                  'lr_class_weight': ['balanced']}
classifier = LogisticRegressionClassifierWrapper()
# params_to_tune = {'svd_num_dim': [200, 300],
#                   'xgb_num_tree': [10, 100],
#                   'xgb_max_depth': [3, 4],
#                   'xgb_reg_lambda': [1, 2],
#                   'xgb_learning_rate': [0.1, 0.05],
#                   'xgb_min_child_weight': [1, 10]}
# classifier = XGBoostClassifierWrapper()


start = time()
classifier_with_grid_search = GridSearchCV(classifier,
                                           params_to_tune, cv=2, verbose=3)
classifier_with_grid_search.fit(dataset.X_train, dataset.y_train)
print('Found best params:', classifier_with_grid_search.best_params_)
print('Best Params:', classifier_with_grid_search.best_params_)
score_test = classifier_with_grid_search.score(dataset.X_test, dataset.y_test)
print('Test Accuracy: ', score_test)
print('Tuning and evaluation took {0} seconds.'.format(time()-start))


# classifier = DeepLearningClassifierWrapper(keras_model_path="model/dex/keras_deep_learning.model",
#                                            load_keras_model=False,
#                                            embedding_trainable=False)
# classifier.fit(dataset.X_train, dataset.y_train)
# score_test = classifier.score(dataset.X_test, dataset.y_test)
# print('Test Accuracy: ', score_test)


start = time()
# save_model(classifier_with_grid_search, 'model/dex/classifier_with_grid_search_linear_regression.pickle')
# save_model(classifier_with_grid_search, 'model/dex/classifier_with_grid_search_xgboost.pickle')
# command below does not work... need to investigate pickle and keras
# save_model(classifier, 'model/dex/classifier_deep_learning.pickle')
print('Saving model took {0} seconds.'.format(time()-start))


# Linear Regression
# Best Params: {'lr_dual': True, 'lr_fit_intercept': False, 'lr_inverse_reg': 2}
# Test Accuracy:  0.62
# Tuning and evaluation took 1.751326560974121 seconds.

# XGBoost
# Best Params: {'svd_num_dim': 200, 'xgb_learning_rate': 0.1,
# 'xgb_max_depth': 3, 'xgb_min_child_weight': 1,
# 'xgb_num_tree': 10, 'xgb_reg_lambda': 2}
# Test Accuracy:  0.62
# Tuning and evaluation took 64.03435730934143 seconds.

# Deep Learning
# Test Accuracy:  0.62
