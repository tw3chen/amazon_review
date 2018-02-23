from review.dataset import prepare_amazon_fine_food_review_dataset
from review.feature import add_features_to_dataset
from review.clean import clean_dataset
from review.model import LogisticRegressionClassifierWrapper, \
    XGBoostClassifierWrapper, \
    DeepLearningClassifierWrapper, \
    save_model, load_model
from sklearn.model_selection import GridSearchCV
from time import time


start = time()
dataset = prepare_amazon_fine_food_review_dataset(validation_proportion=0.15,
                                                  test_proportion=0.15)
dataset = clean_dataset(dataset)
# dataset = add_features_to_dataset(dataset)
print('Preparing dataset, cleaning dataset, and adding features took {0} seconds.'.
      format(time()-start))


params_to_tune = {'tfidf_sublinear_tf': [True, False],
                  'lr_inverse_reg': [2, 4]}
classifier = LogisticRegressionClassifierWrapper()


start = time()
classifier_with_grid_search = GridSearchCV(classifier,
                                           params_to_tune, cv=2, verbose=3)
classifier_with_grid_search.fit(dataset.X_train, dataset.y_train)
print('Found best params:', classifier_with_grid_search.best_params_)
y_validation_pred = classifier_with_grid_search.predict(dataset.X_validation)
print('Best Params:', classifier_with_grid_search.best_params_)
score_validation = classifier_with_grid_search.score(dataset.X_validation, dataset.y_validation)
print('Validation Accuracy: ', score_validation)
score_test = classifier_with_grid_search.score(dataset.X_test, dataset.y_test)
print('Test Accuracy: ', score_test)
print('Tuning and evaluation took {0} seconds.'.format(time()-start))


start = time()
save_model(classifier_with_grid_search, 'model/classifier_with_grid_search.pickle')
print('Saving model took {0} seconds.'.format(time()-start))
