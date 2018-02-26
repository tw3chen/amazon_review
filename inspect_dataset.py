from review.dataset import prepare_amazon_fine_food_review_dataset
from sklearn.metrics import mean_absolute_error, accuracy_score


amazon_fine_food_review_dataset = prepare_amazon_fine_food_review_dataset(
    validation_proportion=0.15,
    test_proportion=0.15)
amazon_fine_food_review_reviews = amazon_fine_food_review_dataset.X_train.text
amazon_fine_food_review_labels = amazon_fine_food_review_dataset.y_train

# manual labelling for top 20 reviews
amazon_fine_food_review_manual_labels = [5, 5, 4, 5, 5, 1, 5, 5, 4, 5]
amazon_fine_food_review_accuracy = accuracy_score(
    amazon_fine_food_review_labels[:len(amazon_fine_food_review_manual_labels)],
    amazon_fine_food_review_manual_labels)
amazon_fine_food_review_mae = mean_absolute_error(
    amazon_fine_food_review_labels[:len(amazon_fine_food_review_manual_labels)],
    amazon_fine_food_review_manual_labels)
print('Amazon fine food review:')
print('Human accuracy: ', amazon_fine_food_review_accuracy)
print('Human MAE: ', amazon_fine_food_review_mae)
print('Dataset size: ', amazon_fine_food_review_dataset.X_train.shape[0] +
      amazon_fine_food_review_dataset.X_validation.shape[0] +
      amazon_fine_food_review_dataset.X_test.shape[0])
