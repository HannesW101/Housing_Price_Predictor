import pandas as pd
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV
from Classes.GradientBoosting import GradientBoosting
from Classes.PreprocessData import PreprocessData
from Classes.RandomForest import RandomForest
from Classes.VisualizeData import VisualizeData


# Define a function to calculate mean absolute error during cross-validation
def custom_mae(y_true, y_predicted):
    return mean_absolute_error(y_true, y_predicted)


# Load your train and test datasets
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')
y_train = train_data['SalePrice']  # separate target variable from training data

# Visualize the sales price against frequency from training data
visualize = VisualizeData(train_data)
visualize.load_data()
visualize.plot_distribution()

# Preprocess the training and testing data
preprocessor = PreprocessData(train_data, test_data)
x_train_processed = preprocessor.process_x_train()
x_test_processed = preprocessor.process_x_test()
print("Preprocessing of data complete.")

'''
# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring=make_scorer(custom_mae), n_jobs=-1)
grid_search.fit(x_train_processed, y_train)

# Get the best model and its hyperparameters
best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Train the best model on the full training data
best_rf_model.fit(x_train_processed, y_train)
print("Random Forest Model training complete with best parameters.")
rf_predictions = best_rf_model.predict(x_test_processed)
print("Random Forest Model Predictions complete")
'''

# Initialize the RandomForest model
rf_model = RandomForest(n_estimators=100, max_depth=None, min_samples_split=2)

# train the random forest model and predict future sales prices
rf_model.fit(x_train_processed, y_train)
print("Random Forest Model training complete.")
rf_predictions = rf_model.predict(x_test_processed)
print("Random Forest Model Predictions complete")

# Visualize the predicted sales price against frequency from Random Forest model
visualize.plot_predictions_without_actuals(predictions=rf_predictions, title='Random Forest Predictions')

# Perform cross-validation with mean absolute error scoring for Random Forest model
mae_scores_rf = cross_val_score(rf_model, x_train_processed, y_train, cv=5, scoring=make_scorer(custom_mae), n_jobs=-1)
# Calculate the mean MAE across all folds
mean_mae_rf = mae_scores_rf.mean()
print(f"Random Forest Model Mean absolute error: {mean_mae_rf}")

'''
# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6]
}

# Initialize the GradientBoosting model
gb_model = GradientBoosting()

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring=make_scorer(custom_mae), n_jobs=-1)
grid_search.fit(x_train_processed, y_train)

# Get the best model and its hyperparameters
best_gb_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Train the best model on the full training data
best_gb_model.fit(x_train_processed, y_train)
print("Gradient Boosting Model training complete with best parameters.")
gb_predictions = best_gb_model.predict(x_test_processed)
print("Gradient Boosting Model Predictions complete")
'''

# Initialize the GradientBoosting model
gb_model = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the Gradient Boosting model and predict future sales prices
gb_model.fit(x_train_processed, y_train)
print("Gradient Boosting Model training complete.")
gb_predictions = gb_model.predict(x_test_processed)
print("Gradient Boosting Model Predictions complete")

# Visualize the predicted sales price against frequency from Gradient Boosting model
visualize.plot_predictions_without_actuals(predictions=gb_predictions, title='Gradient Boosting Predictions')

# Perform cross-validation with mean absolute error scoring for Gradient Boosting model
mae_scores_gb = cross_val_score(gb_model, x_train_processed, y_train, cv=5, scoring=make_scorer(custom_mae), n_jobs=-1)

# Calculate the mean MAE across all folds
mean_mae_gb = mae_scores_gb.mean()
print(f"Gradient Boosting Model Mean absolute error: {mean_mae_gb}")

# Create a DataFrame with the results obtained from random forest model
final_predictions = pd.DataFrame({'Id': pd.read_csv('Data/test.csv')['Id'], 'SalePrice': rf_predictions})
# Save the results to a CSV file
final_predictions.to_csv('Data/final_predictions.csv', index=False)
