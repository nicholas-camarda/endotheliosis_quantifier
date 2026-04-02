
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold

# Function definition


def tune_learning_rate(elr_X_train, elr_y_train, elr_cv, gpu_dict):
    # Define search space
    param_grid = {'eta': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3], }

    # Set up the lightGBM model with the gpu_dict parameters
    xgb_model = xgb.XGBRegressor(**gpu_dict, logging_level=logging_level)

    # Set up GridSearchCV with the specified search space
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=elr_cv, n_jobs=4)

    # Perform the grid search to find the best learning rate
    grid_search.fit(elr_X_train, elr_y_train)

    # Get the best learning rate
    best_learning_rate = grid_search.best_params_['eta']
    print(f'Best learning rate: {best_learning_rate}')

    return best_learning_rate


def tune_n_estimators(elr_X_train, elr_y_train, elr_learning_rate, elr_cv, gpu_dict):
    # Define search space
    param_grid = {
        'n_estimators': range(50, 400, 50),
    }

    # Set up the lightGBM model with the gpu_dict parameters and the best learning rate
    xgb_model = xgb.XGBRegressor(**gpu_dict, eta=elr_learning_rate, logging_level=logging_level)

    # Set up GridSearchCV with the specified search space
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=elr_cv, n_jobs=4)

    # Perform the grid search to find the best number of estimators
    grid_search.fit(elr_X_train, elr_y_train)

    # Get the best number of estimators
    best_n_estimators = grid_search.best_params_['n_estimators']
    print(f'Best number of estimators: {best_n_estimators}')

    return best_n_estimators

# Create random input data and labels
np.random.seed(42)
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100)
logging_level='debug'

# Cross-validation splitter
cv = KFold(n_splits=5)

# GPU dictionary (update the values according to your setup)
gpu_dict = {
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
}

# Call the function with the generated data
best_learning_rate = tune_learning_rate(X_train, y_train, cv, gpu_dict)
