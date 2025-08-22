import os

import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real

# Generate random data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_val, y_val = make_regression(n_samples=200, n_features=20, noise=0.1)

print(X.shape)
print(y.shape)


def tune_learning_rate_and_estimators(elr_X_train, elr_y_train, elr_cv, device_type):
    # Define search space
    search_space = {
        'learning_rate': Real(0.0001, 0.3),
        'n_estimators': Integer(50, 400),
    }

    # Set up the LightGBM model with the device type (CPU or GPU)
    lgb_model = lgb.LGBMRegressor(device=device_type, verbose=1)

    # Set up BayesSearchCV with the specified search space
    bayes_search = BayesSearchCV(estimator=lgb_model, search_spaces=search_space, scoring='neg_mean_absolute_error', cv=elr_cv, n_jobs=4, n_iter=50)

    # Perform the search to find the best learning rate and number of estimators
    bayes_search.fit(elr_X_train, elr_y_train)

    # Get the best learning rate and number of estimators
    best_learning_rate = bayes_search.best_params_['learning_rate']
    best_n_estimators = bayes_search.best_params_['n_estimators']
    print(f'Best learning rate: {best_learning_rate}')
    print(f'Best number of estimators: {best_n_estimators}')

    return best_learning_rate, best_n_estimators


# LightGBM
print("Running LightGBM model training...")
device_type = 'gpu'  # Use 'gpu' for GPU acceleration or 'cpu' for CPU-only mode

print("Tuning the learning rate and number of estimators...")
cv = KFold(n_splits=5, shuffle=True, random_state=1)
best_learning_rate, best_n_estimators = tune_learning_rate_and_estimators(X, y, cv, device_type)

lgb_model = lgb.LGBMRegressor(
    device=device_type,
    learning_rate=best_learning_rate,
    n_estimators=best_n_estimators,
    max_depth=3,
    subsample=0.8
)

# Train the model on training data
lgb_model.fit(
    X, y,
    eval_set=[(X_val, y_val)],
    verbose=1
)

# Save the trained LightGBM model
top_output_directory_regresion_models = "regression_models"
os.makedirs(top_output_directory_regresion_models, exist_ok=True)
lgb_model.booster_.save_model(os.path.join(top_output_directory_regresion_models, 'test_lgb_model_glom_openness.model'))
