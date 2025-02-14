import pandas as pd
import numpy as np
from itertools import product

from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                  ElasticNet)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------
# 1. Base CSV data
# ------------------------------------------------------------------------


df_base = pd.read_csv('Housing.csv')

# Convert yes/no columns into 1/0
binary_cols = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    df_base[col] = df_base[col].map({"yes": 1, "no": 0})

# One-hot encoding for furnishingstatus (drop_first to avoid dummy variable trap)
df_base = pd.get_dummies(df_base, columns=["furnishingstatus"], drop_first=True)

# ------------------------------------------------------------------------
# 2. Data Augmentation Function
# ------------------------------------------------------------------------
def augment_data(df, size=10, noise_std=0.02):
    """
    Replicate the rows of df 'size' times, adding random noise
    to numeric columns. noise_std is the standard deviation
    (fractional) of the noise.
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Repeat the data
    df_aug = pd.concat([df]*size, ignore_index=True)

    # For each numeric column, add small random noise
    for col in numeric_cols:
        df_aug[col] *= (1 + np.random.randn(len(df_aug)) * noise_std)

    return df_aug

# ------------------------------------------------------------------------
# 3. Outlier Removal Strategies
# ------------------------------------------------------------------------
def remove_outliers_zscore(df, cols, threshold=3.0):
    """
    Removes rows where any specified column has a z-score
    beyond the given threshold.
    """
    df_out = df.copy()
    for col in cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        # If no variation, skip
        if col_std == 0:
            continue
        z_scores = (df[col] - col_mean) / col_std
        mask = np.abs(z_scores) < threshold
        df_out = df_out[mask]
    return df_out

def remove_outliers_iqr(df, cols, k=1.5):
    """
    Removes rows where any specified column is beyond
    [Q1 - k*(IQR), Q3 + k*(IQR)].
    """
    df_out = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        mask = (df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)
        df_out = df_out[mask]
    return df_out

# ------------------------------------------------------------------------
# 4. Define Models and Hyperparameters
# ------------------------------------------------------------------------
models_config = {
    "LinearRegression": {
        "model": LinearRegression,
        "params": {
            "fit_intercept": [True, False]
        }
    },
    "Ridge": {
        "model": Ridge,
        "params": {
            "alpha": [0.01, 0.1, 1, 10],
            "fit_intercept": [True, False]
        }
    },
    "Lasso": {
        "model": Lasso,
        "params": {
            "alpha": [0.01, 0.1, 1, 10],
            "fit_intercept": [True, False]
        }
    },
    "ElasticNet": {
        "model": ElasticNet,
        "params": {
            "alpha": [0.01, 0.1, 1],
            "l1_ratio": [0.3, 0.7],
            "fit_intercept": [True, False]
        }
    },
    "SVR": {
        "model": SVR,
        "params": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10]
        }
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor,
        "params": {
            "n_estimators": [10, 50],
            "max_depth": [ 5]
        }
    },
    "GradientBoostingRegressor": {
        "model": GradientBoostingRegressor,
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1]
        }
    }
}

# ------------------------------------------------------------------------
# 5. Multiple Configurations:
#    - Different augmentation sizes
#    - Different outlier removal strategies
#    - Multiple models and hyperparameters
# ------------------------------------------------------------------------

outlier_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]

outlier_strategies = [
    ("no_outlier_removal", None, {}),
    ("zscore_thr_2", remove_outliers_zscore, {"threshold": 2.0}),
    ("zscore_thr_3", remove_outliers_zscore, {"threshold": 3.0}),
    ("iqr_k_1.5", remove_outliers_iqr, {"k": 1.5}),
    ("iqr_k_3.0", remove_outliers_iqr, {"k": 3.0})
]

augmentation_sizes = [1]  # Example sizes

for aug_size in augmentation_sizes:
    # 1) Create augmented dataset
    #df_aug = augment_data(df_base, size=aug_size, noise_std=0.02)
    df_aug = df_base
    for outlier_name, outlier_func, outlier_params in outlier_strategies:
        # 2) Possibly remove outliers
        if outlier_func is not None:
            df_clean = outlier_func(df_aug, outlier_cols, **outlier_params)
        else:
            df_clean = df_aug.copy()

        # If too few rows remain, skip
        if len(df_clean) < 2:
            print(f"Skipping (aug_size={aug_size}, outlier={outlier_name}) - too few rows.")
            continue

        # Separate target and features
        y = df_clean["price"]
        X = df_clean.drop("price", axis=1)

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 3) For each model, try all hyperparameter combos
        for model_name, config in models_config.items():
            ModelClass = config["model"]
            params_dict = config["params"]

            # Create all combinations of parameters using product
            param_keys = list(params_dict.keys())
            param_values = list(params_dict.values())  # each is a list of possible values

            # If no params in dictionary, just do a single iteration
            if not param_keys:
                # e.g., if a model had no hyperparameters
                model = ModelClass()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                print(f"[aug_size={aug_size}, outlier={outlier_name}] Model={model_name}")
                print(f"  (No hyperparams)")
                print(f"  Rows used: {len(df_clean)}")
                print(f"  MAE: {mae:.2f} | R^2: {r2:.2f}")
                print("-"*70)
                continue

            for combination in product(*param_values):
                # Build the parameter dictionary
                current_params = dict(zip(param_keys, combination))

                # Initialize and train the model
                model = ModelClass(**current_params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # Evaluate
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                print(f"[aug_size={aug_size}, outlier={outlier_name}] Model={model_name}")
                print(f"  Params: {current_params}")
                print(f"  Rows used: {len(df_clean)}")
                print(f"  MAE: {mae:.2f} | R^2: {r2:.2f}")
                print("-"*70)
