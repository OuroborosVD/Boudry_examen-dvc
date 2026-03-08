import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def main():

    input_dir = "data/processed"
    model_dir = "models"

    os.makedirs(model_dir, exist_ok=True)

    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv")

    y_train = y_train.values.ravel()

    model = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_params = grid.best_params_

    joblib.dump(best_params, f"{model_dir}/best_params.pkl")

    print("Best parameters found:")
    print(best_params)


if __name__ == "__main__":
    main()
