import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor


def main():

    input_dir = "data/processed"
    model_dir = "models"

    os.makedirs(model_dir, exist_ok=True)

    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv")

    y_train = y_train.values.ravel()

    best_params = joblib.load(f"{model_dir}/best_params.pkl")

    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, f"{model_dir}/trained_model.pkl")

    print("Model trained and saved successfully")


if __name__ == "__main__":
    main()
