import os
import json
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, r2_score


def main():

    input_dir = "data/processed"
    model_dir = "models"
    metrics_dir = "metrics"

    os.makedirs(metrics_dir, exist_ok=True)

    X_test = pd.read_csv(f"{input_dir}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{input_dir}/y_test.csv")

    y_test = y_test.values.ravel()

    model = joblib.load(f"{model_dir}/trained_model.pkl")

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    predictions_df = pd.DataFrame({
        "prediction": predictions
    })

    predictions_df.to_csv(f"{input_dir}/predictions.csv", index=False)

    scores = {
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

    with open(f"{metrics_dir}/scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    print("Evaluation completed successfully")
    print(scores)


if __name__ == "__main__":
    main()
