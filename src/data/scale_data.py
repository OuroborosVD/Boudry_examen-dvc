import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():

    input_dir = "data/processed"
    output_dir = "data/processed"

    os.makedirs(output_dir, exist_ok=True)

    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")

    # Suppression de la colonne date si elle existe
    if "date" in X_train.columns:
        X_train = X_train.drop(columns=["date"])
        X_test = X_test.drop(columns=["date"])

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled.to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{output_dir}/X_test_scaled.csv", index=False)

    print("Scaling completed successfully")


if __name__ == "__main__":
    main()
