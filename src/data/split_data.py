import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    input_path = "data/raw/raw.csv"
    output_dir = "data/processed"
    target_col = "silica_concentrate"

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Train-test split completed successfully.")


if __name__ == "__main__":
    main()
