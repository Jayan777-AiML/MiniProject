import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


INPUT_PATH = "data/phishing.csv"

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"


def transform_data():

    print("=" * 50)
    print("       DATA TRANSFORMATION")
    print("=" * 50)

    # 1. Load dataset
    df = pd.read_csv(INPUT_PATH)

    print(f"Dataset loaded: {df.shape}")

    # 2. Separate features and target
    X = df.drop(columns=["Result"])
    y = df["Result"]

    print(f"Features: {X.shape[1]}")
    print("Target: Result")

    # 3. Convert target
    # -1 = Legitimate → 0
    #  1 = Phishing   → 1
    y = y.map({-1: 0, 1: 1})

    print("Target converted: -1/1 → 0/1")

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")

    # 5. Handle missing values
    imputer = SimpleImputer(strategy="most_frequent")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns
    )

    print("Missing values handled")

    # 6. Save transformed datasets
    train_data = X_train.copy()
    train_data["Result"] = y_train.values

    test_data = X_test.copy()
    test_data["Result"] = y_test.values

    train_data.to_csv(TRAIN_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)

    print("Training data saved:", TRAIN_PATH)
    print("Testing data saved:", TEST_PATH)

    print("\n" + "=" * 50)
    print("DATA TRANSFORMATION COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    transform_data()