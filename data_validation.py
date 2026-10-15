import pandas as pd
import os


DATA_PATH = "data/phishing.csv"


def validate_dataset(file_path):
    print("=" * 50)
    print("        DATA VALIDATION")
    print("=" * 50)

    # Check if file exists
    if not os.path.exists(file_path):
        print("Dataset not found:", file_path)
        return False

    print("Dataset found")

    # Load dataset
    df = pd.read_csv(file_path)

    # Basic information
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Check number of columns
    if df.shape[1] != 31:
        print("Expected 31 columns")
        return False

    print("Column count is correct")

    # Check target column
    if "Result" not in df.columns:
        print("Target column 'Result' not found")
        return False

    print("Target column 'Result' found")

    # Check completely empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()

    if empty_columns:
        print("Completely empty columns:", empty_columns)
        return False

    print("No completely empty columns")

    # Check target values
    valid_targets = {-1, 1}
    actual_targets = set(df["Result"].dropna().unique())

    if not actual_targets.issubset(valid_targets):
        print("Invalid target values:", actual_targets)
        return False

    print("Target values are valid:", actual_targets)

    # Check missing target values
    if df["Result"].isnull().any():
        print("Missing values found in target column")
        return False

    print("No missing target values")

    print("\n" + "=" * 50)
    print("DATA VALIDATION PASSED")
    print("=" * 50)

    return True


if __name__ == "__main__":
    validate_dataset(DATA_PATH)