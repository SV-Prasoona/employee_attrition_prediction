import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Drop unnecessary columns
    if "EmployeeID" in df.columns:
        df = df.drop("EmployeeID", axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    # Check missing values
    df = df.fillna(method="ffill")

    return df