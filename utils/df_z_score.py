import pandas as pd


def z_score(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    for column in df.columns.tolist():
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df
