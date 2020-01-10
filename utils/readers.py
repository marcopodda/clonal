import pandas as pd

__all__ = ["read_csv", "read_tsv", "write_csv", "write_tsv",
           "write_pickle", "read_pickle", "dataframe_from_cols"]


def read_csv(path):
    return pd.read_csv(path)


def read_tsv(path):
    return pd.read_csv(path, sep="\t")


def write_csv(df, path):
    df.to_csv(path, index=False)


def write_tsv(df, path):
    df.to_csv(path, sep="\t", index=False)


def read_pickle(path):
    return pd.read_pickle(path)


def write_pickle(df, path):
    df.to_pickle(path)


def dataframe_from_cols(cols):
    return pd.DataFrame(columns=cols)
