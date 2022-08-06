import pandas as pd


class Load_data:

    @classmethod
    def load_csv(cls, path: str):
        df = pd.read_csv(path)
        return df
