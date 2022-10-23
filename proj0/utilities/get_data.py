from pathlib import Path

import pandas as pd
import ccxt


def get_historical_from_db(exchange, symbol, timeframe, path=Path("database")):
    symbol = symbol.replace("/", "-")
    df = pd.read_csv(
        filepath_or_buffer=path / str(exchange.name) / timeframe / f"{symbol}.csv"
    )
    df = df.set_index(df["date"])
    df.index = pd.to_datetime(df.index, unit="ms")
    del df["date"]
    return df


def get_historical_from_path(path):
    df = pd.read_csv(filepath_or_buffer=path)
    df = df.set_index(df["date"])
    df.index = pd.to_datetime(df.index, unit="ms")
    del df["date"]
    return df
