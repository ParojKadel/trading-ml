from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd


@dataclass(frozen=True)
class FetchConfig:
    exchange_id: str = "binanceusdm"          # Binance USDT-M futures (perps)
    symbol: str = "ETH/USDT:USDT"             # ETHUSDT perpetual
    timeframe: str = "1m"
    limit: int = 1500                        # Binance max per request is typically 1500
    out_path: str = "data/raw/binanceusdm_ETHUSDT_1m.parquet"


def _mk_exchange(exchange_id: str) -> ccxt.Exchange:
    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})
    return ex


def fetch_history_minutes(cfg: FetchConfig, minutes: int, sleep_s: float = 0.25) -> pd.DataFrame:
    """
    Fetch approx last `minutes` candles.
    Returns a UTC-indexed DataFrame with columns: open, high, low, close, volume
    """
    ex = _mk_exchange(cfg.exchange_id)
    ex.load_markets()

    now_ms = ex.milliseconds()
    since_ms = now_ms - minutes * 60_000

    all_rows: list[list] = []
    cursor = since_ms
    last_cursor: Optional[int] = None

    while True:
        rows = ex.fetch_ohlcv(cfg.symbol, timeframe=cfg.timeframe, since=cursor, limit=cfg.limit)
        if not rows:
            break

        all_rows.extend(rows)
        cursor = rows[-1][0] + 1  # move forward (avoid duplicates)

        # safety: stop if no progress
        if last_cursor is not None and cursor <= last_cursor:
            break
        last_cursor = cursor

        if cursor >= now_ms:
            break

        time.sleep(sleep_s)

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        # return empty but well-formed dataframe
        out = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return out

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)
    df = df.loc[df.index >= cutoff]

    return df


def main() -> None:
    cfg = FetchConfig()
    Path(cfg.out_path).parent.mkdir(parents=True, exist_ok=True)

    # 30 days of 1m data
    minutes = 30 * 24 * 60
    df = fetch_history_minutes(cfg, minutes=minutes)

    df.to_parquet(cfg.out_path)
    print(f"Saved {len(df):,} rows to {cfg.out_path}")
    if len(df) > 0:
        print(df.tail())


if __name__ == "__main__":
    main()
