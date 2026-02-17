from __future__ import annotations

from pathlib import Path
import pandas as pd


def ohlcv_resample(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df_1m.resample(rule).agg(agg).dropna()
    return out


def main() -> None:
    in_path = Path("data/raw/binanceusdm_ETHUSDT_1m.parquet")
    out_15m = Path("data/raw/binanceusdm_ETHUSDT_15m.parquet")
    out_1h = Path("data/raw/binanceusdm_ETHUSDT_1h.parquet")

    df = pd.read_parquet(in_path).sort_index()

    df_15m = ohlcv_resample(df, "15min")
    df_1h = ohlcv_resample(df, "1h")

    out_15m.parent.mkdir(parents=True, exist_ok=True)
    df_15m.to_parquet(out_15m)
    df_1h.to_parquet(out_1h)

    print("Saved resampled data:")
    print(f"  15m: {len(df_15m):,} rows -> {out_15m}")
    print(f"  1h : {len(df_1h):,} rows -> {out_1h}")
    print("\n15m tail:")
    print(df_15m.tail())


if __name__ == "__main__":
    main()
