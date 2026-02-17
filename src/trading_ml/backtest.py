from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .features import build_multitimeframe_features


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def simple_backtest_hold(
    df_15m: pd.DataFrame,
    proba_up: pd.Series,
    horizon_bars: int = 4,        # 4 x 15m = 1 hour
    fee_bps: float = 4.0,         # rough placeholder
    slippage_bps: float = 2.0,    # rough placeholder
    long_th: float = 0.55,
    short_th: float = 0.45,
) -> pd.DataFrame:
    """
    1h-consistent backtest (Option A):
      - Decide at time t based on proba_up[t]
      - If signal triggers, enter and HOLD for `horizon_bars`
      - PnL is paid over close(t) -> close(t+horizon_bars)
      - Fees/slippage applied on entry and exit (2 turns total) per trade
      - Equity is tracked on the 15m grid for visibility (returns realized at exit bar)
    """
    close = df_15m["close"].reindex(proba_up.index).astype("float64")

    # Future return over the horizon: ret_h[t] = close[t+H]/close[t] - 1
    ret_h = (close.shift(-horizon_bars) / close) - 1.0

    idx = proba_up.index
    n = len(idx)

    # Position series on 15m grid (mostly flat, held during trades)
    pos = pd.Series(0.0, index=idx)

    # We'll realize PnL at the exit bar (t+H), not every bar.
    strat_ret = pd.Series(0.0, index=idx)

    cost = (fee_bps + slippage_bps) / 10_000.0

    i = 0
    trades = []  # list of dicts with entry/exit info

    while i < n:
        p = float(proba_up.iloc[i])

        # decide signal at time i
        signal = 0.0
        if p > long_th:
            signal = 1.0
        elif p < short_th:
            signal = -1.0

        if signal == 0.0:
            i += 1
            continue

        entry_i = i
        exit_i = i + horizon_bars

        # if we don't have enough future bars to exit, stop
        if exit_i >= n:
            break

        # mark held position on the grid for reporting/exposure
        pos.iloc[entry_i:exit_i] = signal

        # realized return for the trade paid at exit bar
        gross = float(signal * ret_h.iloc[entry_i])

        # costs: entry + exit = 2 turns (0->pos, pos->0)
        tc = 2.0 * cost
        net = gross - tc

        strat_ret.iloc[exit_i] += net

        trades.append(
            {
                "entry_ts": idx[entry_i],
                "exit_ts": idx[exit_i],
                "side": "LONG" if signal > 0 else "SHORT",
                "proba": p,
                "gross_ret": gross,
                "tc": tc,
                "net_ret": net,
            }
        )

        # jump to exit (no overlapping trades)
        i = exit_i

    # Build equity curve (15m grid; returns realized at exits)
    equity = (1.0 + strat_ret.fillna(0.0)).cumprod()

    out = pd.DataFrame(
        {
            "proba_up": proba_up,
            "pos": pos,
            "ret_h": ret_h,
            "strat_ret": strat_ret,
            "equity": equity,
        }
    ).dropna()

    # Attach trades as an attribute for convenience
    out.attrs["trades"] = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_ts", "exit_ts", "side", "proba", "gross_ret", "tc", "net_ret"]
    )

    return out


def main() -> None:
    df_15m = pd.read_parquet("data/raw/binanceusdm_ETHUSDT_15m.parquet").sort_index()
    df_1h = pd.read_parquet("data/raw/binanceusdm_ETHUSDT_1h.parquet").sort_index()

    X = build_multitimeframe_features(df_15m, df_1h)

    model = XGBClassifier()
    model.load_model("models/xgb_baseline.json")

    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="proba_up")

    # backtest on the last 20% only (rough “out of sample-ish”)
    cut = int(len(proba) * 0.8)
    proba_bt = proba.iloc[cut:]

    df_bt = simple_backtest_hold(
        df_15m=df_15m,
        proba_up=proba_bt,
        horizon_bars=4,      # 1 hour horizon
        fee_bps=4.0,
        slippage_bps=2.0,
        long_th=0.55,
        short_th=0.45,
    )

    trades = df_bt.attrs["trades"]
    total_ret = float(df_bt["equity"].iloc[-1] - 1.0)
    mdd = _max_drawdown(df_bt["equity"])

    # Per-15m-bar stats (returns realized sparsely at exits)
    avg = float(df_bt["strat_ret"].mean())
    std = float(df_bt["strat_ret"].std())
    sharpe_like = (avg / std) * np.sqrt(365 * 24 * 4) if std > 0 else np.nan

    # Trade stats
    n_trades = len(trades)
    win_rate = float((trades["net_ret"] > 0).mean()) if n_trades > 0 else np.nan
    avg_trade = float(trades["net_ret"].mean()) if n_trades > 0 else np.nan
    exposure = float((df_bt["pos"] != 0).mean())

    print(f"Backtest bars: {len(df_bt):,}")
    print(f"Trades: {n_trades:,}")
    print(f"Exposure: {exposure*100:.1f}%")
    print(f"Total return: {total_ret*100:.2f}%")
    print(f"Max drawdown: {mdd*100:.2f}%")
    print(f"Win rate (net): {win_rate*100:.1f}%")
    print(f"Avg trade (net): {avg_trade*100:.3f}%")
    print(f"Sharpe-like (rough): {sharpe_like:.2f}")
    print("Equity tail:")
    print(df_bt[["equity"]].tail())

    # Optional: print last few trades
    if n_trades > 0:
        print("\nLast 5 trades:")
        print(trades.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
