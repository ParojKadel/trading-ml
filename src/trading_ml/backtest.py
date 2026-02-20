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
    fee_bps: float = 12.0,
    slippage_bps: float = 6.0,
    long_th: float = 0.70,
    short_th: float = 0.25,
) -> pd.DataFrame:
    """
    1h-consistent backtest:
      - Decide at time t based on proba_up[t]
      - If signal triggers, enter and HOLD for `horizon_bars`
      - PnL is paid over close(t) -> close(t+horizon_bars)
      - Fees/slippage applied on entry and exit (2 turns total) per trade
      - Equity is tracked on the 15m grid (returns realized at exit bar)
    """
    close = df_15m["close"].reindex(proba_up.index).astype("float64")

    # Future return over horizon: ret_h[t] = close[t+H]/close[t] - 1
    ret_h = (close.shift(-horizon_bars) / close) - 1.0

    idx = proba_up.index
    n = len(idx)

    pos = pd.Series(0.0, index=idx)
    strat_ret = pd.Series(0.0, index=idx)

    cost = (fee_bps + slippage_bps) / 10_000.0  # bps -> fraction

    i = 0
    trades: list[dict] = []

    while i < n:
        p = float(proba_up.iloc[i])

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
        if exit_i >= n:
            break

        pos.iloc[entry_i:exit_i] = signal

        gross = float(signal * ret_h.iloc[entry_i])

        # entry + exit costs
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

        # no overlapping trades
        i = exit_i

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

    out.attrs["trades"] = (
        pd.DataFrame(trades)
        if trades
        else pd.DataFrame(columns=["entry_ts", "exit_ts", "side", "proba", "gross_ret", "tc", "net_ret"])
    )

    return out


def _time_split_idx(n: int, train_frac: float = 0.7, val_frac: float = 0.15):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    i_train = slice(0, n_train)
    i_val = slice(n_train, n_train + n_val)
    i_test = slice(n_train + n_val, n)
    return i_train, i_val, i_test


def main() -> None:
    # =========================
    # Config
    # =========================
    MODE = "real"  # "real" | "permute" | "random"

    horizon_bars = 4            # 1 hour on 15m grid
    long_th = 0.70
    short_th = 0.25

    fee_bps = 12.0
    slippage_bps = 6.0

    model_path = "models/xgb_baseline.json"

    # =========================
    # Load data + features
    # =========================
    df_15m = pd.read_parquet("data/raw/binanceusdm_ETHUSDT_15m.parquet").sort_index()
    df_1h = pd.read_parquet("data/raw/binanceusdm_ETHUSDT_1h.parquet").sort_index()

    X = build_multitimeframe_features(df_15m, df_1h).dropna()

    # =========================
    # Load trained model
    # =========================
    model = XGBClassifier()
    model.load_model(model_path)

    proba_real = pd.Series(
        model.predict_proba(X)[:, 1],
        index=X.index,
        name="proba_up",
    )

    # =========================
    # Choose proba series (controls)
    # =========================
    if MODE == "real":
        proba_used = proba_real
    elif MODE == "permute":
        proba_used = proba_real.sample(frac=1.0, random_state=42)
        proba_used.index = proba_real.index  # keep timestamps
    elif MODE == "random":
        rng = np.random.default_rng(42)
        proba_used = pd.Series(rng.uniform(0.0, 1.0, len(proba_real)), index=proba_real.index, name="proba_up")
    else:
        raise ValueError(f"Unknown MODE={MODE}. Use 'real', 'permute', or 'random'.")

    print(f"MODE: {MODE}")
    print(f"Model loaded from: {model_path}")
    print("proba head:")
    print(proba_used.head())

    # =========================
    # Backtest only on TEST split
    # =========================
    i_train, i_val, i_test = _time_split_idx(len(X))
    proba_bt = proba_used.iloc[i_test]

    df_bt = simple_backtest_hold(
        df_15m=df_15m,
        proba_up=proba_bt,
        horizon_bars=horizon_bars,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        long_th=long_th,
        short_th=short_th,
    )

    trades = df_bt.attrs["trades"]
    total_ret = float(df_bt["equity"].iloc[-1] - 1.0) if len(df_bt) else np.nan
    mdd = _max_drawdown(df_bt["equity"]) if len(df_bt) else np.nan
    exposure = float((df_bt["pos"] != 0).mean()) if len(df_bt) else np.nan

    n_trades = int(len(trades))
    win_rate = float((trades["net_ret"] > 0).mean()) if n_trades > 0 else np.nan
    avg_trade = float(trades["net_ret"].mean()) if n_trades > 0 else np.nan

    print("\n== TEST BACKTEST ==")
    print(f"Backtest bars: {len(df_bt):,}")
    print(f"Trades: {n_trades:,}")
    print(f"Exposure: {exposure*100:.1f}%")
    print(f"Total return: {total_ret*100:.2f}%")
    print(f"Max drawdown: {mdd*100:.2f}%")
    print(f"Win rate (net): {win_rate*100:.1f}%")
    print(f"Avg trade (net): {avg_trade*100:.3f}%")
    print("Equity tail:")
    print(df_bt[["equity"]].tail())

    if n_trades > 0:
        print("\nLast 5 trades:")
        print(trades.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()