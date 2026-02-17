from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .features import build_multitimeframe_features
from .labels import make_future_return_label
from .backtest import simple_backtest_hold, _max_drawdown  # uses your existing backtest functions


def time_split_idx(n: int, train_frac=0.7, val_frac=0.15):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    i_train = slice(0, n_train)
    i_val = slice(n_train, n_train + n_val)
    i_test = slice(n_train + n_val, n)
    return i_train, i_val, i_test


def summarize_backtest(df_bt: pd.DataFrame) -> dict:
    trades = df_bt.attrs.get("trades", pd.DataFrame())
    total_ret = float(df_bt["equity"].iloc[-1] - 1.0) if len(df_bt) else np.nan
    mdd = _max_drawdown(df_bt["equity"]) if len(df_bt) else np.nan
    exposure = float((df_bt["pos"] != 0).mean()) if len(df_bt) else np.nan

    n_trades = int(len(trades))
    win_rate = float((trades["net_ret"] > 0).mean()) if n_trades > 0 else np.nan
    avg_trade = float(trades["net_ret"].mean()) if n_trades > 0 else np.nan
    profit_factor = (
        float(trades.loc[trades["net_ret"] > 0, "net_ret"].sum()
              / (-trades.loc[trades["net_ret"] < 0, "net_ret"].sum()))
        if n_trades > 0 and (trades["net_ret"] < 0).any()
        else np.nan
    )

    return {
        "bars": int(len(df_bt)),
        "trades": n_trades,
        "exposure": exposure,
        "total_return": total_ret,
        "max_drawdown": float(mdd),
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "profit_factor": profit_factor,
    }


def main() -> None:
    # --- Config (keep simple for v1) ---

    horizon_bars = 4            # 1 hour on 15m grid
    fee_bps = 4.0
    slippage_bps = 2.0

    # threshold grids (adjust as you like)
    long_grid = [0.55, 0.60, 0.65, 0.70, 0.75]
    short_grid = [0.45, 0.40, 0.35, 0.30, 0.25]

    # --- Load data ---
    df_15m = pd.read_parquet("data/raw/binanceusdm_ETHUSDT_15m.parquet").sort_index()
    df_1h = pd.read_parquet("data/raw/binanceusdm_ETHUSDT_1h.parquet").sort_index()

    # --- Build features + labels ---
    X = build_multitimeframe_features(df_15m, df_1h)
    y = make_future_return_label(df_15m, horizon_bars=horizon_bars).reindex(X.index)

    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=[y.name])
    y = data[y.name].astype("int64")

    # --- Time split ---
    i_train, i_val, i_test = time_split_idx(len(X))
    X_train, y_train = X.iloc[i_train], y.iloc[i_train]
    X_val, y_val = X.iloc[i_val], y.iloc[i_val]

    # --- Train baseline model on TRAIN only ---
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Predict probabilities on VAL only ---
    proba_val = pd.Series(
        model.predict_proba(X_val)[:, 1],
        index=X_val.index,
        name="proba_up",
    )

    # --- Sweep thresholds ---
    rows = []
    for long_th in long_grid:
        for short_th in short_grid:
            if short_th >= long_th:
                continue  # must have a gap

            df_bt = simple_backtest_hold(
                df_15m=df_15m,
                proba_up=proba_val,
                horizon_bars=horizon_bars,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                long_th=long_th,
                short_th=short_th,
            )
            s = summarize_backtest(df_bt)
            s.update({"long_th": long_th, "short_th": short_th})
            rows.append(s)

    out = pd.DataFrame(rows)

    # Rank: prioritize robustness-ish metrics over raw return
    # (You can change this later.)
    out = out.sort_values(
        by=["profit_factor", "total_return", "max_drawdown"],
        ascending=[False, False, True],
        na_position="last",
    )

    Path("reports").mkdir(exist_ok=True)
    csv_path = Path("reports/threshold_sweep_val.csv")
    out.to_csv(csv_path, index=False)

    print(f"\nSaved: {csv_path}")
    print("\nTop 10 (VAL):")
    cols = ["long_th", "short_th", "trades", "exposure", "total_return", "max_drawdown", "win_rate", "avg_trade", "profit_factor"]
    print(out[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
