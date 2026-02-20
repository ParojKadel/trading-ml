from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from .features import build_multitimeframe_features
from .labels import make_future_return_label


def time_split(X: pd.DataFrame, y: pd.Series, train_frac=0.7, val_frac=0.15):
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train:n_train + n_val], y.iloc[n_train:n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

RANDOM_LABEL_TEST = False
RANDOM_LABEL_SEED = 42


def main() -> None:
    df_15m = pd.read_parquet("data/raw/binanceusdm_ETHUSDT_15m.parquet").sort_index()
    df_1h = pd.read_parquet("data/raw/binanceusdm_ETHUSDT_1h.parquet").sort_index()

    X = build_multitimeframe_features(df_15m, df_1h)
    y = make_future_return_label(df_15m, horizon_bars=4).reindex(X.index)

    # align and drop NaNs (label has NaNs at the end because of shift)
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=[y.name])
    y = data[y.name].astype("int64")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X, y)

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
    )
    if RANDOM_LABEL_TEST:
        y_train = y_train.sample(frac=1.0, random_state=RANDOM_LABEL_SEED)

    model.fit(X_train, y_train)

    def eval_split(name, Xs, ys):
        proba = model.predict_proba(Xs)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(ys, proba)
        print(f"\n== {name} ==")
        print(f"AUC: {auc:.4f}")
        print(classification_report(ys, pred, digits=4))

    eval_split("VAL", X_val, y_val)
    eval_split("TEST", X_test, y_test)

    Path("models").mkdir(exist_ok=True)
    out_path = Path("models/xgb_baseline.json")
    model.save_model(out_path.as_posix())
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    main()
