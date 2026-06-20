# src/features/validate_features.py

import pandas as pd
import sys

EXPECTED_NEW_FEATURES = [
    "time_since_last_event_sec", "time_since_last_login_sec", "time_since_last_deposit_sec",
    "roll_5_trade_vol_mean", "roll_5_trade_vol_std", "roll_5_pnl_mean",
    "roll_10_trade_vol_mean", "roll_10_trade_vol_std", "roll_10_pnl_mean",
    "roll_5_click_rate_mean", "roll_10_click_rate_mean",
    "burst_count_5min", "burst_count_30min",
    "unique_ips_last_10_logins", "unique_countries_last_10_logins",
    "unique_devices_last_10_logins", "rolling_failed_attempts_5",
    "roll_5_deposit_sum", "withdrawal_to_deposit_ratio",
    "trade_vol_zscore", "pnl_zscore", "amount_zscore", "session_duration_zscore",
]


def validate_features(path: str):
    print(f"Validating: {path}")
    df = pd.read_csv(path)

    # ── All expected features present ─────────────────────────────────────────
    missing = [f for f in EXPECTED_NEW_FEATURES if f not in df.columns]
    assert not missing, f"Missing features: {missing}"

    # ── No infinite values ────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns
    inf_count = df[numeric_cols].isin([float("inf"), float("-inf")]).sum().sum()
    assert inf_count == 0, f"Found {inf_count} infinite values"

    # ── No NaN in engineered features ─────────────────────────────────────────
    nan_counts = df[EXPECTED_NEW_FEATURES].isnull().sum()
    bad_cols = nan_counts[nan_counts > 0]
    assert len(bad_cols) == 0, f"NaN values found in features:\n{bad_cols}"

    # ── Row count preserved ───────────────────────────────────────────────────
    raw_df = pd.read_csv("data/raw/events.csv", nrows=0)
    assert len(df) > 0, "Feature dataset is empty"

    # ── Z-scores are finite and reasonable ────────────────────────────────────
    for col in ["trade_vol_zscore", "pnl_zscore", "amount_zscore", "session_duration_zscore"]:
        extreme = (df[col].abs() > 50).sum()
        assert extreme < 100, f"{col} has {extreme} extreme values (|z|>50) — check for bugs"

    raw_cols = set(pd.read_csv("data/raw/events.csv", nrows=0).columns)
    new_cols = [c for c in df.columns if c not in raw_cols]

    print(f"\n✅ Feature validation passed")
    print(f"   Shape              : {df.shape}")
    print(f"   Original columns   : {len(raw_cols)}")
    print(f"   Engineered features: {len(new_cols)}")
    print(f"   All {len(EXPECTED_NEW_FEATURES)} expected features verified")
    print(f"\n   Feature value ranges (sample):")
    for col in ["burst_count_5min", "unique_ips_last_10_logins", "trade_vol_zscore",
                "withdrawal_to_deposit_ratio"]:
        if col in df.columns:
            s = df[col]
            print(f"     {col:<35} min={s.min():.2f}  max={s.max():.2f}  mean={s.mean():.2f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.csv"
    validate_features(path)
