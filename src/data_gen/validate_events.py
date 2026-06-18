# src/data_gen/validate_events.py

import pandas as pd
import sys

EXPECTED_EVENT_TYPES = {"login", "trade", "deposit", "withdrawal", "session", "kyc_change"}

EXPECTED_ANOMALY_TYPES = {
    "ip_hopper", "wash_trader", "deposit_withdrawal_cycler", "bot_trader",
    "structurer", "brute_forcer", "dormant_withdrawer", "consistent_winner",
    "device_switcher", "kyc_manipulator",
}

REQUIRED_COLUMNS = [
    "event_id", "user_id", "event_type", "timestamp", "is_anomalous", "anomaly_type",
    "ip_address", "country", "device", "trade_volume", "pnl", "amount",
    "click_rate_per_min", "timezone_gap_hours", "trade_volume_vs_baseline",
    "failed_attempts", "session_duration_mins",
]


def validate_events(path: str):
    print(f"Validating: {path}")
    df = pd.read_csv(path)

    # ── Column completeness ───────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing_cols, f"Missing required columns: {missing_cols}"

    # ── Non-empty ─────────────────────────────────────────────────────────────
    assert len(df) > 0, "Dataset is empty"

    # ── Event type completeness ───────────────────────────────────────────────
    actual_types = set(df["event_type"].dropna().unique())
    missing_types = EXPECTED_EVENT_TYPES - actual_types
    assert not missing_types, f"Missing event types: {missing_types}"

    # ── All 10 anomaly types present ──────────────────────────────────────────
    anom_df = df[df["is_anomalous"] == 1]
    actual_anomalies = set(anom_df["anomaly_type"].dropna().unique())
    missing_anomalies = EXPECTED_ANOMALY_TYPES - actual_anomalies
    assert not missing_anomalies, f"Missing anomaly types: {missing_anomalies}"

    # ── Timestamp ordering ────────────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    assert df["timestamp"].is_monotonic_increasing, "Events not sorted by timestamp"

    # ── No fully null rows ────────────────────────────────────────────────────
    fully_null = df.isnull().all(axis=1).sum()
    assert fully_null == 0, f"Found {fully_null} fully null rows"

    # ── Anomaly label consistency ─────────────────────────────────────────────
    none_labeled = df[(df["is_anomalous"] == 0) & (df["anomaly_type"] != "none")]
    assert len(none_labeled) == 0, f"{len(none_labeled)} rows labeled normal but have anomaly_type set"

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n✅ Validation passed")
    print(f"   Shape            : {df.shape}")
    print(f"   Date range       : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"   Anomalous events : {len(anom_df):,} / {len(df):,} ({len(anom_df)/len(df)*100:.1f}%)")
    print(f"\n   Event types:")
    for etype, cnt in df["event_type"].value_counts().items():
        print(f"     {etype:<30} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")
    print(f"\n   Anomaly types:")
    for atype, cnt in anom_df["anomaly_type"].value_counts().items():
        print(f"     {atype:<35} {cnt:>5,}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/events.csv"
    validate_events(path)
