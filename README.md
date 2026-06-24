# AnomX — Trader Behavior Anomaly Detection Engine

A real-time anomaly detection system for trading platforms, built as part of the
IIT Bombay Seasons of Code 2026 mentorship program.

The system ingests raw trading event logs, engineers behavioral features, and uses
unsupervised machine learning to surface suspicious traders without requiring labeled fraud data.

---

## Week 1–2: Python, Git & Foundations

Before writing any pipeline code, I needed to fill in gaps in the fundamentals. Weeks 1 and 2
covered Python, version control, and the core data libraries.

**Python foundations.** Coming in, I was comfortable with basic scripting but hadn't used list
comprehensions, generators, or `*args`/`**kwargs` much. Working through these made the data
pipeline code significantly cleaner — things like
`[f"roll_{w}_{c}" for w in windows for c in cols]` are much more readable than nested loops.

**Git workflow.** I set up the repo with a proper `.gitignore` (venv, `__pycache__`, `.ipynb_checkpoints`,
the mentor's reference folder) and practiced the commit → push cycle. I learned that committing
data files alongside code is actually fine for teaching repos — it means anyone can clone and
immediately reproduce results without re-running the pipeline.

**Pandas & NumPy.** These were the most important two weeks for the project. The feature
engineering in Week 4 is almost entirely Pandas — `groupby`, `rolling`, `transform`, `merge`,
`ffill`. The operations that would have taken hundreds of lines in plain Python take 5 lines in
Pandas, and they're vectorized so they run on 50,000 rows in seconds.

**Understanding anomaly detection.** This is the conceptual core of the project. The key insight
is that fraud detection is fundamentally an *unsupervised* problem. You can't label every fraud
case in advance — fraud patterns evolve, labels are noisy, and many platform operators don't even
know they're being defrauded until it's too late. Algorithms like Isolation Forest and Local
Outlier Factor (LOF) don't need labels — they learn what "normal" looks like across all users and
flag anything that's statistically far from that norm. The challenge is calibration: every
threshold you set is a tradeoff between catching real fraud and falsely flagging legitimate users.

---

## Week 3: Synthetic Event Data Generation

### Why synthetic data?

Real financial event logs from trading platforms are tightly regulated. Sharing them externally
violates GDPR/financial privacy laws, and even internally they require data masking. Building a
real anomaly detection system requires a dataset that has ground-truth fraud labels — which real
operational data almost never has. Synthetic data solves both problems: it's freely shareable and
every fraudulent record is known exactly because we injected it.

The tradeoff is that synthetic data doesn't fully capture the messiness of production logs. But
for learning feature engineering and model development, it's the right tool.

### The pipeline

`src/data_gen/generate_events.py` generates the dataset in four steps:

1. **Build user profiles** — 200 users, each with a home country, typical device, and trade volume baseline. The baseline varies from user to user (50K–500K), which matters for z-score features later.

2. **Generate normal events** — for each of 50,000 event slots, pick a random user and generate a realistic event of one of 6 types. All values (trade volume, PnL, deposit amounts, click rates) are sampled from distributions that approximate real platform behavior.

3. **Inject anomalies** — 10% of users (20 users, 2 per fraud pattern) have their events replaced by fraud-specific generators. Each generator models a specific evasion or manipulation technique.

4. **Sort and save** — events are globally sorted by timestamp and written to `data/raw/events.csv`.

### The 6 event types

| Event type | % of events | What it represents |
|------------|-------------|-------------------|
| `trade`    | 34.9%       | Buy/sell order on an instrument |
| `login`    | 22.1%       | Authentication event |
| `deposit`  | 15.1%       | Funds deposited to trading account |
| `session`  | 12.0%       | Active session (page clicks, duration) |
| `withdrawal` | 10.1%    | Funds withdrawn from account |
| `kyc_change` | 5.8%     | KYC identity document update |

### The 10 anomaly types

| Pattern | Events | What it models |
|---------|--------|---------------|
| `wash_trader` | 524 | Executes large coordinated trades to inflate volume metrics. Trade volume is 5–20× the user's own baseline. |
| `bot_trader` | 520 | Automated session with click rates of 258–2205 clicks/min. Normal sessions peak at about 10/min. |
| `ip_hopper` | 519 | Logs in from countries 4–8 time zones away from their registered home. Models account sharing or compromised credentials. |
| `consistent_winner` | 519 | Executes extremely fast trades (1–10 seconds). Real trades take minutes; sub-second execution suggests front-running or insider information. |
| `device_switcher` | 511 | Rotates through 5 different device types across logins. Legitimate users use 1–2 devices. |
| `kyc_manipulator` | 510 | Repeatedly updates KYC documents (address, phone, ID). Legitimate users rarely change KYC after onboarding. |
| `brute_forcer` | 505 | Multiple failed login attempts before success. Normal users almost never have `failed_attempts > 1`. |
| `deposit_withdrawal_cycler` | 484 | Deposits followed immediately by near-equal withdrawals. Classic money laundering pass-through pattern. |
| `structurer` | 477 | Deposits in the $490–$999 range, deliberately staying below the $1,000 AML reporting threshold. |
| `dormant_withdrawer` | 463 | Account shows no deposits for months, then withdraws a large sum. Models a compromised dormant account. |

### Dataset summary

- **50,000 events** across 200 users, spanning Jan 1–Mar 30, 2024
- **31 columns** including event metadata, financial values, and behavioral signals
- **5,032 anomalous events (10.1%)** — roughly 500 events per fraud pattern, spread across 2 users per type
- Validation: `python -m src.data_gen.validate_events` — checks all 6 event types, all 10 anomaly types, timestamp monotonicity, and label consistency

### Key observations from Week 3 EDA

The most striking finding was how invisible most fraud is in the raw data. Looking at the event
type distribution, anomalous and normal events are nearly indistinguishable at the aggregate level
— the fraud signal is buried in behavioral *patterns* over time, not in individual events.

A few signals do stand out in the raw columns:
- **Bot traders** have `click_rate_per_min` up to 2,205/min during burst sessions — normal sessions
  top out at about 10/min. But only 16 of the 520 bot events are actually from burst sessions;
  the rest look normal.
- **Structurers** deposit in the $490–$999 range for about 30% of their deposits, but the other
  70% are indistinguishable from normal deposits. Flagging every sub-$1,000 deposit would be
  useless — the signal only becomes visible with rolling deposit amount analysis.
- **Consistent winners** execute trades in 1 second, while normal trades have a median duration
  of about 30 minutes. This is probably the clearest single-column signal in the raw data.

---

## Week 4: Feature Engineering & EDA

### Why raw events aren't enough for ML

Machine learning models need numbers. But more importantly, they need numbers that capture
*context*. A single trade event with volume 1,000,000 tells you almost nothing. Is that large for
this user? Is it large relative to their recent history? Did they just make 10 other large trades
in the last hour?

Feature engineering converts raw event records into behavioral summaries. The feature pipeline in
`src/features/feature_engineering.py` adds 23 new columns to the original 31, producing a 54-column
dataset where each row carries its own behavioral context.

### The 6 feature categories

**1. Time-delta features** (`time_since_last_event_sec`, `time_since_last_login_sec`,
`time_since_last_deposit_sec`): Measure how much time has passed since the user's last event of
each type. A bot firing requests every 2 seconds looks very different from a human trading
every few hours. The median time between consecutive events is about 8 hours; bots collapse
that to seconds.

**2. Rolling window features** (windows 5 and 10): `roll_5_trade_vol_mean`, `roll_10_trade_vol_std`,
`roll_5_pnl_mean`, `roll_10_click_rate_mean`. These summarize the last N events of a given type.
Wash traders executing a burst of inflated-volume trades show up as a sudden spike in the rolling
mean. Without this, the signal is just "this trade was large" — with it, it's "this trade is
15× this user's recent average."

**3. Burst count features** (`burst_count_5min`, `burst_count_30min`): Count events in a
rolling time window. Most users generate 1–2 events per 5 minutes. The bot trader, during burst
sessions, generates many more. The 30-minute window captures more of the pattern than the
5-minute one.

**4. Login behavior features** (`unique_ips_last_10_logins`, `unique_countries_last_10_logins`,
`unique_devices_last_10_logins`, `rolling_failed_attempts_5`): Capture diversity in the last 10
login events. IP hoppers show up with elevated country counts — up to 8 different countries in
10 consecutive logins. Normal users use 1–2 countries at most. Brute forcers show
`rolling_failed_attempts_5` values that are consistently higher than any normal user.

**5. Financial behavior features** (`roll_5_deposit_sum`, `withdrawal_to_deposit_ratio`): The
deposit-withdrawal cycler pattern is essentially invisible in any single row — the signal is
the ratio of withdrawal amount to recent deposits. When `withdrawal_to_deposit_ratio` is
consistently near or above 1.0, the user is withdrawing everything they deposit. Dormant
withdrawers show very high ratios because their recent deposit sum is near zero.

**6. Z-score features** (`trade_vol_zscore`, `pnl_zscore`, `amount_zscore`,
`session_duration_zscore`): Measure how many standard deviations a value is from that user's
own historical mean. This is the user-relative normalization that makes the pipeline
robust to different trader profiles. A retail trader and an institutional trader have very
different typical volumes — z-scores make them comparable.

### Feature engineering output

- **54 total columns** (31 original + 23 engineered)
- **50,000 rows** — no events dropped, features are computed per user using `groupby` + `rolling`
- No NaN values in engineered features (trailing zeros for first events, forward-fill for type-specific features)
- Validation: `python -m src.features.validate_features` — checks all 23 features present, no infinities, no NaN

### Key EDA findings

**Strongest single signals:**
- `click_rate_per_min` — bot burst sessions are off the charts (2,205/min vs normal max ~10/min)
- `trade_duration_seconds` for consistent winners — 1 second vs normal median of ~30 minutes
- `rolling_failed_attempts_5` for brute forcers — the only anomaly type with sustained values above 3

**User-relative features work better than absolute ones:**
- `trade_vol_zscore` separates wash traders more cleanly than raw `trade_volume`, because users
  have different baseline volumes. The z-score captures "unusually large for this user" rather
  than just "large."
- `unique_countries_last_10_logins` is a stronger signal than raw country diversity for the
  same reason — it's relative to the user's own login history.

**Correlation observations:**
- `roll_5_trade_vol_mean` and `roll_10_trade_vol_mean` are highly correlated (r > 0.9). Keeping
  both in the model adds computation but probably minimal signal — worth pruning in ablation
  experiments during Week 5.
- `withdrawal_to_deposit_ratio` has near-zero correlation with everything else — an independent
  behavioral signal that should be valuable as a model feature.
- `burst_count_5min` and `burst_count_30min` are moderately correlated, but not redundant: the
  short window catches intense micro-bursts that the 30-minute window dilutes.

**The class imbalance challenge:**
5,032 of 50,000 events are anomalous (10.1%). At the user level it's more severe — 20 of 200
users are anomalous (10%), and each anomalous user generates ~250 events. Most anomaly detection
algorithms (Isolation Forest, LOF) are designed for this kind of imbalanced setting. The
contamination hyperparameter will need careful calibration against the feature distributions
found in the EDA.

---

## Project Structure

```
trader-behavior-anomaly-engine/
├── configs/
│   └── config.yaml                   — runtime config (seed, dataset size, feature params)
├── src/
│   ├── data_gen/
│   │   ├── generate_events.py        — synthetic event generation with 10 fraud patterns
│   │   └── validate_events.py        — assertion-based data quality checks
│   ├── features/
│   │   ├── feature_engineering.py    — behavioral feature pipeline (23 new features)
│   │   └── validate_features.py      — feature validation checks
│   └── utils/
│       └── logger.py                 — shared logging wrapper
├── data/
│   ├── raw/events.csv                — 50,000 events × 31 columns
│   └── processed/features.csv       — 50,000 events × 54 columns
├── notebooks/
│   ├── create_notebooks.py           — script to programmatically create EDA notebooks
│   ├── week3_events_exploration.ipynb
│   └── week4_feature_analysis.ipynb
├── reports/figures/                  — 14 PNG charts from EDA
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone <repo-url>
cd trader-behavior-anomaly-engine
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate raw event data
python -m src.data_gen.generate_events
python -m src.data_gen.validate_events     # should print: ✅ Validation passed

# Run feature engineering
python -m src.features.feature_engineering
python -m src.features.validate_features   # should print: ✅ Feature validation passed

# (Optional) Recreate and execute EDA notebooks
python notebooks/create_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/week3_events_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/week4_feature_analysis.ipynb
```

---

## What's Next (Week 5+)

With clean features in place, the next step is training unsupervised anomaly detection models:
- **Isolation Forest** — works by randomly partitioning the feature space; anomalies require
  fewer splits to isolate and get lower anomaly scores
- **Local Outlier Factor (LOF)** — compares each point's local density to its neighbors;
  anomalies sit in regions with lower density than their neighbors
- **Evaluation** — since labels exist in the synthetic dataset, I can compute precision/recall
  and AUROC. In a real deployment there would be no labels, so evaluation would rely on
  downstream analyst review rates and feedback loops
