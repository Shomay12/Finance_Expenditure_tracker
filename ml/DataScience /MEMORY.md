# Agent Context & Memory Document
## AI-Based Personal Finance Tracking & Intelligence System (Data Science & AI/ML)

---

## 1. System State & Path Details

- **Workspace Root**: `/Users/quaxiom/Documents/DEVLOPMENT/Projects/Finance project/DataScience ` *(Note: Workspace path contains a trailing space)*.
- **Python Environment**: Python 3.13.9 with `scikit-learn`, `pandas`, `numpy`, `xgboost`, `scipy`, `joblib`.
- **Currency Perspective**: Indian Rupee (**INR ₹**), Indian merchant ecosystem (UPI handles, NetBanking, Swiggy, Zepto, Blinkit, TCS/Infosys salaries, Lakhs Per Annum income brackets).
- **Inference & Reporting CLI**: [`test.py`](file:///Users/quaxiom/Documents/DEVLOPMENT/Projects/Finance%20project/DataScience/test.py) (Pure inference + Aggregated Financial Intelligence Report layer).
- **Master Pipeline**: [`run_all_pipelines.py`](file:///Users/quaxiom/Documents/DEVLOPMENT/Projects/Finance%20project/DataScience/run_all_pipelines.py) (End-to-end training and evaluation).
- **Report Storage Directory**: `data/test_results/` (Saves `financial_report_<TIMESTAMP>.txt` and `.json`).

---

## 2. Core Architectural Invariants

1. **Strict Dataset Segregation**:
   - `/data/raw/` is **IMMUTABLE**. Never overwrite or write temporary processed files into raw data paths.
   - All transformations are outputted to `/data/processed/`, `/data/features/`, or split directories (`/data/train/`, `/data/validation/`, `/data/test/`).
2. **Data Leakage Invariants**:
   - Classification uses stratified splitting on composite labels (`Category::Subcategory`).
   - Time-series forecasting and longitudinal data use **STRICT CHRONOLOGICAL SPLITS** (Months 1–5 Train $\rightarrow$ Month 6 Validation $\rightarrow$ Months 7–8 Test). Shuffling time-series is strictly forbidden.
   - Rolling features (7-day, 30-day, velocity) are computed backwards with `shift(1)` to prevent lookahead leakage.
3. **No LLM Training on Financial Transactions**: Supervised ML models (TF-IDF, Logistic Regression, XGBoost) are utilized for deterministic, low-latency, and interpretable classification.
4. **Report Layer Separation of Concerns**:
   - The ML models answer: *"What happened to this transaction?"* (structured predictions).
   - The Report Layer answers: *"What does all this transaction data tell us about the user's financial activity?"* (aggregation, cash flow, anomaly summaries, recurring distinction, data quality warnings).
   - The report generator does **NOT** run new ML models.
5. **Confidence & Data Completeness Safeguards**:
   - Classification outputs with confidence $< 60.0\%$ display visible warnings (`CATEGORY UNCERTAIN: Requires manual review`).
   - Forecasts for isolated transactions explicitly state `Status: Insufficient data for reliable forecast` instead of fabricating numbers.
   - Recurring investments (SIP, RD) are strictly separated from recurring consumption expenses (Netflix, Rent, Gym).
6. **Strict User-Level Data Isolation (`GLOBAL MODEL != GLOBAL USER DATA`)**:
   - Global ML models (Classifier, Anomaly Regressors, Forecaster, K-Means) share weights.
   - User Financial State (Transactions, Baselines, Profiles, LLM Context, Conversations) is strictly partitioned by authenticated `user_id`.
   - Category baselines for anomaly detection are computed per user and per category (e.g. `user_category_baselines[(user_id, subcategory)]`).
   - AI Orchestrator's `build_user_llm_context(user_id, ...)` guarantees that only the authenticated user's transactions, anomalies, commitments, and chat history enter LLM prompts.
   - Multi-tenant isolation verified by `tests/test_user_isolation.py`.
7. **Pure Inference in `test.py`**: No training routines, no `fit()`, no `fit_transform()`, no dataset regeneration inside `test.py`.

---

## 3. Benchmarked Model Performance Reference

| Module | Winning Model | Primary Metric | Baseline Comparison | Artifact Location |
| :--- | :--- | :--- | :--- | :--- |
| **Transaction Classifier** | Logistic Regression / XGBoost | **Macro F1: 1.0000** | Random Forest: 0.9783 | `models/transaction_classifier/` |
| **Contextual Anomaly Detector** | Isolation Forest + Baselines | **Val Anomaly: 1.60%** | LOF: 1.67%, OCSVM: 2.45% | `models/anomaly_detector/` |
| **Spending Forecaster** | XGBoost Regressor | **Test MAE: ₹7,380.37** | Baseline MTD Extrapolation: ₹30,923.54 | `models/forecasting/` |
| **User Segmentation** | K-Means ($k=2..4$) | **Silhouette: 0.2369** | DB Index: 1.4472 | `models/segmentation/` |
| **Econometric Research** | OLS Multiple Regression | **$R^2 = 0.5140$** | Top Driver: Missed Txns ($\beta=+0.3830$) | `src/analysis/behavioral_research.py` |

---

## 4. Key Behavioral Research Findings

- **Primary Research Question**: *"Why do people struggle to track where their money goes each month?"*
- **Key Determinants**:
  1. **Payment Channel Fragmentation**: Users managing 4+ payment apps/cards (UPI apps, wallets, cards) experience significantly higher tracking friction ($t = 22.28, p < 10^{-60}$).
  2. **Missed Micro-Transactions**: High volume of UPI QR micro-payments ($\beta = +0.3830$) leads to cognitive leakage and forgotten spend.
  3. **Subscription Accumulation**: Silent auto-debits ($\beta = +0.0752$) degrade month-end budgeting clarity.
- **Methodological Distinction**: Clearly distinguishes correlation ($r = 0.61$) and statistical association (regression controlling for income/budgeting) from causal claims.

---

## 5. Execution Commands for New Agents

```bash
# 1. Run full end-to-end pipeline (Data Generation -> Preprocessing -> All Models -> Orchestrator validation)
python3 run_all_pipelines.py

# 2. Interactive manual model testing console
python3 test.py

# 3. Run preset 8-scenario demo with aggregated Financial Intelligence Report
python3 test.py --demo

# 4. Run batch CSV testing with aggregated Financial Intelligence Report
python3 test.py --csv data/test_transactions_sample.csv

# 5. Run debug inspection mode
python3 test.py --debug --demo

# 6. Run User Isolation Verification Test Suite
python3 tests/test_user_isolation.py

# 7. Test individual components
python3 src/models/transaction_classifier.py
python3 src/models/anomaly_detector.py
python3 src/models/recurring_detector.py
python3 src/models/spending_forecaster.py
python3 src/analysis/behavioral_research.py
python3 src/models/user_segmenter.py
python3 src/orchestrator.py
```
