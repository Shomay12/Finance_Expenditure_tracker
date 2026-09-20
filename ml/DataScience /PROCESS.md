# Operational Process & Pipeline Guide
## AI-Based Personal Finance Tracking & Intelligence System (Indian Context - INR ₹)

This document outlines the standard operating procedures, data flow pipelines, leakage prevention protocols, evaluation workflows, and manual testing routines for the Data Science & AI/ML layer.

---

## 1. Directory Structure

```
.
├── PROJECT.md                                # High-level architecture & domain definitions
├── PROCESS.md                                # Operational execution workflows (this document)
├── MEMORY.md                                 # State, lessons learned, and context memory for agents
├── run_all_pipelines.py                      # Master end-to-end training & evaluation runner
├── test.py                                   # Pure inference testing & Financial Intelligence Report console
├── data/
│   ├── raw/                                  # Immutable raw input files (INR ₹)
│   │   ├── category_mapping.csv
│   │   ├── raw_transactions_classification.csv
│   │   ├── raw_historical_transactions.csv
│   │   └── raw_user_behavior_survey.csv
│   ├── processed/                            # Normalized, cleaned datasets
│   ├── features/                             # Intermediate engineered feature vectors
│   ├── train/                                # Training splits (stratified / chronological)
│   ├── validation/                           # Validation splits
│   ├── test/                                 # Final held-out test splits
│   ├── test_transactions_sample.csv          # Sample batch test file in Indian context
│   └── test_results/                         # Automated exports from manual CSV batch testing & reports
├── models/
│   ├── transaction_classifier/               # model.pkl, label_encoder.pkl, metadata.json
│   ├── anomaly_detector/                     # model.pkl, scaler.pkl, baselines.pkl, metadata.json
│   ├── forecasting/                          # model.pkl, scaler.pkl, metadata.json
│   └── segmentation/                         # model.pkl, scaler.pkl, pca.pkl, metadata.json
├── tests/
│   └── test_user_isolation.py                # Multi-tenant isolation & cross-contamination test suite
└── src/
    ├── data/
    │   ├── data_generators.py                # Synthetic realistic raw data generators (Indian entities)
    │   ├── preprocessing.py                  # Cleaning, normalization, splitting pipeline
    │   └── user_data_access.py               # User Data Access Layer (Multi-tenant partition & chat memory)
    ├── models/
    │   ├── transaction_classifier.py         # Supervised NLP + ML classification pipeline
    │   ├── anomaly_detector.py               # Contextual Isolation Forest anomaly engine
    │   ├── recurring_detector.py             # Deterministic subscription & bill detector
    │   ├── spending_forecaster.py            # Chronological XGBoost spending forecaster
    │   └── user_segmenter.py                 # K-Means behavioral clustering & personas
    ├── analysis/
    │   └── behavioral_research.py            # Statistical EDA, hypothesis tests & OLS regression
    └── orchestrator.py                       # Unified inference & structured JSON engine
```

---

## 2. Training Pipeline Execution Flow (Steps 1 to 8)

### Step 1: Raw Data Generation / Ingestion
- Executable: `python3 src/data/data_generators.py`
- Outputs:
  - `data/raw/raw_transactions_classification.csv` (6,000 records)
  - `data/raw/raw_historical_transactions.csv` (20,000+ records, 40 users, 8 months)
  - `data/raw/raw_user_behavior_survey.csv` (1,500 respondents)

### Step 2: Preprocessing & Leakage-Proof Splitting
- Executable: `python3 src/data/preprocessing.py`
- Rules:
  - Classification Split: 70% Train, 15% Validation, 15% Test with stratification on `Category::Subcategory`.
  - Historical Split: Strict chronological partition (Months 1–5 Train, Month 6 Validation, Months 7–8 Test).

### Step 3: Supervised Classification Training
- Executable: `python3 src/models/transaction_classifier.py`
- Models Evaluated: Logistic Regression, Random Forest, XGBoost.

### Step 4: Contextual Anomaly Detection Training
- Executable: `python3 src/models/anomaly_detector.py`
- Evaluated: Isolation Forest, Local Outlier Factor, One-Class SVM.

### Step 5: Recurring Expense & Subscription Detection
- Executable: `python3 src/models/recurring_detector.py`
- Separates recurring consumption expenses from recurring investments/transfers.

### Step 6: Temporal Spending Forecasting Training
- Executable: `python3 src/models/spending_forecaster.py`
- Evaluated: MTD Baseline, 30d Moving Avg Baseline, Ridge, Random Forest, XGBoost.

### Step 7: Behavioral Research & Econometric Analysis
- Executable: `python3 src/analysis/behavioral_research.py`
- Statistical EDA, hypothesis testing, OLS regression ($R^2 = 0.5140$).

### Step 8: User Behavioral Segmentation
- Executable: `python3 src/models/user_segmenter.py`
- K-Means clustering with optimal $k$ and descriptive personas.

---

## 3. Step 9: Financial Intelligence Report & Testing Protocol (`test.py`)

The inference test console [`test.py`](file:///Users/quaxiom/Documents/DEVLOPMENT/Projects/Finance%20project/DataScience/test.py) incorporates a dedicated **Financial Intelligence Report Generator**:

```
                    TRANSACTION DATA
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Categorization   Anomaly      Forecast
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                 Recurring Detection
                           │
                           ▼
                  Financial Analytics
                           │
                           ▼
                FINANCIAL REPORT LAYER
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
              Human Report       JSON Report
```

### Report Sections
1. **Dataset Summary & Net Cash Flow**: Total income, total expenses, total investments, net cash flow.
2. **Spending Breakdown**: Sorted category allocation table with percentages.
3. **Income Analysis**: Breakdown across salary, freelance, business, interest, other income.
4. **Anomaly Analysis**: Flags unusual spikes with historical reasons and highlights `CATEGORY UNCERTAIN` if confidence $< 60\%$.
5. **Recurring Expense Analysis**: Distinguishes recurring expenses (Netflix, Gym, Broadband) from recurring investments (SIP, RD).
6. **Spending Forecast**: Evaluates longitudinal MTD data or displays explicit insufficient data status for isolated transactions.
7. **Financial Behavior Signals**: Weekend ratio, recurring ratio, subscription ratio, savings ratio (purely descriptive, no psychological claims).
8. **User Segmentation**: Descriptive persona if feature vector is supplied.
9. **Tracking Difficulty Signals**: Connects back to the research question (active payment channels, micro-transactions, subscription count).
10. **Key Observations**: Concise, factual numbered findings.
11. **Data Quality / Model Warnings**: Explicit alerts for low confidence, uncalibrated confidence bounds, and synthetic data caveats.
12. **Final Financial Intelligence Summary**: Executive summary of findings.
13. **Auto-Saved Reports**:
    - `data/test_results/financial_report_TIMESTAMP.txt`
    - `data/test_results/financial_report_TIMESTAMP.json`

### CLI Commands
```bash
# 1. Interactive Menu
python3 test.py

# 2. Preset 8-Scenario Demo with Full Financial Intelligence Report
python3 test.py --demo

# 3. Batch CSV Execution with Full Financial Intelligence Report
python3 test.py --csv data/test_transactions_sample.csv

# 4. Debug Inspection Mode
python3 test.py --debug --demo
```

---

## 4. Step 10: Multi-Tenant User-Isolation Verification Protocol (`tests/test_user_isolation.py`)

Run automated isolation and cross-contamination tests:
```bash
python3 tests/test_user_isolation.py
```

### Verification Checks Performed:
1. **Transaction Partitioning**: Confirms transactions are partitioned by `user_id` and cannot be retrieved globally.
2. **Personal Category Baselines**: Confirms User A (Food avg ₹500) and User B (Food avg ₹5,000) have separate historical distributions.
3. **Cross-Contamination Anomaly Detection**: A ₹5,000 Food transaction is flagged as an anomaly for User A, but evaluated as normal for User B.
4. **Recurring Stream Isolation**: Disjoint detection of individual subscription amounts (User A ₹649 Netflix vs User B ₹1,499 Netflix).
5. **Behavioral Feature Vectors**: Scoped vector generation per user.
6. **LLM Context & Report Isolation**: Verifies disjoint transaction sets and zero cross-tenant financial value leakage in prompts and reports.
7. **Chat Memory Isolation**: Verifies multi-user conversational history partitions.
