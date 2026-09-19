# AI-Based Personal Finance Tracking & Intelligence System

An end-to-end Data Science, Machine Learning, and AI system designed to
understand personal financial activity, identify spending patterns,
detect unusual transactions, recognize recurring expenses, forecast
spending, analyze user behavior, and provide personalized financial
intelligence.

The central research question is:

> **Why do people struggle to track where their money goes each month?**

The system addresses this problem by combining transaction-level machine
learning, behavioral analysis, financial analytics, personalized user
baselines, and an AI orchestration layer.

------------------------------------------------------------------------

## 1. Project Vision

Traditional expense trackers primarily record transactions.

This project aims to go further:

``` text
Raw Financial Transactions
          ↓
Data Cleaning & Validation
          ↓
Feature Engineering
          ↓
Transaction Categorization
          ↓
Contextual Anomaly Detection
          ↓
Recurring Expense Detection
          ↓
Spending Forecasting
          ↓
Behavioral Analysis
          ↓
User Segmentation
          ↓
Financial Intelligence
          ↓
AI Orchestrator
          ↓
Local / Open-Weight LLM
          ↓
Personalized Financial Explanation
```

The system is designed around two principles:

1.  **ML models provide structured financial intelligence.**
2.  **The LLM explains that intelligence rather than inventing financial
    calculations.**

------------------------------------------------------------------------

# 2. Complete System Architecture

``` text
                              USER
                                │
                                ▼
                         ┌─────────────┐
                         │ React.js UI │
                         └──────┬──────┘
                                │ HTTPS
                                ▼
                         ┌─────────────┐
                         │ Cloudflare  │
                         │ DNS / WAF   │
                         │ TLS / Rate  │
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │   FastAPI   │
                         │   Backend   │
                         └──────┬──────┘
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
       ┌───────────┐      ┌────────────┐     ┌──────────────┐
       │ Supabase  │      │ Data       │     │ AI           │
       │ PostgreSQL│      │ Science/ML │     │ Orchestrator │
       └───────────┘      └─────┬──────┘     └──────┬───────┘
                                │                   │
                    ┌───────────┼───────────┐       │
                    │           │           │       │
                    ▼           ▼           ▼       │
              Categorization Anomaly    Forecast    │
                    │           │           │       │
                    └───────────┼───────────┘       │
                                ▼                   │
                       ┌─────────────────┐          │
                       │ Financial       │◄─────────┘
                       │ Intelligence    │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Local/Open      │
                       │ Weight LLM      │
                       │ vLLM / llama.cpp│
                       └────────┬────────┘
                                │
                                ▼
                       Personalized Insight
                                │
                                ▼
                               USER
```

------------------------------------------------------------------------

# 3. Core Data Science Pipeline

The current Data Science pipeline contains eight major stages:

``` text
STEP 1  → Dataset Generation / Verification
STEP 2  → Preprocessing & Leakage-Proof Splitting
STEP 3  → Transaction Categorization
STEP 4  → Contextual Anomaly Detection
STEP 5  → Recurring Expense Detection
STEP 6  → Temporal Spending Forecasting
STEP 7  → Behavioral Research & Econometrics
STEP 8  → User Segmentation
```

After these stages:

``` text
All ML Modules
      ↓
Financial Intelligence Orchestrator
      ↓
Unified Structured Output
      ↓
AI / LLM Layer
```

------------------------------------------------------------------------

# 4. Dataset Architecture

The project currently uses three primary datasets.

## Dataset 1: Transaction Classification

Current generated size:

``` text
6,000 records
```

Purpose:

-   Transaction categorization
-   Income/expense/transfer classification
-   Subcategory prediction
-   Confidence estimation

Example:

``` text
SWIGGY*BANGALORE FOOD
₹650
        ↓
EXPENSE → Food
```

------------------------------------------------------------------------

## Dataset 2: Historical Transactions

Current generated size:

``` text
20,393 transactions
40 users
8 months
```

Purpose:

-   User-specific spending history
-   Anomaly detection
-   Recurring expense detection
-   Spending forecasting
-   Behavioral feature engineering
-   User segmentation

Current chronological split:

``` text
Training:
2025-01 → 2025-05

Validation:
2025-06

Test:
2025-07 → 2025-08
```

This avoids using future transaction periods to train a model that
predicts earlier periods.

------------------------------------------------------------------------

## Dataset 3: User Behavior Survey

Current generated size:

``` text
1,500 survey records
```

Purpose:

-   Descriptive statistics
-   Correlation analysis
-   Hypothesis testing
-   Regression/econometric analysis
-   Investigation of financial tracking difficulty

------------------------------------------------------------------------

# 5. Unified Transaction Schema

The internal transaction representation is designed around fields such
as:

``` text
transaction_id
user_id
date
merchant_raw
merchant_normalized
description
amount
currency
transaction_type
category
subcategory
payment_method
is_recurring
is_anomaly
anomaly_score
category_confidence
day_of_week
month
year
created_at
```

The `user_id` field is particularly important because financial
intelligence is user-specific.

------------------------------------------------------------------------

# 6. Standard Financial Taxonomy

The system uses three top-level transaction types.

## Income

``` text
Salary
Freelance
Business Income
Interest
Other Income
```

## Expense

``` text
Food
Groceries
Shopping
Transport
Fuel
Bills & Utilities
Rent & Housing
Healthcare
Education
Entertainment
Travel
Subscriptions
Insurance
Personal Care
Gifts & Donations
Fees & Charges
Other Expense
```

## Transfer

``` text
Bank Transfer
Credit Card Payment
Investment
Savings
Cash Withdrawal
Other Transfer
```

------------------------------------------------------------------------

# 7. Data Preprocessing

The unified preprocessing flow is:

``` text
Raw Dataset
    ↓
Schema Validation
    ↓
Missing Value Handling
    ↓
Duplicate Detection
    ↓
Merchant Normalization
    ↓
Category Mapping
    ↓
Feature Engineering
    ↓
Data Quality Validation
    ↓
Model-Specific Dataset
```

Raw data should not be overwritten.

Suggested structure:

``` text
data/
├── raw/
├── processed/
├── features/
├── train/
├── validation/
└── test/
```

------------------------------------------------------------------------

# 8. Transaction Categorization

The transaction classifier predicts:

``` text
Category
Subcategory
Confidence
```

Example:

``` json
{
  "category": "EXPENSE",
  "subcategory": "Food",
  "category_confidence": 0.9723
}
```

Models currently benchmarked:

``` text
Logistic Regression
Random Forest
XGBoost
```

Current validation results:

  Model                   Accuracy   Macro F1
  --------------------- ---------- ----------
  Logistic Regression       1.0000     1.0000
  Random Forest             0.9856     0.9783
  XGBoost                   1.0000     1.0000

The current selected model is:

``` text
Logistic Regression
```

Current held-out test:

``` text
Accuracy: 1.0000
Macro F1: 1.0000
Weighted F1: 1.0000
```

Because the current benchmark data is generated/synthetic, these perfect
results should be treated as a pipeline validation result rather than
proof of real-world generalization. External or realistically perturbed
transaction data should be used for further validation.

------------------------------------------------------------------------

# 9. Confidence-Aware Classification

The system should not treat every prediction as equally reliable.

Example:

``` text
Category:
Shopping

Confidence:
34.79%

Status:
LOW CONFIDENCE
```

A configurable threshold is used:

``` text
60%
```

If confidence is below the threshold, the transaction should be marked
for review.

Conceptually:

``` text
Transaction
     ↓
Classifier
     ↓
Confidence
     │
     ├── High → Accept prediction
     │
     └── Low  → Flag for review
```

This is important because downstream anomaly detection can be affected
by an incorrect category.

------------------------------------------------------------------------

# 10. Contextual Anomaly Detection

The anomaly system identifies transactions that deviate from a user's
historical behavior.

Models benchmarked:

``` text
Isolation Forest
One-Class SVM
Local Outlier Factor
```

Validation results:

``` text
Isolation Forest:
41 anomalies / 1.60%

One-Class SVM:
63 anomalies / 2.45%

Local Outlier Factor:
43 anomalies / 1.67%
```

The current selected implementation uses:

``` text
Isolation Forest
```

Example:

``` text
User's normal Food spending:
~₹482

New Food transaction:
₹8,500

        ↓

17.6× historical average

        ↓

Potential anomaly
```

The system should not simply use:

``` python
amount > 10000
```

because a transaction's meaning depends on user context and transaction
type.

------------------------------------------------------------------------

# 11. User-Specific Personalization

A central architectural requirement is:

> **User A's financial data must not become User B's personal financial
> context.**

The global ML models can be shared.

User-specific financial state cannot be shared.

``` text
                 GLOBAL ML MODELS
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        USER A       USER B       USER C
          │            │            │
      A's data      B's data      C's data
          │            │            │
      A's profile   B's profile   C's profile
          │            │            │
      A's baseline  B's baseline  C's baseline
          │            │            │
      A inference   B inference   C inference
          │            │            │
      A context     B context     C context
          │            │            │
      A report      B report      C report
```

For example:

User A:

``` text
Food history:
₹300
₹450
₹500
₹380
```

User B:

``` text
Food history:
₹2,000
₹3,000
₹4,500
₹2,800
```

A ₹5,000 Food transaction must be evaluated against the appropriate
user's history.

The same global model can be used for both users, but the personal
baseline and feature state must remain separate.

------------------------------------------------------------------------

# 12. User-Specific Baselines

Personalized baselines can include:

``` text
Category average
Category standard deviation
Merchant frequency
Merchant amount
Monthly spending
Transaction frequency
Recurring commitments
Weekend spending
Subscription behavior
```

Conceptually:

``` json
{
  "user_id": "USR_0001",
  "category_baselines": {
    "Food": {
      "mean": 482.63,
      "std": 210.42
    },
    "Shopping": {
      "mean": 1437.28,
      "std": 720.12
    }
  }
}
```

These values should be calculated from that user's historical
transactions.

------------------------------------------------------------------------

# 13. Recurring Expense Detection

Recurring expenses are initially detected deterministically rather than
forcing another ML model into the architecture.

Signals include:

``` text
Merchant similarity
Amount similarity
Transaction frequency
Time interval
Day of month
Historical occurrence
```

Example:

``` text
Netflix
₹649
Every ~30 days

        ↓

Recurring monthly subscription
```

The system can identify:

``` text
Netflix
Broadband
Gym
Rent
SIP / Investment
Other recurring commitments
```

Recurring expenses and recurring investments should be distinguished in
reporting.

------------------------------------------------------------------------

# 14. Spending Forecasting

The forecasting system uses historical temporal information.

Models benchmarked:

``` text
Baseline: MTD Daily Rate Extrapolation
Baseline: Rolling 30-day Moving Average
Ridge Regression
Random Forest Regressor
XGBoost Regressor
```

Validation results:

  Model                      MAE         RMSE     MAPE
  ----------------- ------------ ------------ --------
  MTD Daily Rate      ₹30,923.54   ₹52,230.42   98.86%
  Rolling 30d Avg     ₹13,889.94   ₹18,731.82   51.55%
  Ridge                ₹9,981.51   ₹14,660.73   75.95%
  Random Forest        ₹9,689.09   ₹16,202.52   32.90%
  XGBoost              ₹8,560.86   ₹13,986.15   33.16%

The model selected by validation MAE was:

``` text
XGBoost Regressor
```

Held-out chronological test:

``` text
MAE:
₹7,380.37

RMSE:
₹10,226.77

MAPE:
60.70%
```

The validation/test difference shows that forecasting performance needs
continued evaluation on more realistic and larger historical datasets.

Forecasting requires sufficient historical or month-to-date data. An
isolated transaction should not be treated as sufficient evidence for a
monthly forecast.

------------------------------------------------------------------------

# 15. Behavioral Research

The research component investigates:

> **Why do people struggle to track where their money goes each month?**

Current survey sample:

``` text
N = 1,500
```

Important observed associations in the current generated survey dataset
include:

``` text
Payment method count:
Pearson r = 0.6667

Transaction count:
Pearson r = 0.6484

Missed transaction frequency:
Pearson r = 0.6222

Subscription count:
Pearson r = 0.5422
```

OLS regression:

``` text
R² = 0.6547
Adjusted R² = 0.6528
N = 1500
```

Reported coefficients include:

``` text
payment_method_count           +0.1772
transaction_count              +0.0133
subscription_count             +0.1740
missed_transaction_frequency    +0.4088
manual_tracking_frequency       +0.1391
budgeting_frequency             -0.2842
impulse_spending_tendency       +0.1402
income_ordinal                  -0.0409
```

These results represent statistical associations in the current dataset.

They should not be presented as proof of causality. Establishing causal
effects would require a suitable experimental or longitudinal design.

------------------------------------------------------------------------

# 16. User Behavioral Features

The system can construct a user-level feature vector containing:

``` text
monthly_income
monthly_expense
savings_ratio
transaction_count
average_transaction
food_ratio
shopping_ratio
transport_ratio
subscription_ratio
recurring_expense_ratio
spending_variability
weekend_spending_ratio
merchant_count
category_count
category_entropy
```

These features can power:

``` text
Behavior analysis
Forecasting
Segmentation
Personalized insights
```

------------------------------------------------------------------------

# 17. User Segmentation

The segmentation engine uses:

``` text
Feature aggregation
      ↓
StandardScaler
      ↓
Optional PCA
      ↓
K-Means
      ↓
Cluster evaluation
```

The current implementation evaluated:

``` text
k = 2 → 6
```

Current scores:

``` text
k=2
Silhouette = 0.2316
Davies-Bouldin = 1.4928

k=3
Silhouette = 0.2047
Davies-Bouldin = 1.4919

k=4
Silhouette = 0.1836
Davies-Bouldin = 1.4359

k=5
Silhouette = 0.1888
Davies-Bouldin = 1.3549

k=6
Silhouette = 0.1968
Davies-Bouldin = 1.2375
```

The current implementation selected:

``` text
k = 2
```

Current descriptive personas:

``` text
Cluster 0:
Discretionary Weekend & Lifestyle Spender

Cluster 1:
High-Savings Structured Planner
```

These should be treated as descriptive behavioral segments, not
psychological diagnoses or fixed identities.

------------------------------------------------------------------------

# 18. Financial Intelligence Layer

The individual ML models answer:

> "What happened to this transaction?"

The Financial Intelligence layer answers:

> "What does the user's transaction history tell us about their
> financial activity?"

Pipeline:

``` text
Categorization
       +
Anomaly Detection
       +
Recurring Detection
       +
Forecasting
       +
Behavioral Features
       ↓
Financial Intelligence
```

Example structured result:

``` json
{
  "categorization": {
    "category": "EXPENSE",
    "subcategory": "Food",
    "category_confidence": 0.94
  },
  "anomaly_detection": {
    "is_anomaly": true,
    "anomaly_score": 0.87,
    "reason": "Amount significantly exceeds user's historical category spending"
  },
  "recurring_expense": {
    "is_recurring": false
  },
  "forecast": {
    "status": "available"
  }
}
```

------------------------------------------------------------------------

# 19. Financial Intelligence Report

The system can aggregate model outputs into a user-facing report.

The report should contain:

``` text
1. User / analysis metadata
2. Income summary
3. Expense summary
4. Spending breakdown
5. Anomaly analysis
6. Recurring expenses
7. Spending forecast
8. Behavioral signals
9. User segmentation
10. Tracking-difficulty signals
11. Model/data warnings
12. Key observations
13. Structured JSON output
```

Example:

``` text
FINANCIAL INTELLIGENCE REPORT

Total Income:
₹85,000

Total Expenses:
₹42,000

Largest Spending Category:
Rent & Housing

Anomalous Transactions:
3

Recurring Expenses:
4

Low-confidence Classifications:
2

Forecast:
Available / Insufficient Data
```

The report layer should aggregate existing model outputs. It should not
become another ML model.

------------------------------------------------------------------------

# 20. AI Orchestrator

The AI Orchestrator connects the deterministic financial intelligence
system to the LLM.

Example user query:

> "Why did I spend more this month?"

The system should execute:

``` text
User Question
      ↓
Intent Detection
      ↓
Determine Required Tools
      ↓
Retrieve User-Specific Data
      ↓
Run Financial Analytics
      ↓
Run Required ML Modules
      ↓
Build Structured Context
      ↓
Local LLM
      ↓
Natural-Language Explanation
```

Example:

``` text
User:
"Why did I spend more this month?"

        ↓

Intent:
MONTHLY_SPENDING_ANALYSIS

        ↓

Tools:
get_monthly_spending()
compare_previous_month()
get_category_breakdown()
get_anomalies()

        ↓

Financial Intelligence

        ↓

Structured JSON

        ↓

Local LLM

        ↓

Personalized Explanation
```

The LLM should explain verified results rather than independently
calculating financial values.

------------------------------------------------------------------------

# 21. Local LLM Architecture

The recommended deployment separates application infrastructure from
model inference.

``` text
Hostinger KVM 2
│
├── FastAPI
├── ML / Data Science services
├── AI Orchestrator
└── Application services
        │
        │ Secure connection
        ▼
   GPU Inference Server
        │
        ├── vLLM
        │
        └── Open-weight LLM
```

For smaller CPU-oriented inference:

``` text
llama.cpp
```

For GPU inference:

``` text
vLLM
```

The inference server should not be exposed unnecessarily to the public
internet. Service authentication and secure networking should be used.

------------------------------------------------------------------------

# 22. End-to-End AI Request

Example:

``` text
USER
"Why did I spend more this month?"
        │
        ▼
React
        │
        ▼
Cloudflare
        │
        ▼
FastAPI
        │
        ▼
Authenticated user_id
        │
        ▼
AI Orchestrator
        │
        ├── Retrieve user's transactions
        ├── Retrieve user's history
        ├── Calculate spending comparison
        ├── Get category breakdown
        ├── Get anomalies
        └── Get forecast
        │
        ▼
User-Specific Financial Intelligence
        │
        ▼
Structured Context
        │
        ▼
Local LLM
        │
        ▼
Personalized Explanation
        │
        ▼
React
        │
        ▼
USER
```

------------------------------------------------------------------------

# 23. User Isolation Architecture

User isolation is a fundamental security and correctness requirement.

Every user-specific operation must be scoped by `user_id`.

Conceptually:

``` python
get_transactions(user_id)
get_user_profile(user_id)
get_user_baseline(user_id)
get_user_features(user_id)
generate_user_report(user_id)
build_user_llm_context(user_id)
```

Database queries should follow the same principle:

``` sql
SELECT *
FROM transactions
WHERE user_id = current_user_id;
```

and not:

``` sql
SELECT *
FROM transactions;
```

The same isolation principle applies to:

``` text
Transactions
Profiles
Baselines
Anomalies
Recurring expenses
Forecasts
Behavioral features
Segmentation inputs
Reports
AI context
Conversation history
```

A global ML model can be shared, but personal financial state must
remain user-scoped.

------------------------------------------------------------------------

# 24. Inference Testing

The project contains a manual inference test tool:

``` bash
python3 test.py
```

Demo mode:

``` bash
python3 test.py --demo
```

Debug mode:

``` bash
python3 test.py --debug
```

CSV testing:

``` bash
python3 test.py --csv path/to/test_transactions.csv
```

The test tool loads existing artifacts and does not retrain the models.

------------------------------------------------------------------------

# 25. Example Manual Test

Example input:

``` text
User ID:
USR_0001

Merchant:
SWIGGY*BANGALORE FOOD

Description:
Lunch delivery biryani coke

Amount:
₹650

Payment:
UPI
```

Possible output:

``` text
CLASSIFICATION
Category:
EXPENSE

Subcategory:
Food

Confidence:
83.66%

ANOMALY
Is Anomaly:
FALSE

Score:
0.3531

RECURRING:
FALSE
```

The output should always distinguish model predictions from certainty.

------------------------------------------------------------------------

# 26. Important Model Behavior

The system should explicitly handle uncertain predictions.

For example:

``` text
APOLLO HOSPITALS EMERGENCY CARE
₹55,000

Predicted:
Rent & Housing

Confidence:
46.15%
```

The system should report:

``` text
LOW CONFIDENCE
REQUIRES REVIEW
```

rather than treating the prediction as ground truth.

This is especially important because an incorrect category can affect
downstream anomaly explanations.

------------------------------------------------------------------------

# 27. Important Anomaly Considerations

Anomaly detection should be transaction-type aware.

For example, salary income should not necessarily be compared against
expense distributions.

Preferred conceptual structure:

``` text
Transaction
     │
     ├── INCOME
     │      ↓
     │   Income baseline
     │
     ├── EXPENSE
     │      ↓
     │   Category/user baseline
     │
     └── TRANSFER
            ↓
        Transfer baseline
```

A ₹85,000 salary should be evaluated against the user's income history,
not against food, shopping, or rent transactions.

Similarly, a recurring Netflix payment should be evaluated in the
context of the user's subscription history.

------------------------------------------------------------------------

# 28. Model Artifacts

Suggested artifact structure:

``` text
models/
├── transaction_classifier/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   ├── label_encoder.pkl
│   └── metadata.json
│
├── anomaly_detector/
│   ├── model.pkl
│   ├── preprocessing.pkl
│   └── metadata.json
│
├── forecasting/
│   ├── model.pkl
│   ├── preprocessing.pkl
│   └── metadata.json
│
└── segmentation/
    ├── model.pkl
    ├── scaler.pkl
    └── metadata.json
```

Metadata should ideally include:

``` text
Model name
Model version
Training dataset
Features
Training date
Evaluation metrics
Category mapping version
Preprocessing version
```

------------------------------------------------------------------------

# 29. Recommended Project Structure

``` text
DataScience/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── test_results/
│
├── models/
│   ├── transaction_classifier/
│   ├── anomaly_detector/
│   ├── forecasting/
│   └── segmentation/
│
├── src/
│   ├── preprocessing/
│   ├── classification/
│   ├── anomaly/
│   ├── recurring/
│   ├── forecasting/
│   ├── behavior/
│   ├── segmentation/
│   └── orchestration/
│
├── tests/
│   └── test_user_isolation.py
│
├── notebooks/
│   ├── EDA/
│   ├── classification/
│   ├── anomaly/
│   ├── forecasting/
│   └── behavior/
│
├── run_all_pipelines.py
├── test.py
├── requirements.txt
└── README.md
```

The exact structure may differ depending on the current implementation.

------------------------------------------------------------------------

# 30. End-to-End Data Flow

``` text
                  RAW DATA
                     │
                     ▼
             DATA VALIDATION
                     │
                     ▼
             DATA PREPROCESSING
                     │
                     ▼
             FEATURE ENGINEERING
                     │
          ┌──────────┼───────────┐
          │          │           │
          ▼          ▼           ▼
    CLASSIFIER    ANOMALY    FORECASTER
          │          │           │
          └──────────┼───────────┘
                     ▼
            RECURRING DETECTOR
                     │
                     ▼
             USER FEATURES
                     │
                     ▼
              SEGMENTATION
                     │
                     ▼
        FINANCIAL INTELLIGENCE
                     │
                     ▼
             AI ORCHESTRATOR
                     │
                     ▼
                LOCAL LLM
                     │
                     ▼
         PERSONALIZED INSIGHT
                     │
                     ▼
                   USER
```

------------------------------------------------------------------------

# 31. Why the LLM Is Not the Financial Calculator

The architecture intentionally separates calculation from language
generation.

For example:

``` text
Backend / ML:
Previous spending = ₹31,000
Current spending = ₹39,500
Difference = ₹8,500
```

The LLM receives:

``` json
{
  "previous_spending": 31000,
  "current_spending": 39500,
  "difference": 8500
}
```

and explains it.

This reduces the risk of:

``` text
LLM hallucinated calculation
LLM invented transaction
LLM invented category
LLM invented spending trend
```

The financial intelligence layer remains the source of structured facts.

------------------------------------------------------------------------

# 32. Security and Privacy Principles

Financial data is sensitive.

The architecture should follow:

``` text
Authentication
      ↓
Authorization
      ↓
User-scoped database access
      ↓
User-scoped ML context
      ↓
User-scoped AI context
```

Important requirements:

-   Never expose another user's transactions.
-   Never use another user's history for personal baselines.
-   Do not send unrelated users' data to the LLM.
-   Use secure authentication.
-   Use HTTPS/TLS.
-   Apply rate limiting.
-   Protect APIs from unauthorized access.
-   Use database-level access controls where supported.
-   Minimize stored personal information.
-   Avoid logging raw financial data unnecessarily.

------------------------------------------------------------------------

# 33. Current Pipeline Validation

The current complete pipeline has successfully executed all eight Data
Science stages.

Current generated data:

``` text
Classification:
6,000 records

Historical:
20,393 records
40 users
8 months

Survey:
1,500 records
```

The complete pipeline reported successful execution of:

``` text
Dataset generation
Preprocessing
Classification
Anomaly detection
Recurring detection
Forecasting
Behavioral research
Segmentation
Financial Intelligence Orchestrator
```

The current end-to-end execution completed in approximately:

``` text
15.36 seconds
```

This timing is for the current local/generated benchmark pipeline and
should not be interpreted as production latency.

------------------------------------------------------------------------

# 34. Current Manual Demo Findings

The manual test tool has demonstrated that the system can process
examples such as:

``` text
Normal grocery purchase
Food delivery
Recurring subscription
Salary credit
Mutual fund SIP
Large food-related transaction
Unknown merchant
High-value healthcare transaction
```

The tests also revealed useful areas for improvement:

``` text
1. Some ambiguous merchants receive low classification confidence.

2. Some transaction categories are incorrectly predicted for
   unusual merchants.

3. Anomaly detection should be more explicitly aware of
   transaction type.

4. Forecast confidence should not be hard-coded unless calibrated.

5. Forecasting requires sufficient historical data.

6. Synthetic benchmark performance should be validated on
   external/realistic transaction data.

7. User isolation must be explicitly tested for cross-user
   contamination.
```

These are development findings, not failures of the overall
architecture.

------------------------------------------------------------------------

# 35. Development Roadmap

## Phase 1 --- Data Science Foundation

``` text
✓ Dataset generation
✓ Data preprocessing
✓ Leakage-proof splitting
✓ Feature engineering
✓ Classification
✓ Anomaly detection
✓ Recurring detection
✓ Forecasting
✓ Behavioral research
✓ Segmentation
```

## Phase 2 --- Robustness

``` text
→ External/realistic transaction validation
→ Error analysis
→ Confidence calibration
→ Transaction-type-aware anomaly detection
→ Forecast uncertainty validation
→ User isolation testing
→ Data leakage auditing
```

## Phase 3 --- Financial Intelligence

``` text
→ Unified financial report
→ User-specific baselines
→ Personalized behavioral features
→ User-specific financial context
→ Structured intelligence API
```

## Phase 4 --- AI

``` text
→ AI Orchestrator
→ Intent detection
→ Tool calling
→ Context construction
→ Local/open-weight LLM
→ Personalized financial explanations
```

## Phase 5 --- Application

``` text
→ FastAPI
→ Supabase/PostgreSQL
→ React
→ Cloudflare
→ Authentication
→ User dashboard
→ AI chat
```

## Phase 6 --- Production

``` text
→ Monitoring
→ Model versioning
→ Model drift detection
→ Data quality monitoring
→ Secure inference
→ Privacy controls
→ Backup/recovery
→ Performance optimization
```

------------------------------------------------------------------------

# 36. Final Project Pipeline

The complete project can be summarized as:

``` text
                           USER
                             │
                             ▼
                         REACT UI
                             │
                             ▼
                        CLOUDFLARE
                             │
                             ▼
                          FASTAPI
                             │
                    Authenticated user_id
                             │
                             ▼
                     USER-SCOPED DATA
                             │
                             ▼
                    SUPABASE / POSTGRES
                             │
                             ▼
                    DATA SCIENCE LAYER
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   Categorization         Anomaly             Forecast
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                  Recurring Detection
                             │
                             ▼
                    Behavioral Features
                             │
                             ▼
                      Segmentation
                             │
                             ▼
                FINANCIAL INTELLIGENCE
                             │
                             ▼
                    AI ORCHESTRATOR
                             │
                    User-specific context
                             │
                             ▼
                     LOCAL LLM
                             │
                             ▼
                 PERSONALIZED INSIGHT
                             │
                             ▼
                          REACT UI
                             │
                             ▼
                           USER
```

------------------------------------------------------------------------

# 37. Core Design Principle

The project can be summarized in one statement:

> **The system combines transaction-level machine learning, statistical
> behavioral analysis, personalized user baselines, financial
> forecasting, and a local/open-weight LLM to transform raw financial
> transactions into user-specific financial intelligence and
> understandable insights.**

The most important architectural distinction is:

``` text
GLOBAL MODEL
    ≠
GLOBAL USER DATA
```

The ML models can be shared.

The financial state cannot be shared.

``` text
User A transactions
        ↓
User A intelligence

User B transactions
        ↓
User B intelligence
```

This creates a personalized but user-isolated financial intelligence
system.
