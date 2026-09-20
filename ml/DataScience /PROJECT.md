# AI-Based Personal Finance Tracking & Intelligence System
## Data Science & AI/ML Module (Indian Context - INR ₹)

---

## 1. Executive Overview & Core Research Problem

### The Core Question
> **"Why do people struggle to track where their money goes each month?"**

Modern consumers, particularly in dynamic digital economies like **India**, face extreme financial tracking friction driven by:
1. **Multi-Channel Payment Fragmentation**: Active simultaneous usage of multiple UPI apps (PhonePe, Google Pay, Paytm, CRED), credit cards, debit cards, and net banking creates disconnected data silos.
2. **High-Velocity Micro-Transactions**: Frictionless UPI QR code payments (₹10–₹500 for tea, snacks, rides) cause severe cognitive fatigue and forgotten transactions.
3. **Subscription Blindness**: Auto-debited monthly subscriptions (OTT, cloud storage, broadband, gym) accumulate silently.
4. **Manual Tracking Breakdown**: Reliance on spreadsheets, notes, or mental math collapses once transaction volumes exceed 35–40 transactions per month.

This module delivers a production-grade, reproducible Data Science and Machine Learning engine that automatically categorizes transactions, detects contextual anomalies, discovers recurring commitments, forecasts month-end spending, evaluates user behavioral personas, and answers the core research question with empirical statistical rigor.

---

## 2. Standardized Taxonomy (Indian Ecosystem)

All transactions map strictly into a 3-tier hierarchy with 28 standardized categories:

```
├── INCOME
│   ├── Salary (TCS, Infosys, Corporate Payroll NEFT/IMPS)
│   ├── Freelance (Upwork, Fiverr, Razorpay invoices)
│   ├── Business Income (BharatPe, Paytm QR settlements)
│   ├── Interest (HDFC/SBI Fixed Deposit interest, SB interest)
│   └── Other Income (Tax refunds, GPay/CRED rewards)
│
├── EXPENSE
│   ├── Food (Swiggy, Zomato, Starbucks, Chai Point, Haldiram's)
│   ├── Groceries (Zepto, Blinkit, Instamart, BigBasket, DMart)
│   ├── Shopping (Amazon IN, Flipkart, Myntra, Nykaa, Croma)
│   ├── Transport (Uber, Ola, Rapido, Metro DMRC, FASTag)
│   ├── Fuel (IndianOil BPCL, HPCL, Shell)
│   ├── Bills & Utilities (Electricity BESCOM, ACT Fibernet, Airtel/Jio)
│   ├── Rent & Housing (NoBroker, CRED RentPay, Society Maintenance)
│   ├── Healthcare (Apollo Pharmacy, Tata 1mg, Dr Lal Pathlabs)
│   ├── Education (PhysicsWallah, Unacademy, School fees)
│   ├── Entertainment (BookMyShow, Steam, PVR Inox)
│   ├── Travel (IRCTC Train, MakeMyTrip, IndiGo Flights)
│   ├── Subscriptions (Netflix IN, Spotify IN, Hotstar, Prime)
│   ├── Insurance (HDFC Ergo, LIC, Acko)
│   ├── Personal Care (Urban Company Salon, Cult.fit Gym)
│   ├── Gifts & Donations (PM Cares, CRY)
│   ├── Fees & Charges (ATM surcharge, card fees, penalties)
│   └── Other Expense (Laundry, home repairs)
│
└── TRANSFER
    ├── Bank Transfer (PhonePe P2P UPI, GPay transfer, NEFT)
    ├── Credit Card Payment (CRED bill pay, HDFC Autopay)
    ├── Investment (Groww Mutual Fund SIP, Zerodha Demat)
    ├── Savings (SBI Recurring Deposit, Fixed Deposit)
    ├── Cash Withdrawal (SBI/HDFC ATM cashout)
    └── Other Transfer (Paytm wallet load)
```

---

## 3. Machine Learning Architecture & Modules

```
                                  DATA DIRECTORY (/data)
                 ┌───────────────────────────┼───────────────────────────┐
                 ▼                           ▼                           ▼
          /data/raw/                  /data/processed/            /data/train,val,test/
     (Immutable Datasets)          (Cleaned & Normalized)       (Stratified & Temporal)
                 │                           │                           │
                 └───────────────────────────┼───────────────────────────┘
                                             │
                                             ▼
                             SUPERVISED TRANSACTION CLASSIFIER
                        (TF-IDF N-grams + OHE + Scaler + Logistic/XGB)
                                             │
                                             ▼
                                CONTEXTUAL ANOMALY DETECTOR
                        (Isolation Forest with User-Category Baselines)
                                             │
                                             ▼
                               DETERMINISTIC RECURRING ENGINE
                         (Cadence Analysis & Next Billing Prediction)
                                             │
                                             ▼
                                SPENDING FORECASTING ENGINE
                         (Chronological XGBoost with Rolling MTD/7d/30d)
                                             │
                                             ▼
                                 USER BEHAVIOR SEGMENTATION
                            (15-D Vectors + PCA + K-Means)
                                             │
                                             ▼
                               FINANCIAL INTELLIGENCE ORCHESTRATOR
                             (Structured JSON Inference Payloads)
```

---

## 4. Multi-Tenant User-Isolation Architecture
 
```
                         AUTHENTICATED USER
                                 │
                                 ▼
                             user_id
                                 │
                                 ▼
                         DATA ACCESS LAYER
                   (src/data/user_data_access.py)
                                 │
                  ┌─────────────┴─────────────┐
                  │                           │
               USER A                      USER B
                  │                           │
                  ▼                           ▼
          A's Transactions             B's Transactions
                  │                           │
                  ▼                           ▼
          A's User Profile             B's User Profile
                  │                           │
                  ▼                           ▼
          A's Personal Baseline       B's Personal Baseline
                  │                           │
                  ▼                           ▼
          A's ML Inference             B's ML Inference
                  │                           │
                  ▼                           ▼
          A's Financial Intel.         B's Financial Intel.
                  │                           │
                  ▼                           ▼
              A's LLM Context             B's LLM Context
                  │                           │
                  ▼                           ▼
               A's Report                  B's Report
```

### Global Model Sharing vs User State Isolation
- **Global Models (Shared Weights)**: `TransactionClassifier` (TF-IDF + LR/XGBoost), `ContextualAnomalyDetector` (Isolation Forest), `SpendingForecaster` (XGBoost Regressor), `UserSegmenter` (K-Means).
- **User Financial State (Strictly Isolated)**: Transactions, Category Baselines, Recurring Streams, Behavioral Profiles, Anomaly Assessments, Forecast Inputs, LLM Contexts, Conversation Memory.

---

## 5. Key Rules & Constraints

1. **No LLM Training on Financial Transactions**: Supervised ML models (TF-IDF, Logistic Regression, XGBoost) are utilized for deterministic, low-latency, and interpretable classification.
2. **Immutable Raw Data**: Files in `/data/raw/` are never overwritten or modified during pipeline runs.
3. **Strict Temporal Splitting**: Time-series forecasting and longitudinal evaluations enforce strict chronological splits (Months 1–5 Train $\rightarrow$ Month 6 Validation $\rightarrow$ Months 7–8 Test) to prevent lookahead data leakage.
4. **Contextual Anomaly Logic**: Anomalies are never simple absolute amount thresholds. A ₹28,000 monthly rent is normal for a user paying regular rent, whereas a ₹11,500 restaurant dinner for someone averaging ₹350/meal triggers an anomaly with transparent explanations.
5. **Standardized Model Metadata**: Every model directory in `/models/` must contain `metadata.json` with model name, version, feature list, training dataset, category mapping version, preprocessing version, and evaluation metrics.
6. **User Isolation Enforced**: Multi-tenant data access layer ensures zero cross-user transaction leakage or baseline pollution. Verified via `tests/test_user_isolation.py`.
