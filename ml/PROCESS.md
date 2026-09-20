# AI-Based Personal Finance Tracking & Intelligence System
## System Execution, Process & Workflow Guide

---

## 1. End-to-End System Workflow

```
                        ┌─────────────────────────────────────────┐
                        │      RAW TRANSACTION INGESTION          │
                        │  (Bank SMS, UPI QR, Statements, Cards)  │
                        └───────────────────┬─────────────────────┘
                                            │
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │    USER DATA ACCESS LAYER (Isolated)    │
                        │  (Strict user_id partition verification)│
                        └───────────────────┬─────────────────────┘
                                            │
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │   DATA SCIENCE & ML INFERENCE PIPELINE  │
                        │  - Category Classification (LR/XGB)     │
                        │  - Contextual Anomaly Detection (IF)    │
                        │  - Recurring Cadence Discovery          │
                        │  - Spending Forecast (XGBoost Regressor)│
                        │  - Behavioral Segmentation (K-Means)    │
                        └───────────────────┬─────────────────────┘
                                            │
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │ STRUCTURED FINANCIAL INTEL CONTEXT (JSON│
                        │  (Section 19 Structured Payload Schema) │
                        └───────────────────┬─────────────────────┘
                                            │
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │     AI ORCHESTRATOR & LLM AGENT         │
                        │  - Prompt Injection Sanitization Shield │
                        │  - System Directives (24 Strict Rules)  │
                        │  - User-Isolated Session Chat Memory    │
                        │  - Groq LLM (llama-3.3-70b-versatile)   │
                        └───────────────────┬─────────────────────┘
                                            │
                                            ▼
                        ┌─────────────────────────────────────────┐
                        │     EVIDENCE-BASED USER RESPONSE        │
                        │  (Explain, contextualize, never invent) │
                        └─────────────────────────────────────────┘
```

---

## 2. Model Training & Re-Training Process (DataScience Module)

### Step 1: Synthetic Data Generation
Generates 12,000+ realistic Indian transactions across 10 distinct user personas:
```bash
cd "DataScience "
python3 src/data/data_generators.py
```

### Step 2: Feature Engineering & Preprocessing
Executes text normalization, TF-IDF n-grams extraction, temporal/cyclical transformations, and user-category baseline calculations with strict chronological splits:
```bash
python3 src/data/preprocessing.py
```

### Step 3: Train All ML Models
Trains and validates:
1. `TransactionClassifier` (TF-IDF + Logistic Regression / XGBoost)
2. `ContextualAnomalyDetector` (Isolation Forest + User-Category Baselines)
3. `RecurringExpenseDetector` (Cadence Clustering & Next Billing Date)
4. `SpendingForecaster` (Chronological XGBoost Regressor)
5. `UserSegmenter` (PCA + K-Means Clustering)
```bash
python3 run_all_pipelines.py
```

### Step 4: Validate Data Science Inference
Run the standalone inference and financial report generator:
```bash
python3 test.py --demo
```

---

## 3. LLM Agent Operations & Execution Process (LLM_Agent Module)

### Configuration & Environment Setup
The agent connects to Groq Cloud using high-performance models (`llama-3.3-70b-versatile` / `qwen/qwen3.8-27b`).
API keys are loaded securely from the `.env` file or environment variables.

Copy the `.env.example` file to `.env` and provide your Groq API key:
```bash
cp .env.example .env
# Edit .env and set:
# GROQ_API_KEY="gsk_your_groq_api_key_here"
```
Or export it directly:
```bash
export GROQ_API_KEY="gsk_your_groq_api_key_here"
```

### Running the Interactive Agent Console Connected to Test Data
Allows interactive chat with the financial intelligence assistant against the latest test report from `data/test_results/`:
```bash
python3 LLM_Agent/cli.py --interactive --test-data
```

### Asking Single Questions Directly Against Test Report
```bash
python3 LLM_Agent/cli.py --test-data --query "Where did my money go in this test run and which transactions look unusual?"
```

### Running Interactive Agent for a Specific Profile
```bash
python3 LLM_Agent/cli.py --interactive --user USR_0001
```

### Running the Preset 8-Scenario Benchmark Suite
Runs automated evaluations testing key user queries:
1. Where is my money going? (Category spending breakdown)
2. Why did I spend more this month? (Month-to-month variance)
3. Why was my transaction flagged? (Contextual anomaly explanation without fraud accusations)
4. What subscriptions do I have? (Recurring expense vs investment cadence)
5. What is my predicted month-end spending? (Forecasting with limitations)
6. What behavioral traits are visible? (Neutral behavioral profile)
7. Bank balance inquiry (Handling out-of-scope / missing data)
8. Prompt injection attempt (Attempting to override system prompt via transaction text)

```bash
python3 LLM_Agent/cli.py --demo
```

### Running Automated Unit & Security Tests
Executes the comprehensive automated test suite verifying user isolation, injection defense, and Groq API connectivity:
```bash
python3 LLM_Agent/tests/test_agent.py
```

---

## 4. Multi-Tenant User Isolation & Security Protocol

1. **Authentication Token $\rightarrow$ User Context**: Every inference and LLM prompt must be bound to a single verified `user_id`.
2. **Memory Isolation**: Conversation histories in `MemoryManager` are indexed by `(user_id, session_id)` tuples. No memory lookups cross user keys.
3. **Prompt Injection Sanitization**: Financial transaction fields (merchant, description, memo) are treated strictly as JSON string literals inside user data payloads. The LLM is strictly instructed never to follow directives embedded in data fields.
4. **Credential Redaction**: Secrets, API keys, and internal backend prompts are never echoed in assistant responses.
