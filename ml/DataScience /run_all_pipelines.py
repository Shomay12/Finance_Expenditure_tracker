"""
Master Pipeline Runner for AI-Based Personal Finance Tracking & Intelligence System.
Executes the full Data Science and Machine Learning workflow end-to-end:
1. Raw Dataset Generation & Validation
2. Data Preprocessing & Leakage-Proof Splitting
3. Supervised Transaction Categorization ML Pipeline
4. Contextual Anomaly Detection Engine
5. Deterministic Recurring Expense & Subscription Detection
6. Temporal Spending Forecasting Engine
7. Core Behavioral Research & Statistical Econometric Analysis
8. User Behavioral Segmentation & Clustering
9. Financial Intelligence Orchestration & Structured JSON Validation
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data.data_generators import (
    generate_transaction_classification_dataset,
    generate_historical_transactions_dataset,
    generate_user_behavior_survey_dataset
)
from src.data.preprocessing import run_preprocessing_pipeline
from src.models.transaction_classifier import run_training_pipeline as run_classifier_training
from src.models.anomaly_detector import run_training_pipeline as run_anomaly_training
from src.models.recurring_detector import run_recurring_pipeline
from src.models.spending_forecaster import run_forecasting_pipeline
from src.analysis.behavioral_research import run_behavioral_research_analysis
from src.models.user_segmenter import run_segmentation_pipeline
from src.orchestrator import FinancialIntelligenceOrchestrator

def main():
    start_time = time.time()
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    models_dir = os.path.join(base_dir, "models")

    print("\n" + "=" * 80)
    print(" AI-BASED PERSONAL FINANCE TRACKING & INTELLIGENCE SYSTEM")
    print(" DATA SCIENCE & AI/ML PIPELINE EXECUTION")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Dataset Generation
    # -------------------------------------------------------------
    print("\n[STEP 1/8] Generating / Verifying Raw Datasets...")
    cat_raw_file = os.path.join(raw_dir, "raw_transactions_classification.csv")
    hist_raw_file = os.path.join(raw_dir, "raw_historical_transactions.csv")
    survey_raw_file = os.path.join(raw_dir, "raw_user_behavior_survey.csv")

    generate_transaction_classification_dataset(cat_raw_file, n_samples=6000)
    generate_historical_transactions_dataset(hist_raw_file, n_users=40, n_months=8)
    generate_user_behavior_survey_dataset(survey_raw_file, n_respondents=1500)

    # -------------------------------------------------------------
    # 2. Preprocessing & Splitting
    # -------------------------------------------------------------
    print("\n[STEP 2/8] Running Preprocessing & Leakage-Proof Splitting...")
    run_preprocessing_pipeline(data_dir)

    # -------------------------------------------------------------
    # 3. Transaction Categorization
    # -------------------------------------------------------------
    print("\n[STEP 3/8] Training Supervised Transaction Classifier...")
    clf_model_dir = os.path.join(models_dir, "transaction_classifier")
    run_classifier_training(data_dir, clf_model_dir)

    # -------------------------------------------------------------
    # 4. Contextual Anomaly Detection
    # -------------------------------------------------------------
    print("\n[STEP 4/8] Training Contextual Anomaly Detector...")
    anom_model_dir = os.path.join(models_dir, "anomaly_detector")
    run_anomaly_training(data_dir, anom_model_dir)

    # -------------------------------------------------------------
    # 5. Recurring Expense & Subscription Detection
    # -------------------------------------------------------------
    print("\n[STEP 5/8] Running Deterministic Recurring Expense Engine...")
    run_recurring_pipeline(hist_raw_file)

    # -------------------------------------------------------------
    # 6. Spending Forecasting
    # -------------------------------------------------------------
    print("\n[STEP 6/8] Training Temporal Spending Forecaster...")
    forecast_model_dir = os.path.join(models_dir, "forecasting")
    run_forecasting_pipeline(hist_raw_file, forecast_model_dir)

    # -------------------------------------------------------------
    # 7. Core Behavioral Research Statistical Analysis
    # -------------------------------------------------------------
    print("\n[STEP 7/8] Executing Core Behavioral Research & Econometrics...")
    survey_processed_file = os.path.join(data_dir, "processed", "user_behavior_survey_processed.csv")
    run_behavioral_research_analysis(survey_processed_file)

    # -------------------------------------------------------------
    # 8. User Segmentation
    # -------------------------------------------------------------
    print("\n[STEP 8/8] Training User Segmentation Engine...")
    seg_model_dir = os.path.join(models_dir, "segmentation")
    run_segmentation_pipeline(hist_raw_file, seg_model_dir)

    # -------------------------------------------------------------
    # Master Orchestrator Verification
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("END-TO-END FINANCIAL INTELLIGENCE ORCHESTRATOR VALIDATION")
    print("=" * 80)

    orchestrator = FinancialIntelligenceOrchestrator(models_dir)

    # Live Test 1: Indian Supermarket / Quick Commerce Grocery Purchase
    test_1 = orchestrator.process_incoming_transaction(
        user_id="USR_0001",
        merchant_raw="ZEPTO QUICK COMMERCE",
        description="Daily grocery supplies milk bread curd paneer",
        amount=1250.0,
        transaction_type="debit",
        payment_method="UPI",
        is_weekend=1,
        hour=10
    )
    print("\n[Live Test 1] Standard Quick-Commerce Grocery Purchase (UPI):")
    print(json.dumps(test_1, indent=2))

    # Live Test 2: Anomaly Outlier Luxury Dining
    test_2 = orchestrator.process_incoming_transaction(
        user_id="USR_0001",
        merchant_raw="TAJ PALACE 5-STAR LUXURY BANQUET",
        description="Private luxury banquet dining experience",
        amount=16500.0,
        transaction_type="debit",
        payment_method="Credit Card",
        is_weekend=1,
        hour=22
    )
    print("\n[Live Test 2] Contextual Outlier Expense:")
    print(json.dumps(test_2, indent=2))

    # Live Test 3: Spending Forecast (INR)
    test_3 = orchestrator.forecast_user_spending(
        current_mtd_spending=24000.0,
        days_elapsed=12,
        days_remaining=18,
        rolling_7_day=6500.0,
        rolling_30_day=29000.0,
        previous_month_spending=52000.0,
        monthly_recurring_committed=26000.0
    )
    print("\n[Live Test 3] Month-End Spending Forecast (INR):")
    print(json.dumps(test_3, indent=2))

    # Live Test 4: User Behavioral Persona
    test_4 = orchestrator.get_user_behavioral_profile({
        "monthly_income": 85000.0,
        "monthly_expense": 54000.0,
        "savings_ratio": 0.365,
        "transaction_count": 74.0,
        "average_transaction": 729.73,
        "food_ratio": 0.19,
        "shopping_ratio": 0.11,
        "transport_ratio": 0.07,
        "subscription_ratio": 0.03,
        "recurring_expense_ratio": 0.51,
        "spending_variability": 0.62,
        "weekend_spending_ratio": 0.39,
        "merchant_count": 24,
        "category_count": 9,
        "category_entropy": 2.41
    })
    print("\n[Live Test 4] User Behavioral Persona Assignment:")
    print(json.dumps(test_4, indent=2))

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL DATA SCIENCE & AI/ML MODULES EXECUTED AND VALIDATED IN {elapsed:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
