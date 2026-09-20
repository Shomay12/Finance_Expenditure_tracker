#!/usr/bin/env python3
"""
================================================================================
FINANCIAL AI MODEL TEST CONSOLE & INTELLIGENCE REPORT GENERATOR (test.py)
================================================================================
Inference testing and aggregated Financial Intelligence reporting CLI for
pre-trained Data Science & AI/ML models of the Personal Finance Tracking &
Intelligence System.

IMPORTANT ARCHITECTURAL RULES:
- Performs ZERO retraining (No fit(), fit_transform(), or training routines).
- Loads existing artifacts from /models/ directory.
- Reuses exact production inference pipelines from src/models/ and src/orchestrator.py.
- The Financial Intelligence Report layer aggregates ML predictions without performing
  new ML tasks.
================================================================================
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.transaction_classifier import TransactionClassifierPipeline
from src.models.anomaly_detector import ContextualAnomalyDetector
from src.models.recurring_detector import RecurringExpenseDetector
from src.models.spending_forecaster import SpendingForecaster
from src.models.user_segmenter import UserSegmenter
from src.orchestrator import FinancialIntelligenceOrchestrator

# Configurable confidence threshold for warnings
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.60


class FinancialIntelligenceReportGenerator:
    """
    Aggregates structured predictions from existing ML modules into a
    comprehensive human-readable and JSON Financial Intelligence Report.
    """

    @staticmethod
    def generate_report(
        transactions_results: list,
        user_id: str = "USR_0001",
        user_segmentation_result: dict = None,
        forecasting_result: dict = None,
        save_report: bool = True
    ) -> tuple:
        """
        Builds the human-readable text report and matching structured JSON report.
        """
        now_ts = datetime.now()
        timestamp_str = now_ts.strftime("%Y%m%d_%H%M%S")
        generated_at_iso = now_ts.isoformat()

        if not transactions_results:
            return "No transactions provided for reporting.", {}

        # ---------------------------------------------------------------------
        # 1. Dataset & Cash Flow Summaries
        # ---------------------------------------------------------------------
        dates = []
        for t in transactions_results:
            d_str = t.get("date", "")
            if d_str:
                try:
                    dates.append(datetime.strptime(d_str.split()[0], "%Y-%m-%d"))
                except Exception:
                    pass

        if dates:
            start_date_str = min(dates).strftime("%Y-%m-%d")
            end_date_str = max(dates).strftime("%Y-%m-%d")
        else:
            start_date_str = now_ts.strftime("%Y-%m-%d")
            end_date_str = start_date_str

        total_income = 0.0
        total_expenses = 0.0
        total_transfers = 0.0

        income_by_category = defaultdict(float)
        expense_by_category = defaultdict(float)
        transfer_by_category = defaultdict(float)

        payment_methods = set()
        weekend_expenses_count = 0
        weekend_expenses_total = 0.0
        total_expense_txns = 0

        anomalies_list = []
        recurring_expenses_list = []
        recurring_investments_list = []
        low_confidence_txns = []

        rec_keywords = ["netflix", "spotify", "broadband", "rent", "lease", "gym", "internet", "airtel", "fibernet", "cult", "sip", "nobroker"]

        for t in transactions_results:
            amt = float(t.get("amount", 0.0))
            pm = t.get("payment_method", "Unknown")
            payment_methods.add(pm)

            cat = t.get("predicted_category", "EXPENSE")
            subcat = t.get("predicted_subcategory", "Other Expense")
            conf = float(t.get("confidence", 1.0))
            is_anom = bool(t.get("is_anomaly", False))
            anom_score = float(t.get("anomaly_score", 0.0))
            anom_reason = t.get("anomaly_reason", "Not provided")
            merch = t.get("merchant", "Unknown")
            desc = t.get("description", "")
            d_str = t.get("date", "")

            # Check if low confidence
            if conf < CLASSIFICATION_CONFIDENCE_THRESHOLD:
                low_confidence_txns.append({
                    "merchant": merch,
                    "predicted_category": f"{cat}::{subcat}",
                    "confidence": conf,
                    "amount": amt
                })

            if cat == "INCOME":
                total_income += amt
                income_by_category[subcat] += amt
            elif cat == "TRANSFER":
                total_transfers += amt
                transfer_by_category[subcat] += amt
            else: # EXPENSE
                total_expenses += amt
                expense_by_category[subcat] += amt
                total_expense_txns += 1
                try:
                    if d_str and datetime.strptime(d_str.split()[0], "%Y-%m-%d").weekday() >= 5:
                        weekend_expenses_count += 1
                        weekend_expenses_total += amt
                except Exception:
                    pass

            # Anomalies
            if is_anom:
                anomalies_list.append({
                    "date": d_str.split()[0] if d_str else "N/A",
                    "merchant": merch,
                    "category": subcat,
                    "amount": amt,
                    "score": anom_score,
                    "reason": anom_reason,
                    "confidence": conf,
                    "low_confidence": conf < CLASSIFICATION_CONFIDENCE_THRESHOLD
                })

            # Recurring detection logic
            is_rec_kw = any(rk in merch.lower() or rk in desc.lower() for rk in rec_keywords)
            if is_rec_kw or subcat in ["Subscriptions", "Rent & Housing", "Bills & Utilities", "Insurance"]:
                if cat == "TRANSFER" or subcat in ["Investment", "Savings"]:
                    recurring_investments_list.append({
                        "merchant": merch,
                        "amount": amt,
                        "category": subcat,
                        "frequency": "Monthly",
                        "type": "transfer_investment"
                    })
                else:
                    recurring_expenses_list.append({
                        "merchant": merch,
                        "amount": amt,
                        "category": subcat,
                        "frequency": "Monthly",
                        "type": "recurring_expense"
                    })

        net_cash_flow = total_income - total_expenses - total_transfers

        # ---------------------------------------------------------------------
        # 2. Financial Behavior Signals
        # ---------------------------------------------------------------------
        n_total_txns = len(transactions_results)
        avg_txn_val = (total_income + total_expenses + total_transfers) / max(1, n_total_txns)
        weekend_spending_ratio = (weekend_expenses_total / total_expenses) if total_expenses > 0 else 0.0

        rec_exp_total = sum(r["amount"] for r in recurring_expenses_list)
        rec_inv_total = sum(r["amount"] for r in recurring_investments_list)

        rec_expense_ratio = (rec_exp_total / total_expenses) if total_expenses > 0 else 0.0
        sub_total = expense_by_category.get("Subscriptions", 0.0)
        subscription_ratio = (sub_total / total_expenses) if total_expenses > 0 else 0.0
        savings_investment_ratio = (total_transfers / total_income) if total_income > 0 else 0.0

        # ---------------------------------------------------------------------
        # 3. Text Report Formatting
        # ---------------------------------------------------------------------
        lines = []
        lines.append("=" * 70)
        lines.append("                    FINANCIAL INTELLIGENCE REPORT")
        lines.append("=" * 70)
        lines.append("\nREPORT OBJECTIVE")
        lines.append("-" * 70)
        lines.append("Analyze transaction behavior to understand where money is being spent,")
        lines.append("identify unusual transactions, detect recurring commitments, and")
        lines.append("estimate future spending where sufficient historical data exists.")

        lines.append("\nUSER")
        lines.append("-" * 70)
        lines.append(f"User ID: {user_id}")

        lines.append("\nDATASET SUMMARY")
        lines.append("-" * 70)
        lines.append(f"Transactions analyzed      : {n_total_txns}")
        lines.append(f"Analysis period            : {start_date_str} → {end_date_str}")
        lines.append(f"Total income               : ₹{total_income:,.2f}")
        lines.append(f"Total expenses             : ₹{total_expenses:,.2f}")
        lines.append(f"Total transfers/investments: ₹{total_transfers:,.2f}")
        lines.append(f"Net cash flow              : ₹{net_cash_flow:,.2f}")

        # Section 1: Spending Breakdown
        lines.append("\n" + "=" * 70)
        lines.append("1. SPENDING BREAKDOWN")
        lines.append("=" * 70)
        sorted_expenses = sorted(expense_by_category.items(), key=lambda x: x[1], reverse=True)
        lines.append(f"{'Category':<28} | {'Amount':>14} | {'% of Expenses':>14}")
        lines.append("-" * 70)
        if sorted_expenses:
            for c_name, c_amt in sorted_expenses:
                pct = (c_amt / total_expenses * 100) if total_expenses > 0 else 0.0
                lines.append(f"{c_name:<28} | ₹{c_amt:>13,.2f} | {pct:>13.1f}%")
        else:
            lines.append("No expense transactions recorded.")
        lines.append("-" * 70)
        lines.append(f"{'TOTAL EXPENSES':<28} | ₹{total_expenses:>13,.2f} | {'100.0%':>14}")

        # Section 2: Income Analysis
        lines.append("\n" + "=" * 70)
        lines.append("2. INCOME ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"Total Income: ₹{total_income:,.2f}\n")
        income_types = ["Salary", "Freelance", "Business Income", "Interest", "Other Income"]
        for it in income_types:
            val = income_by_category.get(it, 0.0)
            lines.append(f"{it:<25}: ₹{val:,.2f}")

        # Section 3: Anomaly Analysis
        lines.append("\n" + "=" * 70)
        lines.append("3. ANOMALY ANALYSIS")
        lines.append("=" * 70)
        anom_rate = (len(anomalies_list) / max(1, n_total_txns)) * 100
        total_anom_val = sum(a["amount"] for a in anomalies_list)
        lines.append(f"Anomalous Transactions: {len(anomalies_list)} / {n_total_txns}")
        lines.append(f"Anomaly Rate          : {anom_rate:.1f}%")
        lines.append(f"Total Anomalous Value : ₹{total_anom_val:,.2f}\n")

        if anomalies_list:
            lines.append(f"{'Date':<11} | {'Merchant':<24} | {'Category':<16} | {'Amount':>10} | {'Score':>6}")
            lines.append("-" * 70)
            for a in anomalies_list:
                cat_display = a['category']
                if a['low_confidence']:
                    cat_display += " (!)"
                lines.append(f"{a['date']:<11} | {a['merchant'][:23]:<24} | {cat_display:<16} | ₹{a['amount']:>9,.0f} | {a['score']:>6.2f}")
                lines.append(f"  Reason: {a['reason']}")
                if a['low_confidence']:
                    lines.append(f"  CATEGORY UNCERTAIN: Confidence is {a['confidence']*100:.1f}%. Requires manual review.")
        else:
            lines.append("No transactions flagged as anomalous.")

        # Section 4: Recurring Expense Analysis
        lines.append("\n" + "=" * 70)
        lines.append("4. RECURRING EXPENSE ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"Recurring Expenses Detected   : {len(recurring_expenses_list)}")
        lines.append(f"Recurring Investments/Transfers: {len(recurring_investments_list)}")
        lines.append(f"Estimated Monthly Commitments : ₹{(rec_exp_total + rec_inv_total):,.2f}\n")

        if recurring_expenses_list:
            lines.append("Recurring Consumption Expenses:")
            for r in recurring_expenses_list:
                lines.append(f"  • {r['merchant']:<30} : ₹{r['amount']:>8,.2f}/month ({r['category']})")

        if recurring_investments_list:
            lines.append("\nRecurring Transfers & Investments (Separated from Consumption):")
            for r in recurring_investments_list:
                lines.append(f"  • {r['merchant']:<30} : ₹{r['amount']:>8,.2f}/month ({r['category']})")

        if not recurring_expenses_list and not recurring_investments_list:
            lines.append("No recurring commitments detected.")

        # Section 5: Spending Forecast
        lines.append("\n" + "=" * 70)
        lines.append("5. SPENDING FORECAST")
        lines.append("=" * 70)
        if forecasting_result and forecasting_result.get("status") != "insufficient_data":
            lines.append(f"Current Spending (MTD)         : ₹{forecasting_result['current_spending']:,.2f}")
            lines.append(f"Predicted Remaining Month Spend: ₹{forecasting_result['predicted_remaining']:,.2f}")
            lines.append(f"Predicted Month-End Total      : ₹{forecasting_result['predicted_month_end']:,.2f}")
            lines.append(f"Model Used                     : {forecasting_result.get('model_name', 'SpendingForecaster')}")
            lines.append("Forecast Confidence            : Not statistically calibrated (heuristic model output)")
        else:
            lines.append("Status: Insufficient data for reliable forecast.")
            lines.append("\nReason:")
            lines.append("Spending forecasting requires longitudinal/month-to-date transaction")
            lines.append("data rather than isolated transactions.")

        # Section 6: Financial Behavior Signals
        lines.append("\n" + "=" * 70)
        lines.append("6. FINANCIAL BEHAVIOR SIGNALS")
        lines.append("=" * 70)
        lines.append(f"Average Transaction Value      : ₹{avg_txn_val:,.2f}")
        lines.append(f"Weekend Spending Ratio         : {weekend_spending_ratio*100:.1f}% of expense value occurred on weekends")
        lines.append(f"Recurring Expense Ratio        : {rec_expense_ratio*100:.1f}% of expenses are committed recurring")
        lines.append(f"Subscription Ratio             : {subscription_ratio*100:.1f}% of expenses dedicated to digital subscriptions")
        lines.append(f"Savings / Investment Ratio     : {savings_investment_ratio*100:.1f}% of income allocated to transfers/investments")

        # Section 7: User Segmentation
        lines.append("\n" + "=" * 70)
        lines.append("7. USER SEGMENTATION")
        lines.append("=" * 70)
        if user_segmentation_result and user_segmentation_result.get("persona_name"):
            lines.append(f"Cluster ID         : {user_segmentation_result.get('cluster_id', 0)}")
            lines.append(f"Segment Persona    : {user_segmentation_result.get('persona_name')}")
            lines.append(f"Behavioral Profile : {user_segmentation_result.get('persona_description')}")
            lines.append("\nKey Observed Features:")
            lines.append(f"  • Weekend Spending Ratio: {user_segmentation_result.get('weekend_spending_ratio', 0.0)}")
            lines.append(f"  • Subscription Ratio    : {user_segmentation_result.get('subscription_ratio', 0.0)}")
            lines.append(f"  • Savings Ratio         : {user_segmentation_result.get('savings_ratio', 0.0)}")
        else:
            lines.append("Status: Segmentation unavailable for this isolated test input.")
            lines.append("Reason: User-level segmentation requires multi-month aggregated user feature vectors.")

        # Section 8: Tracking Difficulty Signals
        lines.append("\n" + "=" * 70)
        lines.append("8. TRACKING DIFFICULTY SIGNALS")
        lines.append("=" * 70)
        lines.append("Measurable transaction complexity indicators (Addressing tracking friction):")
        lines.append(f"  • Active Payment Channels      : {len(payment_methods)} ({', '.join(sorted(payment_methods))})")
        lines.append(f"  • Transaction Volume           : {n_total_txns} transactions")
        lines.append(f"  • Recurring Commitments        : {len(recurring_expenses_list) + len(recurring_investments_list)} active streams")
        lines.append(f"  • Flagged Anomalous Spikes     : {len(anomalies_list)}")
        lines.append(f"  • Low-Confidence / Ambiguous   : {len(low_confidence_txns)}")
        lines.append("\nNote: These transaction characteristics increase the volume and dispersion of")
        lines.append("information a user must reconcile manually.")

        # Section 9: Key Observations
        lines.append("\n" + "=" * 70)
        lines.append("9. KEY OBSERVATIONS")
        lines.append("=" * 70)
        observations = []
        if sorted_expenses:
            top_cat, top_val = sorted_expenses[0]
            top_pct = (top_val / total_expenses * 100) if total_expenses > 0 else 0
            observations.append(f"1. '{top_cat}' represents the largest spending category (₹{top_val:,.2f}, {top_pct:.1f}% of expenses).")
        
        if anomalies_list:
            observations.append(f"2. {len(anomalies_list)} transaction(s) were flagged as unusual relative to historical category baselines.")
        else:
            observations.append("2. All evaluated transactions align with expected historical user baselines.")

        if recurring_expenses_list or recurring_investments_list:
            observations.append(f"3. {len(recurring_expenses_list) + len(recurring_investments_list)} recurring commitment pattern(s) were detected totaling ₹{(rec_exp_total + rec_inv_total):,.2f}/mo.")

        if low_confidence_txns:
            observations.append(f"4. {len(low_confidence_txns)} transaction(s) had classification confidence below 60% and require user review.")

        if not forecasting_result or forecasting_result.get("status") == "insufficient_data":
            observations.append("5. Spending forecasting could not be calculated because isolated test transactions lack longitudinal temporal context.")

        for obs in observations:
            lines.append(obs)

        # Section 10: Model / Data Warnings
        lines.append("\n" + "=" * 70)
        lines.append("10. DATA QUALITY / MODEL WARNINGS")
        lines.append("=" * 70)
        warnings = []
        if low_confidence_txns:
            warnings.append(f"[WARNING] {len(low_confidence_txns)} transaction(s) have classification confidence < 60%. Review required.")
        if any(a['low_confidence'] for a in anomalies_list):
            warnings.append("[WARNING] Some anomaly classifications depend on uncertain category predictions.")
        if not forecasting_result or forecasting_result.get("status") == "insufficient_data":
            warnings.append("[WARNING] Forecast cannot be reliably calculated from isolated test transactions.")
        warnings.append("[INFO] Current test dataset is synthetic/manual and does not establish real-world generalization.")

        for w in warnings:
            lines.append(w)

        # Section 11: Final Summary
        lines.append("\n" + "=" * 70)
        lines.append("11. FINAL FINANCIAL INTELLIGENCE SUMMARY")
        lines.append("=" * 70)
        lines.append(f"The analyzed transactions record a total income of ₹{total_income:,.2f} and expenses of ₹{total_expenses:,.2f}.")
        if sorted_expenses:
            lines.append(f"The primary expenditure driver is '{sorted_expenses[0][0]}' (₹{sorted_expenses[0][1]:,.2f}).")
        if anomalies_list:
            lines.append(f"{len(anomalies_list)} contextual outlier(s) were identified (Total: ₹{total_anom_val:,.2f}) that deviate from user habit.")
        if recurring_expenses_list or recurring_investments_list:
            lines.append(f"Regular recurring commitments total ₹{(rec_exp_total + rec_inv_total):,.2f}/month across {len(recurring_expenses_list) + len(recurring_investments_list)} stream(s).")
        if low_confidence_txns:
            lines.append(f"{len(low_confidence_txns)} ambiguous transaction(s) require manual user verification.")
        lines.append("Longitudinal multi-month history is required to produce statistically calibrated spending forecasts.")
        lines.append("=" * 70)

        report_text = "\n".join(lines)

        # ---------------------------------------------------------------------
        # 4. Structured JSON Report Object
        # ---------------------------------------------------------------------
        json_report = {
            "report_metadata": {
                "user_id": user_id,
                "generated_at": generated_at_iso,
                "transactions_analyzed": n_total_txns,
                "analysis_period": {
                    "start": start_date_str,
                    "end": end_date_str
                }
            },
            "cash_flow": {
                "total_income": round(total_income, 2),
                "total_expenses": round(total_expenses, 2),
                "total_transfers": round(total_transfers, 2),
                "net_cash_flow": round(net_cash_flow, 2)
            },
            "spending_breakdown": [
                {
                    "category": c_name,
                    "amount": round(c_amt, 2),
                    "percentage": round((c_amt / total_expenses * 100), 1) if total_expenses > 0 else 0.0
                }
                for c_name, c_amt in sorted_expenses
            ],
            "income_breakdown": [
                {
                    "source": it,
                    "amount": round(income_by_category.get(it, 0.0), 2)
                }
                for it in income_types if income_by_category.get(it, 0.0) > 0
            ],
            "anomalies": [
                {
                    "date": a["date"],
                    "merchant": a["merchant"],
                    "category": a["category"],
                    "amount": round(a["amount"], 2),
                    "score": round(a["score"], 4),
                    "reason": a["reason"],
                    "confidence": round(a["confidence"], 4),
                    "low_confidence": a["low_confidence"]
                }
                for a in anomalies_list
            ],
            "recurring_commitments": {
                "recurring_expenses": [
                    {
                        "merchant": r["merchant"],
                        "amount": round(r["amount"], 2),
                        "category": r["category"],
                        "frequency": r["frequency"]
                    }
                    for r in recurring_expenses_list
                ],
                "recurring_investments": [
                    {
                        "merchant": r["merchant"],
                        "amount": round(r["amount"], 2),
                        "category": r["category"],
                        "frequency": r["frequency"]
                    }
                    for r in recurring_investments_list
                ],
                "total_recurring_monthly": round(rec_exp_total + rec_inv_total, 2)
            },
            "forecast": forecasting_result if forecasting_result else {
                "status": "insufficient_data",
                "reason": "Isolated test transactions lack longitudinal temporal context.",
                "current_spending": None,
                "predicted_remaining": None,
                "predicted_month_end": None
            },
            "behavior_signals": {
                "transaction_count": n_total_txns,
                "average_transaction": round(avg_txn_val, 2),
                "weekend_spending_ratio": round(weekend_spending_ratio, 3),
                "recurring_expense_ratio": round(rec_expense_ratio, 3),
                "subscription_ratio": round(subscription_ratio, 3),
                "savings_investment_ratio": round(savings_investment_ratio, 3)
            },
            "user_segmentation": user_segmentation_result if user_segmentation_result else {
                "status": "unavailable_for_isolated_input"
            },
            "tracking_difficulty_signals": {
                "payment_methods_count": len(payment_methods),
                "payment_methods_active": sorted(list(payment_methods)),
                "transaction_count": n_total_txns,
                "recurring_streams_count": len(recurring_expenses_list) + len(recurring_investments_list),
                "anomalies_count": len(anomalies_list),
                "low_confidence_count": len(low_confidence_txns)
            },
            "classification_warnings": low_confidence_txns,
            "observations": observations,
            "data_quality_warnings": warnings
        }

        # Save reports automatically to data/test_results/
        if save_report:
            res_dir = os.path.join(BASE_DIR, "data", "test_results")
            os.makedirs(res_dir, exist_ok=True)
            txt_path = os.path.join(res_dir, f"financial_report_{timestamp_str}.txt")
            json_path = os.path.join(res_dir, f"financial_report_{timestamp_str}.json")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(report_text)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_report, f, indent=2)

            print("\n" + "=" * 70)
            print("[REPORT SAVED]")
            print(f"TXT : {txt_path}")
            print(f"JSON: {json_path}")
            print("=" * 70 + "\n")

        return report_text, json_report


class ModelTestConsole:
    def __init__(self, models_dir: str, debug: bool = False):
        self.models_dir = models_dir
        self.debug = debug
        self.classifier = None
        self.anomaly_detector = None
        self.forecaster = None
        self.segmenter = None
        self.recurring_detector = RecurringExpenseDetector(min_occurrences=2)
        self.orchestrator = None
        self.load_models()

    def load_models(self):
        print("\n" + "=" * 60)
        print("MODEL LOADING")
        print("=" * 60)

        # 1. Transaction Classifier
        clf_path = os.path.join(self.models_dir, "transaction_classifier")
        try:
            self.classifier = TransactionClassifierPipeline.load(clf_path)
            model_name = self.classifier.metadata.get("model_name", "TransactionClassifier")
            print(f"[OK] Transaction classifier loaded ({model_name})")
            if self.debug:
                print(f"     Artifact: {clf_path}")
                print(f"     Classes: {len(self.classifier.label_encoder.classes_)} categories")
        except Exception as e:
            print(f"[ERROR] Could not load transaction classifier from {clf_path}: {e}")

        # 2. Anomaly Detector
        anom_path = os.path.join(self.models_dir, "anomaly_detector")
        try:
            self.anomaly_detector = ContextualAnomalyDetector.load(anom_path)
            model_name = self.anomaly_detector.metadata.get("model_name", "ContextualAnomalyDetector")
            print(f"[OK] Anomaly detector loaded ({model_name})")
            if self.debug:
                print(f"     Artifact: {anom_path}")
                print(f"     User Baselines: {len(self.anomaly_detector.user_overall_baselines)} users")
                print(f"     Features: {self.anomaly_detector.feature_names}")
        except Exception as e:
            print(f"[ERROR] Could not load anomaly detector from {anom_path}: {e}")

        # 3. Spending Forecaster
        forecast_path = os.path.join(self.models_dir, "forecasting")
        try:
            self.forecaster = SpendingForecaster.load(forecast_path)
            model_name = self.forecaster.metadata.get("model_name", "SpendingForecaster")
            print(f"[OK] Forecasting model loaded ({model_name})")
            if self.debug:
                print(f"     Artifact: {forecast_path}")
                print(f"     Features: {self.forecaster.feature_cols}")
        except Exception as e:
            print(f"[ERROR] Could not load forecasting model from {forecast_path}: {e}")

        # 4. User Segmenter
        seg_path = os.path.join(self.models_dir, "segmentation")
        try:
            self.segmenter = UserSegmenter.load(seg_path)
            model_name = self.segmenter.metadata.get("model_name", "UserSegmenter")
            print(f"[OK] Segmentation model loaded ({model_name})")
            if self.debug:
                print(f"     Artifact: {seg_path}")
                print(f"     Optimal k: {self.segmenter.n_clusters}")
                print(f"     Features: {self.segmenter.feature_names}")
        except Exception as e:
            print(f"[ERROR] Could not load segmentation model from {seg_path}: {e}")

        # 5. Financial Intelligence Orchestrator
        try:
            self.orchestrator = FinancialIntelligenceOrchestrator(self.models_dir)
            print("[OK] Financial Intelligence Orchestrator ready")
        except Exception as e:
            print(f"[ERROR] Could not initialize orchestrator: {e}")

        print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Option 1: Transaction Classification
    # -------------------------------------------------------------------------
    def test_classification(
        self,
        user_id: str = "USR_0001",
        merchant: str = "SWIGGY",
        description: str = "SWIGGY FOOD ORDER",
        amount: float = 850.0,
        payment_method: str = "UPI",
        transaction_type: str = "debit",
        date_str: str = "2026-09-19"
    ):
        if not self.classifier:
            print("[ERROR] Transaction classifier is not loaded.")
            return

        res = self.classifier.predict_transaction(
            merchant_raw=merchant,
            description=description,
            amount=amount,
            transaction_type=transaction_type,
            payment_method=payment_method,
            date=date_str
        )

        cat = res["category"]
        subcat = res["subcategory"]
        conf = res["category_confidence"]
        conf_pct = conf * 100

        model_name = self.classifier.metadata.get("model_name", "LogisticRegression")
        artifact_path = os.path.join(self.models_dir, "transaction_classifier")

        print("\n" + "=" * 60)
        print("TRANSACTION CLASSIFICATION")
        print("=" * 60)
        print("\nInput")
        print("-" * 60)
        print(f"User ID          : {user_id}")
        print(f"Merchant         : {merchant}")
        print(f"Description      : {description}")
        print(f"Amount           : ₹{amount:,.2f}")
        print(f"Payment Method   : {payment_method}")
        print(f"Transaction Type : {transaction_type}")
        print(f"Date             : {date_str}")

        print("\nPrediction")
        print("-" * 60)
        print(f"Category         : {cat}")
        print(f"Subcategory      : {subcat}")
        print(f"Confidence       : {conf_pct:.2f}%")
        print(f"Model            : {model_name}")
        print(f"Artifact         : {artifact_path}")

        if conf < CLASSIFICATION_CONFIDENCE_THRESHOLD:
            print("\n" + "!" * 60)
            print("WARNING: LOW CLASSIFICATION CONFIDENCE")
            print("-" * 60)
            print(f"Confidence: {conf_pct:.2f}%")
            print("WARNING:")
            print("The classifier is uncertain about this transaction.")
            print("Do not treat the predicted category as reliable without review.")
            print("!" * 60)

        print("\nStructured Output")
        print("-" * 60)
        print(json.dumps(res, indent=2))
        print("=" * 60)
        print("Inference completed successfully.\n")
        return res

    # -------------------------------------------------------------------------
    # Option 2: Anomaly Detection
    # -------------------------------------------------------------------------
    def test_anomaly(
        self,
        user_id: str = "USR_0001",
        merchant: str = "Michelin Star Restaurant",
        subcategory: str = "Food",
        amount: float = 8500.0,
        payment_method: str = "Credit Card",
        date_str: str = "2026-09-19",
        hour: int = 21
    ):
        if not self.anomaly_detector:
            print("[ERROR] Anomaly detector is not loaded.")
            return

        is_weekend = 1 if datetime.strptime(date_str.split()[0], "%Y-%m-%d").weekday() >= 5 else 0

        res = self.anomaly_detector.predict_transaction(
            user_id=user_id,
            merchant=merchant,
            subcategory=subcategory,
            amount=amount,
            is_weekend=is_weekend,
            hour=hour
        )

        model_name = self.anomaly_detector.metadata.get("model_name", "IsolationForest")
        artifact_path = os.path.join(self.models_dir, "anomaly_detector")

        print("\n" + "=" * 60)
        print("ANOMALY DETECTION")
        print("=" * 60)
        print("\nTransaction")
        print("-" * 60)
        print(f"User             : {user_id}")
        print(f"Merchant         : {merchant}")
        print(f"Category         : {subcategory}")
        print(f"Amount           : ₹{amount:,.2f}")
        print(f"Payment Method   : {payment_method}")
        print(f"Date             : {date_str} (Weekend={bool(is_weekend)}, Hour={hour})")

        print("\nPrediction")
        print("-" * 60)
        print(f"Is Anomaly       : {'TRUE' if res['is_anomaly'] else 'FALSE'}")
        print(f"Anomaly Score    : {res['anomaly_score']:.4f}")
        print(f"Model            : {model_name}")
        print(f"Artifact         : {artifact_path}")
        print(f"\nReason:")
        print(f"{res.get('reason', 'Reason: Not provided by model.')}")

        print("\nStructured Output")
        print("-" * 60)
        print(json.dumps(res, indent=2))
        print("=" * 60)
        print("Inference completed successfully.\n")
        return res

    # -------------------------------------------------------------------------
    # Option 3: Complete Financial Intelligence
    # -------------------------------------------------------------------------
    def test_complete_intelligence(
        self,
        user_id: str = "USR_0001",
        merchant: str = "SWIGGY",
        description: str = "SWIGGY FOOD ORDER",
        amount: float = 850.0,
        payment_method: str = "UPI",
        transaction_type: str = "debit",
        date_str: str = "2026-09-19",
        hour: int = 20
    ):
        if not self.orchestrator:
            print("[ERROR] Orchestrator is not loaded.")
            return

        is_weekend = 1 if datetime.strptime(date_str.split()[0], "%Y-%m-%d").weekday() >= 5 else 0

        res = self.orchestrator.process_incoming_transaction(
            user_id=user_id,
            merchant_raw=merchant,
            description=description,
            amount=amount,
            transaction_type=transaction_type,
            payment_method=payment_method,
            is_weekend=is_weekend,
            hour=hour
        )

        cat_info = res["categorization"]
        anom_info = res["anomaly_detection"]

        rec_keywords = ["netflix", "spotify", "broadband", "rent", "lease", "gym", "internet", "airtel", "fibernet", "cult", "sip", "nobroker"]
        is_rec = any(rk in merchant.lower() for rk in rec_keywords) or cat_info.get("subcategory") == "Subscriptions"

        res["recurring_expense"] = {
            "is_recurring": is_rec,
            "recurrence_frequency": "Monthly" if is_rec else None,
            "status": "Identified as subscription/bill pattern" if is_rec else "One-off discretionary spend"
        }

        print("\n" + "=" * 60)
        print("COMPLETE FINANCIAL INTELLIGENCE")
        print("=" * 60)
        print("\nINPUT")
        print("-" * 60)
        print(f"User ID       : {user_id}")
        print(f"Merchant      : {merchant}")
        print(f"Description   : {description}")
        print(f"Amount        : ₹{amount:,.2f}")
        print(f"Payment       : {payment_method}")
        print(f"Date          : {date_str}")

        print("\n" + "-" * 60)
        print("CLASSIFICATION")
        print("-" * 60)
        print(f"Category      : {cat_info['category']}")
        print(f"Subcategory   : {cat_info['subcategory']}")
        print(f"Confidence    : {cat_info['category_confidence']*100:.2f}%")

        if cat_info["category_confidence"] < CLASSIFICATION_CONFIDENCE_THRESHOLD:
            print("WARNING       : Low confidence classification (< 60.0%)")

        print("\n" + "-" * 60)
        print("ANOMALY")
        print("-" * 60)
        print(f"Is Anomaly    : {'TRUE' if anom_info['is_anomaly'] else 'FALSE'}")
        print(f"Score         : {anom_info['anomaly_score']:.4f}")
        print(f"Reason        : {anom_info['reason']}")

        print("\n" + "-" * 60)
        print("RECURRING")
        print("-" * 60)
        print(f"Is Recurring  : {'TRUE' if is_rec else 'FALSE'}")
        if is_rec:
            print(f"Cadence       : Monthly (Projected)")

        print("\n" + "=" * 60)
        print("FINAL STRUCTURED OUTPUT (JSON)")
        print("-" * 60)
        print(json.dumps(res, indent=2))
        print("=" * 60)
        print("Inference completed successfully.\n")

        # Package standardized item for report generator
        return {
            "user_id": user_id,
            "merchant": merchant,
            "description": description,
            "amount": amount,
            "payment_method": payment_method,
            "transaction_type": transaction_type,
            "date": date_str,
            "predicted_category": cat_info["category"],
            "predicted_subcategory": cat_info["subcategory"],
            "confidence": cat_info["category_confidence"],
            "is_anomaly": anom_info["is_anomaly"],
            "anomaly_score": anom_info["anomaly_score"],
            "anomaly_reason": anom_info["reason"],
            "is_recurring": is_rec
        }

    # -------------------------------------------------------------------------
    # Option 4: Spending Forecast Test
    # -------------------------------------------------------------------------
    def test_forecasting(
        self,
        current_mtd_spending: float = 31500.0,
        days_elapsed: int = 15,
        days_remaining: int = 15,
        rolling_7_day: float = 7800.0,
        rolling_30_day: float = 32000.0,
        previous_month_spending: float = 62000.0,
        monthly_recurring_committed: float = 35000.0
    ):
        if not self.forecaster:
            print("[ERROR] Forecasting model is not loaded.")
            return

        res = self.forecaster.predict_spending(
            current_mtd_spending=current_mtd_spending,
            days_elapsed=days_elapsed,
            days_remaining=days_remaining,
            rolling_7_day=rolling_7_day,
            rolling_30_day=rolling_30_day,
            previous_month_spending=previous_month_spending,
            monthly_recurring_committed=monthly_recurring_committed
        )

        model_name = self.forecaster.metadata.get("model_name", "SpendingForecaster")
        artifact_path = os.path.join(self.models_dir, "forecasting")

        print("\n" + "=" * 60)
        print("SPENDING FORECAST")
        print("=" * 60)
        print("\nInput Parameters (Exact Model Features)")
        print("-" * 60)
        print(f"Month-to-Date Spend (MTD)     : ₹{current_mtd_spending:,.2f}")
        print(f"Days Elapsed / Remaining       : {days_elapsed} days / {days_remaining} days")
        print(f"Rolling 7-Day Spending         : ₹{rolling_7_day:,.2f}")
        print(f"Rolling 30-Day Spending        : ₹{rolling_30_day:,.2f}")
        print(f"Previous Month Spending        : ₹{previous_month_spending:,.2f}")
        print(f"Committed Recurring Total      : ₹{monthly_recurring_committed:,.2f}")

        print("\nForecast Projections")
        print("-" * 60)
        print(f"Current Spending (MTD)         : ₹{res['current_spending']:,.2f}")
        print(f"Predicted Remaining Month Spend: ₹{res['predicted_remaining']:,.2f}")
        print(f"Predicted Month-End Total      : ₹{res['predicted_month_end']:,.2f}")
        print(f"Model-Reported Error Margin    : ±₹{res.get('forecast_error_margin', 0):,.2f}")
        print(f"Forecast Confidence            : Not statistically calibrated (heuristic model output)")
        print(f"Model                          : {model_name}")
        print(f"Artifact                       : {artifact_path}")

        print("\nStructured Output")
        print("-" * 60)
        print(json.dumps(res, indent=2))
        print("=" * 60)
        print("Inference completed successfully.\n")
        return res

    # -------------------------------------------------------------------------
    # Option 5: User Segmentation Test
    # -------------------------------------------------------------------------
    def test_segmentation(self, user_features: dict = None):
        if not self.segmenter:
            print("[ERROR] Segmentation model is not loaded.")
            return

        if user_features is None:
            user_features = {
                "monthly_income": 95000.0,
                "monthly_expense": 62000.0,
                "savings_ratio": 0.347,
                "transaction_count": 68.0,
                "average_transaction": 911.76,
                "food_ratio": 0.18,
                "shopping_ratio": 0.12,
                "transport_ratio": 0.08,
                "subscription_ratio": 0.03,
                "recurring_expense_ratio": 0.52,
                "spending_variability": 0.65,
                "weekend_spending_ratio": 0.38,
                "merchant_count": 22,
                "category_count": 9,
                "category_entropy": 2.45
            }

        res = self.segmenter.predict_user_segment(user_features)
        model_name = self.segmenter.metadata.get("model_name", "KMeans")
        artifact_path = os.path.join(self.models_dir, "segmentation")

        print("\n" + "=" * 60)
        print("USER SEGMENTATION")
        print("=" * 60)
        print("\nUser Feature Vector (15 Dimensions)")
        print("-" * 60)
        for k, v in user_features.items():
            if "income" in k or "expense" in k or "transaction" in k and "ratio" not in k and "entropy" not in k and "count" not in k:
                print(f"{k:<25}: ₹{v:,.2f}")
            else:
                print(f"{k:<25}: {v}")

        print("\nSegmentation Result")
        print("-" * 60)
        print(f"Cluster ID         : {res['cluster_id']}")
        print(f"Segment Persona    : {res['persona_name']}")
        print(f"Behavioral Profile : {res['persona_description']}")
        print(f"Model              : {model_name}")
        print(f"Artifact           : {artifact_path}")

        print("\nStructured Output")
        print("-" * 60)
        print(json.dumps(res, indent=2))
        print("=" * 60)
        print("Inference completed successfully.\n")
        return res

    # -------------------------------------------------------------------------
    # Option 6: CSV Batch Testing & Aggregated Report
    # -------------------------------------------------------------------------
    def test_csv_batch(self, csv_path: str):
        if not os.path.exists(csv_path):
            print(f"[ERROR] File not found: {csv_path}")
            print("\nExpected CSV Schema:")
            print("user_id,date,merchant_raw,description,amount,payment_method,transaction_type")
            return

        df = pd.read_csv(csv_path)
        print(f"\n[OK] Loaded CSV with {len(df)} rows from: {csv_path}")

        results = []
        rec_keywords = ["netflix", "spotify", "broadband", "rent", "lease", "gym", "internet", "airtel", "fibernet", "cult", "sip", "nobroker"]

        print("\n" + "=" * 85)
        print(f"{'Transaction':<28} | {'Category':<16} | {'Confidence':<10} | {'Anomaly':<7} | {'Score':<6} | {'Recurring':<8}")
        print("-" * 85)

        for idx, row in df.iterrows():
            uid = str(row.get("user_id", "USR_0001"))
            merch = str(row.get("merchant_raw", row.get("merchant", "UNKNOWN")))
            desc = str(row.get("description", ""))
            amt = float(row.get("amount", 0.0))
            pm = str(row.get("payment_method", "UPI"))
            tt = str(row.get("transaction_type", "debit"))
            dt = str(row.get("date", "2026-09-19 12:00:00"))

            # Classification
            cat_res = self.classifier.predict_transaction(
                merchant_raw=merch,
                description=desc,
                amount=amt,
                transaction_type=tt,
                payment_method=pm,
                date=dt
            )
            conf = cat_res['category_confidence']

            # Anomaly
            try:
                dt_obj = datetime.strptime(dt.split()[0], "%Y-%m-%d")
                is_wk = 1 if dt_obj.weekday() >= 5 else 0
            except Exception:
                is_wk = 0

            anom_res = self.anomaly_detector.predict_transaction(
                user_id=uid,
                merchant=merch,
                subcategory=cat_res['subcategory'],
                amount=amt,
                is_weekend=is_wk,
                hour=14
            )
            is_anom = anom_res['is_anomaly']
            anom_score = anom_res['anomaly_score']

            is_rec = any(rk in merch.lower() or rk in desc.lower() for rk in rec_keywords) or cat_res["subcategory"] == "Subscriptions"

            txn_summary = f"{merch[:18]} (₹{amt:,.0f})"
            cat_short = cat_res['subcategory'][:15]
            conf_str = f"{conf*100:.1f}%"
            anom_str = "YES" if is_anom else "NO"
            score_str = f"{anom_score:.2f}"
            rec_str = "YES" if is_rec else "NO"

            print(f"{txn_summary:<28} | {cat_short:<16} | {conf_str:<10} | {anom_str:<7} | {score_str:<6} | {rec_str:<8}")

            results.append({
                "row_index": idx,
                "user_id": uid,
                "date": dt,
                "merchant": merch,
                "description": desc,
                "amount": amt,
                "payment_method": pm,
                "transaction_type": tt,
                "predicted_category": cat_res["category"],
                "predicted_subcategory": cat_res["subcategory"],
                "confidence": conf,
                "is_anomaly": is_anom,
                "anomaly_score": anom_score,
                "anomaly_reason": anom_res["reason"],
                "is_recurring": is_rec
            })

        print("=" * 85)

        # Generate & Print Aggregated Financial Intelligence Report
        target_uid = str(df["user_id"].iloc[0]) if "user_id" in df.columns and len(df) > 0 else "USR_0001"
        report_text, json_report = FinancialIntelligenceReportGenerator.generate_report(
            transactions_results=results,
            user_id=target_uid,
            save_report=True
        )

        print("\n" + report_text)
        return results

    # -------------------------------------------------------------------------
    # Preset Demo Mode (--demo) with Aggregated Financial Intelligence Report
    # -------------------------------------------------------------------------
    def run_demo(self):
        print("\n" + "=" * 70)
        print("PRESET DEMO TEST CASES (8 REALISTIC FINANCIAL SCENARIOS)")
        print("=" * 70)

        demo_cases = [
            ("1. Normal Grocery Purchase", "USR_0001", "ZEPTO QUICK COMMERCE", "Daily fresh milk curd bread paneer", 450.0, "UPI", "debit", "2026-09-19 09:30:00"),
            ("2. Normal Food Delivery", "USR_0001", "SWIGGY*BANGALORE FOOD", "Lunch delivery biryani coke", 650.0, "UPI", "debit", "2026-09-19 13:15:00"),
            ("3. Recurring Subscription", "USR_0001", "NETFLIX INDIA MONTHLY PLAN", "Monthly 4K Ultra HD streaming subscription", 649.0, "UPI AutoPay", "debit", "2026-09-19 04:00:00"),
            ("4. Corporate Salary Credit", "USR_0001", "TATA CONSULTANCY SERVICES SALARY", "Monthly net salary credit direct deposit", 85000.0, "Direct Deposit", "credit", "2026-09-19 09:00:00"),
            ("5. Mutual Fund Investment SIP", "USR_0001", "GROWW MUTUAL FUND SIP AUTO", "Monthly Nifty 50 index fund SIP investment", 15000.0, "Auto Debit", "transfer", "2026-09-19 14:00:00"),
            ("6. Unusually Large Food Outlier", "USR_0001", "TAJ PALACE 5-STAR LUXURY BANQUET", "Private banquet party dinner catering", 16500.0, "Credit Card", "debit", "2026-09-19 23:00:00"),
            ("7. Unknown / Ambiguous Merchant", "USR_0001", "UNKNOWN MERCHANT XYZ CORP 9812", "General miscellaneous payment ref 49102", 7500.0, "Credit Card", "debit", "2026-09-19 17:30:00"),
            ("8. High-Value Healthcare Transaction", "USR_0001", "APOLLO HOSPITALS EMERGENCY CARE", "Inpatient medical diagnostic emergency treatment", 55000.0, "Credit Card", "debit", "2026-09-19 16:00:00")
        ]

        collected_results = []
        for title, uid, merch, desc, amt, pm, tt, dt in demo_cases:
            print(f"\n[{title}]")
            print(f"Input: {merch} | ₹{amt:,.2f} | {pm} | {tt}")
            item_res = self.test_complete_intelligence(
                user_id=uid,
                merchant=merch,
                description=desc,
                amount=amt,
                payment_method=pm,
                transaction_type=tt,
                date_str=dt
            )
            collected_results.append(item_res)

        # ---------------------------------------------------------------------
        # Final Aggregated Financial Intelligence Report
        # ---------------------------------------------------------------------
        report_text, json_report = FinancialIntelligenceReportGenerator.generate_report(
            transactions_results=collected_results,
            user_id="USR_0001",
            save_report=True
        )

        print("\n" + report_text)


# -------------------------------------------------------------------------
# Interactive Menu Loop
# -------------------------------------------------------------------------
def interactive_menu(console: ModelTestConsole):
    while True:
        print("\n" + "=" * 60)
        print("FINANCIAL AI MODEL TEST CONSOLE")
        print("=" * 60)
        print("1. Test Transaction Classification")
        print("2. Test Anomaly Detection")
        print("3. Test Complete Transaction Intelligence")
        print("4. Test Spending Forecast")
        print("5. Test User Segmentation")
        print("6. Test Multiple Transactions from CSV (With Report)")
        print("7. Exit")
        print("=" * 60)

        choice = input("Select option (1-7): ").strip()

        if choice == "1":
            print("\nEnter Transaction Details:")
            uid = input("User ID [USR_0001]: ").strip() or "USR_0001"
            merch = input("Merchant [SWIGGY]: ").strip() or "SWIGGY"
            desc = input("Description [SWIGGY FOOD ORDER]: ").strip() or "SWIGGY FOOD ORDER"
            amt_str = input("Amount (₹) [850]: ").strip() or "850"
            try:
                amt = float(amt_str)
            except ValueError:
                print("[ERROR] Invalid amount.")
                continue
            pm = input("Payment Method [UPI]: ").strip() or "UPI"
            tt = input("Transaction Type (debit/credit/transfer) [debit]: ").strip() or "debit"
            dt = input("Date (YYYY-MM-DD) [2026-09-19]: ").strip() or "2026-09-19"

            console.test_classification(
                user_id=uid,
                merchant=merch,
                description=desc,
                amount=amt,
                payment_method=pm,
                transaction_type=tt,
                date_str=dt
            )

        elif choice == "2":
            print("\nEnter Anomaly Test Details:")
            uid = input("User ID [USR_0001]: ").strip() or "USR_0001"
            merch = input("Merchant [Michelin Star Restaurant]: ").strip() or "Michelin Star Restaurant"
            subcat = input("Category/Subcategory [Food]: ").strip() or "Food"
            amt_str = input("Amount (₹) [8500]: ").strip() or "8500"
            try:
                amt = float(amt_str)
            except ValueError:
                print("[ERROR] Invalid amount.")
                continue
            pm = input("Payment Method [Credit Card]: ").strip() or "Credit Card"
            dt = input("Date (YYYY-MM-DD) [2026-09-19]: ").strip() or "2026-09-19"

            console.test_anomaly(
                user_id=uid,
                merchant=merch,
                subcategory=subcat,
                amount=amt,
                payment_method=pm,
                date_str=dt
            )

        elif choice == "3":
            print("\nEnter Transaction Intelligence Details:")
            uid = input("User ID [USR_0001]: ").strip() or "USR_0001"
            merch = input("Merchant [SWIGGY]: ").strip() or "SWIGGY"
            desc = input("Description [SWIGGY FOOD ORDER]: ").strip() or "SWIGGY FOOD ORDER"
            amt_str = input("Amount (₹) [850]: ").strip() or "850"
            try:
                amt = float(amt_str)
            except ValueError:
                print("[ERROR] Invalid amount.")
                continue
            pm = input("Payment Method [UPI]: ").strip() or "UPI"
            tt = input("Transaction Type [debit]: ").strip() or "debit"
            dt = input("Date (YYYY-MM-DD) [2026-09-19]: ").strip() or "2026-09-19"

            console.test_complete_intelligence(
                user_id=uid,
                merchant=merch,
                description=desc,
                amount=amt,
                payment_method=pm,
                transaction_type=tt,
                date_str=dt
            )

        elif choice == "4":
            print("\nEnter Spending Forecast Parameters:")
            mtd_str = input("Current Month-to-Date Spend (₹) [31500]: ").strip() or "31500"
            del_str = input("Days Elapsed in Month [15]: ").strip() or "15"
            drem_str = input("Days Remaining in Month [15]: ").strip() or "15"
            r7_str = input("Rolling 7-Day Spending (₹) [7800]: ").strip() or "7800"
            r30_str = input("Rolling 30-Day Spending (₹) [32000]: ").strip() or "32000"
            prev_str = input("Previous Month Spending (₹) [62000]: ").strip() or "62000"
            rec_str = input("Committed Recurring Total (₹) [35000]: ").strip() or "35000"

            try:
                console.test_forecasting(
                    current_mtd_spending=float(mtd_str),
                    days_elapsed=int(del_str),
                    days_remaining=int(drem_str),
                    rolling_7_day=float(r7_str),
                    rolling_30_day=float(r30_str),
                    previous_month_spending=float(prev_str),
                    monthly_recurring_committed=float(rec_str)
                )
            except ValueError as e:
                print(f"[ERROR] Invalid numeric input: {e}")

        elif choice == "5":
            print("\nUser Segmentation Feature Vector (Press Enter to use defaults):")
            inc_str = input("Monthly Income (₹) [95000]: ").strip() or "95000"
            exp_str = input("Monthly Expense (₹) [62000]: ").strip() or "62000"
            sav_str = input("Savings Ratio (0.0-1.0) [0.347]: ").strip() or "0.347"
            txn_str = input("Monthly Transaction Count [68]: ").strip() or "68"
            avg_str = input("Average Transaction (₹) [911.76]: ").strip() or "911.76"
            food_str = input("Food Spending Ratio [0.18]: ").strip() or "0.18"
            shop_str = input("Shopping Ratio [0.12]: ").strip() or "0.12"
            trn_str = input("Transport Ratio [0.08]: ").strip() or "0.08"
            sub_str = input("Subscription Ratio [0.03]: ").strip() or "0.03"
            rec_str = input("Recurring Expense Ratio [0.52]: ").strip() or "0.52"
            var_str = input("Spending Variability [0.65]: ").strip() or "0.65"
            wk_str = input("Weekend Spending Ratio [0.38]: ").strip() or "0.38"
            mcnt_str = input("Merchant Count [22]: ").strip() or "22"
            ccnt_str = input("Category Count [9]: ").strip() or "9"
            ent_str = input("Category Entropy [2.45]: ").strip() or "2.45"

            try:
                feats = {
                    "monthly_income": float(inc_str),
                    "monthly_expense": float(exp_str),
                    "savings_ratio": float(sav_str),
                    "transaction_count": float(txn_str),
                    "average_transaction": float(avg_str),
                    "food_ratio": float(food_str),
                    "shopping_ratio": float(shop_str),
                    "transport_ratio": float(trn_str),
                    "subscription_ratio": float(sub_str),
                    "recurring_expense_ratio": float(rec_str),
                    "spending_variability": float(var_str),
                    "weekend_spending_ratio": float(wk_str),
                    "merchant_count": int(mcnt_str),
                    "category_count": int(ccnt_str),
                    "category_entropy": float(ent_str)
                }
                console.test_segmentation(feats)
            except ValueError as e:
                print(f"[ERROR] Invalid numeric input: {e}")

        elif choice == "6":
            csv_path = input("\nEnter path to test CSV file [data/test_transactions_sample.csv]: ").strip()
            if not csv_path:
                csv_path = os.path.join(BASE_DIR, "data", "test_transactions_sample.csv")
            console.test_csv_batch(csv_path)

        elif choice == "7":
            print("\nExiting Financial AI Model Test Console. Goodbye!\n")
            break
        else:
            print("\n[!] Invalid choice. Please select 1 through 7.")


def main():
    parser = argparse.ArgumentParser(description="Financial AI Model Test Console")
    parser.add_argument("--demo", action="store_true", help="Run preset test cases for 8 realistic financial scenarios with report")
    parser.add_argument("--debug", action="store_true", help="Enable debug inspection mode (shapes, probabilities, paths)")
    parser.add_argument("--csv", type=str, help="Path to CSV file for batch transaction testing and report generation")
    args = parser.parse_args()

    models_path = os.path.join(BASE_DIR, "models")
    console = ModelTestConsole(models_path, debug=args.debug)

    if args.demo:
        console.run_demo()
    elif args.csv:
        console.test_csv_batch(args.csv)
    else:
        interactive_menu(console)


if __name__ == "__main__":
    main()
