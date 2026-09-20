"""
Financial Intelligence & Data Science Orchestrator.
Provides a unified, high-level Python API and inference engine.
Combines:
1. Transaction Categorization
2. Contextual Anomaly Detection
3. Recurring Expense & Subscription Intelligence
4. Monthly Spending Forecasting
5. User Behavioral Segmentation & Entropy Profiling
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transaction_classifier import TransactionClassifierPipeline
from src.models.anomaly_detector import ContextualAnomalyDetector
from src.models.recurring_detector import RecurringExpenseDetector
from src.models.spending_forecaster import SpendingForecaster
from src.models.user_segmenter import UserSegmenter

class FinancialIntelligenceOrchestrator:
    def __init__(self, models_root_dir: str):
        self.models_root_dir = models_root_dir
        
        # Paths to serialized model artifacts
        self.clf_dir = os.path.join(models_root_dir, "transaction_classifier")
        self.anom_dir = os.path.join(models_root_dir, "anomaly_detector")
        self.forecast_dir = os.path.join(models_root_dir, "forecasting")
        self.seg_dir = os.path.join(models_root_dir, "segmentation")

        # Load models
        print(f"[Orchestrator] Loading DS/ML pipelines from {models_root_dir}...")
        self.classifier = TransactionClassifierPipeline.load(self.clf_dir)
        self.anomaly_detector = ContextualAnomalyDetector.load(self.anom_dir)
        self.forecaster = SpendingForecaster.load(self.forecast_dir)
        self.segmenter = UserSegmenter.load(self.seg_dir)
        self.recurring_detector = RecurringExpenseDetector(min_occurrences=2)
        print("[Orchestrator] All DS/ML intelligence modules successfully loaded and ready for inference.")

    def process_incoming_transaction(
        self,
        user_id: str,
        merchant_raw: str,
        description: str,
        amount: float,
        transaction_type: str = "debit",
        payment_method: str = "Credit Card",
        is_weekend: int = 0,
        hour: int = 14
    ) -> Dict[str, Any]:
        """
        Executes end-to-end processing of a real-time incoming transaction:
        1. Categorizes & Subcategorizes transaction via supervised ML
        2. Evaluates contextual anomaly score relative to user's history
        3. Returns unified, structured JSON payload
        """
        # 1. Supervised Categorization
        cat_result = self.classifier.predict_transaction(
            merchant_raw=merchant_raw,
            description=description,
            amount=amount,
            transaction_type=transaction_type,
            payment_method=payment_method
        )

        # 2. Contextual Anomaly Detection
        anom_result = self.anomaly_detector.predict_transaction(
            user_id=user_id,
            merchant=merchant_raw,
            subcategory=cat_result["subcategory"],
            amount=amount,
            is_weekend=is_weekend,
            hour=hour
        )

        return {
            "transaction_input": {
                "user_id": user_id,
                "merchant_raw": merchant_raw,
                "amount": float(amount),
                "payment_method": payment_method
            },
            "categorization": cat_result,
            "anomaly_detection": anom_result
        }

    def forecast_user_spending(
        self,
        current_mtd_spending: float,
        days_elapsed: int,
        days_remaining: int,
        rolling_7_day: float,
        rolling_30_day: float,
        previous_month_spending: float,
        monthly_recurring_committed: float
    ) -> Dict[str, Any]:
        """
        Generates spending forecast and month-end projection.
        """
        return self.forecaster.predict_spending(
            current_mtd_spending=current_mtd_spending,
            days_elapsed=days_elapsed,
            days_remaining=days_remaining,
            rolling_7_day=rolling_7_day,
            rolling_30_day=rolling_30_day,
            previous_month_spending=previous_month_spending,
            monthly_recurring_committed=monthly_recurring_committed
        )

    def register_user_history(self, user_id: str, df_user: pd.DataFrame):
        """
        Dynamically registers user transaction history into the anomaly detector's personal baseline store.
        """
        self.anomaly_detector.register_user_history(user_id, df_user)

    def analyze_user_recurring_subscriptions(
        self,
        user_transactions_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Extracts all recurring subscription streams, next payment dates, and cadences for a user.
        """
        return self.recurring_detector.fit_and_detect_user(user_transactions_df)

    def get_user_behavioral_profile(
        self,
        user_features_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computes behavioral segment assignment and behavioral metrics.
        """
        return self.segmenter.predict_user_segment(user_features_dict)

    def build_user_llm_context(
        self,
        user_id: str,
        user_transactions_df: pd.DataFrame,
        user_query: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Constructs a strictly user-isolated Financial Intelligence context payload for LLM reasoning.
        Guarantees that NO transaction, baseline, anomaly, or conversation data from other users
        is included in this context.
        """
        uid = str(user_id)
        u_df = user_transactions_df.copy()

        # Strict validation: Ensure all transactions belong to user_id
        if not u_df.empty and "user_id" in u_df.columns:
            u_df = u_df[u_df["user_id"].astype(str) == uid].reset_index(drop=True)

        # 1. User Summary Metrics
        if not u_df.empty:
            expenses = float(u_df[u_df["category"] == "EXPENSE"]["amount"].sum()) if "category" in u_df.columns else float(u_df["amount"].sum())
            income = float(u_df[u_df["category"] == "INCOME"]["amount"].sum()) if "category" in u_df.columns else 0.0
            transfers = float(u_df[u_df["category"] == "TRANSFER"]["amount"].sum()) if "category" in u_df.columns else 0.0
            n_txns = len(u_df)

            # Category breakdown
            subcat_col = "subcategory" if "subcategory" in u_df.columns else "category"
            cat_breakdown = u_df[u_df["category"] == "EXPENSE"].groupby(subcat_col)["amount"].sum().to_dict() if "category" in u_df.columns else {}
        else:
            expenses, income, transfers, n_txns = 0.0, 0.0, 0.0, 0
            cat_breakdown = {}

        # 2. Recurring Subscriptions
        recurring_streams = self.recurring_detector.fit_and_detect_user(u_df) if not u_df.empty else []

        # 3. Personal Baselines
        u_base = self.anomaly_detector.user_overall_baselines.get(uid, {"mean": 0.0, "total_txns": 0})

        # 4. Synthesize LLM Context Object
        llm_context = {
            "authenticated_user_id": uid,
            "financial_summary": {
                "total_transactions": n_txns,
                "total_income": round(income, 2),
                "total_expenses": round(expenses, 2),
                "total_transfers_investments": round(transfers, 2),
                "net_cash_flow": round(income - expenses - transfers, 2),
                "category_spending": {k: round(float(v), 2) for k, v in cat_breakdown.items()}
            },
            "user_baseline": {
                "overall_expense_mean": round(u_base.get("mean", 0.0), 2),
                "recorded_history_count": u_base.get("total_txns", 0)
            },
            "recurring_commitments": recurring_streams,
            "user_query": user_query,
            "conversation_history": conversation_history or []
        }

        return llm_context


if __name__ == "__main__":
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    orchestrator = FinancialIntelligenceOrchestrator(models_dir)

    print("\n" + "=" * 75)
    print("FINANCIAL INTELLIGENCE ORCHESTRATOR - REALTIME INFERENCE")
    print("=" * 75)

    # 1. Normal Grocery Transaction (Zepto / Blinkit)
    res1 = orchestrator.process_incoming_transaction(
        user_id="USR_0001",
        merchant_raw="BLINKIT COMMERCE GURGAON",
        description="Weekly grocery supplies milk vegetables eggs",
        amount=1850.0,
        transaction_type="debit",
        payment_method="UPI",
        is_weekend=1,
        hour=11
    )
    print("\n1. Standard Grocery Purchase Evaluation:")
    print(json.dumps(res1, indent=2))

    # 2. Contextual Anomaly (Sudden Outlier Food Bill)
    res2 = orchestrator.process_incoming_transaction(
        user_id="USR_0001",
        merchant_raw="TAJ PALACE LUXURY DINING",
        description="Dinner party banquet restaurant",
        amount=14800.0,
        transaction_type="debit",
        payment_method="Credit Card",
        is_weekend=1,
        hour=23
    )
    print("\n2. Contextual Outlier Expense Evaluation:")
    print(json.dumps(res2, indent=2))

    # 3. Monthly Spending Forecast
    res3 = orchestrator.forecast_user_spending(
        current_mtd_spending=28500.0,
        days_elapsed=14,
        days_remaining=16,
        rolling_7_day=7200.0,
        rolling_30_day=34000.0,
        previous_month_spending=58000.0,
        monthly_recurring_committed=28000.0
    )
    print("\n3. Mid-Month User Spending Forecast:")
    print(json.dumps(res3, indent=2))
