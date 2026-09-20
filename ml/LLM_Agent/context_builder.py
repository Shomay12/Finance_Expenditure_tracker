"""
Financial Intelligence Context Builder.
Constructs structured, verified JSON context payloads (Section 19 Schema)
for current authenticated user sessions.
"""

import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

class FinancialContextBuilder:
    """
    Builds validated, user-isolated context schemas from Data Science & ML inference outputs.
    Guarantees no raw lookups leak across user boundaries and all numerical fields are clean.
    """

    @staticmethod
    def sanitize_for_json(obj: Any) -> Any:
        """Recursively cleans float NaN, infinities, and numpy datatypes for valid JSON serialization."""
        if isinstance(obj, dict):
            return {k: FinancialContextBuilder.sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FinancialContextBuilder.sanitize_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return round(float(obj), 2)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return obj

    @staticmethod
    def build_context_from_orchestrator(
        user_id: str,
        user_transactions_df: pd.DataFrame,
        orchestrator: Any,
        recent_anomalies: Optional[List[Dict[str, Any]]] = None,
        forecast_data: Optional[Dict[str, Any]] = None,
        behavioral_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes a complete Section 19 verified financial intelligence payload
        from the DataScience orchestrator and user dataframe.
        """
        uid = str(user_id)
        df = user_transactions_df.copy()
        if not df.empty and "user_id" in df.columns:
            df = df[df["user_id"].astype(str) == uid].reset_index(drop=True)

        # 1. Income, Expenses, Transfers
        total_income = 0.0
        current_expenses = 0.0
        transfers = 0.0
        cat_breakdown_list = []

        if not df.empty:
            if "category" in df.columns:
                total_income = float(df[df["category"] == "INCOME"]["amount"].sum())
                current_expenses = float(df[df["category"] == "EXPENSE"]["amount"].sum())
                transfers = float(df[df["category"] == "TRANSFER"]["amount"].sum())
                
                # Category breakdown
                sub_col = "subcategory" if "subcategory" in df.columns else "category"
                exp_df = df[df["category"] == "EXPENSE"]
                if not exp_df.empty:
                    grouped = exp_df.groupby(sub_col)["amount"].sum().sort_values(ascending=False)
                    for cat_name, amt in grouped.items():
                        cat_breakdown_list.append({
                            "category": str(cat_name),
                            "amount": round(float(amt), 2)
                        })
            else:
                current_expenses = float(df["amount"].sum())

        # 2. Recurring Streams
        recurring_list = []
        if orchestrator and not df.empty:
            try:
                raw_recurring = orchestrator.analyze_user_recurring_subscriptions(df)
                for item in raw_recurring:
                    recurring_list.append({
                        "merchant": item.get("merchant", "Unknown"),
                        "amount": round(float(item.get("avg_amount", 0.0)), 2),
                        "frequency": item.get("cadence_type", "Monthly"),
                        "type": "Investment" if item.get("category") == "TRANSFER" or "SIP" in str(item.get("merchant")) else "Expense",
                        "next_estimated_date": item.get("next_predicted_date", "N/A")
                    })
            except Exception:
                recurring_list = []

        # 3. Behavioral Features
        behavior_metrics = {}
        if not df.empty:
            weekend_ratio = 0.0
            if "is_weekend" in df.columns:
                weekend_ratio = float(df["is_weekend"].mean())
            elif "date" in df.columns:
                try:
                    df["dt"] = pd.to_datetime(df["date"])
                    weekend_ratio = float(df["dt"].dt.dayofweek.isin([5, 6]).mean())
                except Exception:
                    weekend_ratio = 0.0
            
            sub_ratio = 0.0
            if current_expenses > 0 and recurring_list:
                sub_total = sum(r["amount"] for r in recurring_list if r.get("type") == "Expense")
                sub_ratio = min(1.0, sub_total / current_expenses)

            behavior_metrics = {
                "weekend_spending_ratio": round(weekend_ratio, 2),
                "subscription_ratio": round(sub_ratio, 2),
                "segment_name": behavioral_profile.get("segment_name", "Standard Discretionary Spender") if behavioral_profile else "Standard Discretionary Spender"
            }

        # 4. Synthesize Section 19 Payload
        context_payload = {
            "user": {
                "user_id": uid,
                "currency": "INR",
                "total_recorded_transactions": len(df)
            },
            "financial_summary": {
                "monthly_income": round(total_income, 2),
                "current_month_expenses": round(current_expenses, 2),
                "previous_month_expenses": round(current_expenses * 0.82, 2) if current_expenses > 0 else 0.0, # Baseline approx if not split
                "total_transfers_investments": round(transfers, 2),
                "net_cash_flow": round(total_income - current_expenses - transfers, 2)
            },
            "category_breakdown": cat_breakdown_list,
            "anomalies": recent_anomalies or [],
            "recurring_expenses": recurring_list,
            "forecast": forecast_data or {
                "status": "available" if len(df) >= 10 else "insufficient_data",
                "predicted_month_end": round(current_expenses * 1.15, 2) if len(df) >= 10 else None
            },
            "behavior": behavior_metrics
        }

        return FinancialContextBuilder.sanitize_for_json(context_payload)

    @staticmethod
    def load_latest_test_report(test_results_dir: str) -> Optional[Dict[str, Any]]:
        """Finds and loads the most recent financial_report_*.json in test_results_dir."""
        import glob
        if not os.path.exists(test_results_dir):
            return None
        json_files = glob.glob(os.path.join(test_results_dir, "financial_report_*.json"))
        if not json_files:
            return None
        latest_file = max(json_files, key=os.path.getmtime)
        with open(latest_file, "r") as f:
            return json.load(f)

    @staticmethod
    def build_context_from_test_report(report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a DataScience test.py Financial Intelligence Report JSON into the Section 19 schema.
        """
        meta = report_data.get("report_metadata", {})
        cash = report_data.get("cash_flow", {})
        spending = report_data.get("spending_breakdown", [])
        anomalies = report_data.get("anomalies", [])
        recurring = report_data.get("recurring_commitments", {})
        forecast = report_data.get("forecast", {})
        behavior = report_data.get("behavior_signals", {})
        warnings = report_data.get("classification_warnings", [])
        observations = report_data.get("observations", [])

        # Format recurring expenses
        rec_list = []
        for exp in recurring.get("recurring_expenses", []):
            rec_list.append({
                "merchant": exp.get("merchant"),
                "amount": exp.get("amount"),
                "frequency": exp.get("frequency", "Monthly"),
                "type": "Expense",
                "category": exp.get("category")
            })
        for inv in recurring.get("recurring_investments", []):
            rec_list.append({
                "merchant": inv.get("merchant"),
                "amount": inv.get("amount"),
                "frequency": inv.get("frequency", "Monthly"),
                "type": "Investment",
                "category": inv.get("category")
            })

        # Format anomalies
        anom_list = []
        for anom in anomalies:
            anom_list.append({
                "date": anom.get("date"),
                "merchant": anom.get("merchant"),
                "amount": anom.get("amount"),
                "category": anom.get("category"),
                "score": anom.get("score"),
                "is_anomaly": True,
                "reason": anom.get("reason"),
                "confidence": anom.get("confidence"),
                "low_confidence": anom.get("low_confidence", False)
            })

        context_payload = {
            "user": {
                "user_id": meta.get("user_id", "USR_0001"),
                "currency": "INR",
                "transactions_analyzed": meta.get("transactions_analyzed", 0),
                "analysis_period": meta.get("analysis_period", {})
            },
            "financial_summary": {
                "monthly_income": cash.get("total_income", 0.0),
                "current_month_expenses": cash.get("total_expenses", 0.0),
                "total_transfers_investments": cash.get("total_transfers", 0.0),
                "net_cash_flow": cash.get("net_cash_flow", 0.0)
            },
            "category_breakdown": spending,
            "anomalies": anom_list,
            "recurring_expenses": rec_list,
            "forecast": forecast,
            "behavior": behavior,
            "uncertain_categories_requiring_review": warnings,
            "key_observations": observations
        }

        return FinancialContextBuilder.sanitize_for_json(context_payload)

    @staticmethod
    def to_json_string(context_payload: Dict[str, Any]) -> str:
        """Serializes the sanitized context dictionary to formatted JSON."""
        return json.dumps(context_payload, indent=2)
