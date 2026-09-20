"""
Deterministic Pattern-Detection Engine for Recurring Expenses and Subscriptions.
Analyzes:
1. Normalized Merchant Similarity
2. Amount Consistency (Exact & Tolerance-based for dynamic bills)
3. Cadence Intervals (Weekly, Bi-weekly, Monthly, Quarterly, Annual)
4. Day-of-month anchoring
5. Historical occurrence frequency and Next Billing Date Projection
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from difflib import SequenceMatcher

def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

class RecurringExpenseDetector:
    def __init__(
        self,
        min_occurrences: int = 2,
        amount_tolerance: float = 0.12,  # Allow up to 12% variation for utility bills
        merchant_similarity_threshold: float = 0.75
    ):
        self.min_occurrences = min_occurrences
        self.amount_tolerance = amount_tolerance
        self.merchant_similarity_threshold = merchant_similarity_threshold
        self.detected_recurring = defaultdict(list)

    def fit_and_detect_user(self, user_df: pd.DataFrame) -> list:
        """
        Analyzes a single user's transaction history to detect all recurring streams.
        """
        if len(user_df) < self.min_occurrences:
            return []

        df = user_df.copy()
        df["date_dt"] = pd.to_datetime(df["date"])
        df = df.sort_values("date_dt").reset_index(drop=True)

        # Focus primarily on debits / expenses / transfers
        df = df[df["transaction_type"].isin(["debit", "transfer"])].reset_index(drop=True)

        # Cluster transactions by merchant name & amount
        clusters = []
        for idx, row in df.iterrows():
            m_name = str(row["merchant"]).strip()
            amt = float(row["amount"])
            dt = row["date_dt"]
            subcat = row.get("subcategory", "General")

            matched_cluster = None
            for cl in clusters:
                # Check merchant similarity
                sim = string_similarity(m_name, cl["merchant_sample"])
                if sim >= self.merchant_similarity_threshold:
                    # Check amount similarity against cluster median
                    median_amt = np.median(cl["amounts"])
                    diff = abs(amt - median_amt) / max(1.0, median_amt)
                    if diff <= self.amount_tolerance or subcat in ["Bills & Utilities", "Rent & Housing"]:
                        matched_cluster = cl
                        break

            if matched_cluster is not None:
                matched_cluster["transactions"].append(row)
                matched_cluster["dates"].append(dt)
                matched_cluster["amounts"].append(amt)
            else:
                clusters.append({
                    "merchant_sample": m_name,
                    "subcategory": subcat,
                    "transactions": [row],
                    "dates": [dt],
                    "amounts": [amt]
                })

        recurring_profiles = []
        for cl in clusters:
            if len(cl["dates"]) < self.min_occurrences:
                continue

            dates = sorted(cl["dates"])
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates) - 1)]

            if not intervals:
                continue

            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals) if len(intervals) > 1 else 0.0

            # Determine frequency cadence
            frequency = None
            if 25 <= mean_interval <= 35 and std_interval <= 5.0:
                frequency = "Monthly"
            elif 6 <= mean_interval <= 8 and std_interval <= 2.0:
                frequency = "Weekly"
            elif 12 <= mean_interval <= 16 and std_interval <= 3.0:
                frequency = "Bi-Weekly"
            elif 80 <= mean_interval <= 100 and std_interval <= 10.0:
                frequency = "Quarterly"
            elif 350 <= mean_interval <= 380 and std_interval <= 15.0:
                frequency = "Annual"

            # Check if day-of-month is highly consistent (for monthly)
            days_of_month = [d.day for d in dates]
            dom_std = np.std(days_of_month) if len(days_of_month) > 1 else 0.0

            if frequency or (len(dates) >= 3 and dom_std <= 2.5):
                if not frequency:
                    frequency = "Monthly"

                last_date = dates[-1]
                if frequency == "Monthly":
                    # Project next month
                    # Handle month rollover safely
                    year = last_date.year + (last_date.month // 12)
                    month = (last_date.month % 12) + 1
                    target_day = min(int(round(np.mean(days_of_month))), 28)
                    expected_next = datetime(year, month, target_day)
                elif frequency == "Weekly":
                    expected_next = last_date + timedelta(days=7)
                elif frequency == "Bi-Weekly":
                    expected_next = last_date + timedelta(days=14)
                elif frequency == "Quarterly":
                    expected_next = last_date + timedelta(days=90)
                else: # Annual
                    expected_next = datetime(last_date.year + 1, last_date.month, last_date.day)

                median_amount = round(float(np.median(cl["amounts"])), 2)
                confidence = float(np.clip(1.0 - (std_interval / max(1.0, mean_interval)), 0.60, 0.99))

                recurring_profiles.append({
                    "is_recurring": True,
                    "recurring_merchant": cl["merchant_sample"],
                    "subcategory": cl["subcategory"],
                    "recurring_amount": median_amount,
                    "recurrence_frequency": frequency,
                    "historical_occurrence_count": len(dates),
                    "last_billed_date": last_date.strftime("%Y-%m-%d"),
                    "expected_next_date": expected_next.strftime("%Y-%m-%d"),
                    "confidence_score": round(confidence, 3)
                })

        return recurring_profiles

    def detect_all(self, df_history: pd.DataFrame) -> dict:
        """
        Runs recurring detection over all users in the dataset.
        """
        all_results = {}
        for uid, u_df in df_history.groupby("user_id"):
            recs = self.fit_and_detect_user(u_df)
            all_results[uid] = recs
        self.detected_recurring = all_results
        return all_results


def run_recurring_pipeline(data_path: str):
    print("\n" + "=" * 70)
    print("DETERMINISTIC RECURRING EXPENSE & SUBSCRIPTION DETECTION")
    print("=" * 70)

    df_hist = pd.read_csv(data_path)
    detector = RecurringExpenseDetector(min_occurrences=3)
    results = detector.detect_all(df_hist)

    # Display sample user recurring stream
    sample_user = "USR_0001"
    sample_recs = results.get(sample_user, [])

    print(f"Detected {len(sample_recs)} recurring expenses / subscriptions for user [{sample_user}]:\n")
    for r in sample_recs:
        print(f"• Merchant: {r['recurring_merchant']:<30} | Amount: ₹{r['recurring_amount']:>8,.2f} | Cadence: {r['recurrence_frequency']:<9} | Next: {r['expected_next_date']} (Conf: {r['confidence_score']})")

    # Overall summary stats
    total_recurring_found = sum(len(v) for v in results.values())
    print(f"\nTotal Recurring Subscriptions Identified Across {len(results)} Users: {total_recurring_found}")
    return results


if __name__ == "__main__":
    hist_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/raw_historical_transactions.csv"))
    run_recurring_pipeline(hist_file)
