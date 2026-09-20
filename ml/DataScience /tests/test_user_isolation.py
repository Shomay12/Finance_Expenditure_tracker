#!/usr/bin/env python3
"""
================================================================================
USER ISOLATION & PERSONALIZATION VERIFICATION SUITE
================================================================================
Comprehensive test suite verifying multi-tenant data boundaries:
1. Transaction Partitioning & Scoped Retrieval
2. User-Specific Contextual Baselines
3. Cross-Contamination Anomaly Detection
4. Isolated Recurring Expense Profiles
5. Isolated Behavioral Feature Vectors
6. Isolated AI Orchestrator Context & Reports
7. Isolated Chat Memory & Conversation History

Principle Enforced: GLOBAL MODEL != GLOBAL USER DATA.
================================================================================
"""

import os
import sys
import json
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.user_data_access import UserDataAccessLayer
from src.models.anomaly_detector import ContextualAnomalyDetector
from src.models.recurring_detector import RecurringExpenseDetector
from src.models.user_segmenter import UserSegmenter
from src.orchestrator import FinancialIntelligenceOrchestrator
from test import FinancialIntelligenceReportGenerator


class TestUserIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(BASE_DIR, "models")
        cls.orchestrator = FinancialIntelligenceOrchestrator(cls.models_dir)
        cls.segmenter = UserSegmenter.load(os.path.join(cls.models_dir, "segmentation"))
        cls.recurring_detector = RecurringExpenseDetector(min_occurrences=2)

        # ---------------------------------------------------------------------
        # Create Synthetic User A & User B Datasets (Indian Context)
        # ---------------------------------------------------------------------
        # User A: Moderate earner / spender (Food ~ ₹500, Rent ₹30,000, Netflix ₹649)
        user_a_rows = [
            {"transaction_id": "TXN_A_01", "user_id": "USER_A", "date": "2026-06-01", "merchant": "Swiggy", "description": "Swiggy meal", "amount": 500.0, "category": "EXPENSE", "subcategory": "Food", "transaction_type": "debit", "payment_method": "UPI"},
            {"transaction_id": "TXN_A_02", "user_id": "USER_A", "date": "2026-06-05", "merchant": "Zomato", "description": "Zomato dinner", "amount": 450.0, "category": "EXPENSE", "subcategory": "Food", "transaction_type": "debit", "payment_method": "UPI"},
            {"transaction_id": "TXN_A_03", "user_id": "USER_A", "date": "2026-06-10", "merchant": "Swiggy", "description": "Lunch box", "amount": 550.0, "category": "EXPENSE", "subcategory": "Food", "transaction_type": "debit", "payment_method": "UPI"},
            {"transaction_id": "TXN_A_04", "user_id": "USER_A", "date": "2026-06-12", "merchant": "Myntra", "description": "Casual t-shirt", "amount": 2000.0, "category": "EXPENSE", "subcategory": "Shopping", "transaction_type": "debit", "payment_method": "Credit Card"},
            {"transaction_id": "TXN_A_05", "user_id": "USER_A", "date": "2026-06-02", "merchant": "NoBroker RentPay", "description": "House rent transfer", "amount": 30000.0, "category": "EXPENSE", "subcategory": "Rent & Housing", "transaction_type": "debit", "payment_method": "Net Banking"},
            {"transaction_id": "TXN_A_06", "user_id": "USER_A", "date": "2026-06-15", "merchant": "Netflix India", "description": "Monthly subscription", "amount": 649.0, "category": "EXPENSE", "subcategory": "Subscriptions", "transaction_type": "debit", "payment_method": "Credit Card"},
            {"transaction_id": "TXN_A_07", "user_id": "USER_A", "date": "2026-07-15", "merchant": "Netflix India", "description": "Monthly subscription", "amount": 649.0, "category": "EXPENSE", "subcategory": "Subscriptions", "transaction_type": "debit", "payment_method": "Credit Card"},
            {"transaction_id": "TXN_A_08", "user_id": "USER_A", "date": "2026-06-01", "merchant": "Tech Corp Bangalore", "description": "Monthly Salary", "amount": 85000.0, "category": "INCOME", "subcategory": "Salary", "transaction_type": "credit", "payment_method": "Bank Transfer"},
        ]
        cls.df_user_a = pd.DataFrame(user_a_rows)

        # User B: High earner / luxury spender (Food ~ ₹5,000, Rent ₹80,000, Netflix Premium ₹1,499)
        user_b_rows = [
            {"transaction_id": "TXN_B_01", "user_id": "USER_B", "date": "2026-06-01", "merchant": "ITC Grand Chola Dining", "description": "Gourmet meal", "amount": 5000.0, "category": "EXPENSE", "subcategory": "Food", "transaction_type": "debit", "payment_method": "Credit Card"},
            {"transaction_id": "TXN_B_02", "user_id": "USER_B", "date": "2026-06-05", "merchant": "Taj West End", "description": "Executive dining", "amount": 4500.0, "category": "EXPENSE", "subcategory": "Food", "transaction_type": "debit", "payment_method": "Credit Card"},
            {"transaction_id": "TXN_B_03", "user_id": "USER_B", "date": "2026-06-10", "merchant": "Leela Palace", "description": "Brunch with clients", "amount": 5500.0, "category": "EXPENSE", "subcategory": "Food", "transaction_type": "debit", "payment_method": "Credit Card"},
            {"transaction_id": "TXN_B_04", "user_id": "USER_B", "date": "2026-06-12", "merchant": "Armani Exchange", "description": "Designer wear", "amount": 20000.0, "category": "EXPENSE", "subcategory": "Shopping", "transaction_type": "debit", "payment_method": "Credit Card"},
            {"transaction_id": "TXN_B_05", "user_id": "USER_B", "date": "2026-06-02", "merchant": "Luxury Villa Rentals", "description": "Villa rental payment", "amount": 80000.0, "category": "EXPENSE", "subcategory": "Rent & Housing", "transaction_type": "debit", "payment_method": "Net Banking"},
            {"transaction_id": "TXN_B_06", "user_id": "USER_B", "date": "2026-06-15", "merchant": "Netflix India", "description": "4K Family Plan", "amount": 1499.0, "category": "EXPENSE", "subcategory": "Subscriptions", "transaction_type": "debit", "payment_method": "Credit Card"},
            {"transaction_id": "TXN_B_07", "user_id": "USER_B", "date": "2026-07-15", "merchant": "Netflix India", "description": "4K Family Plan", "amount": 1499.0, "category": "EXPENSE", "subcategory": "Subscriptions", "transaction_type": "debit", "payment_method": "Credit Card"},
            {"transaction_id": "TXN_B_08", "user_id": "USER_B", "date": "2026-06-01", "merchant": "Global Hedge Fund", "description": "Partner Salary & Dividend", "amount": 350000.0, "category": "INCOME", "subcategory": "Salary", "transaction_type": "credit", "payment_method": "Bank Transfer"},
        ]
        cls.df_user_b = pd.DataFrame(user_b_rows)

        # Register baselines dynamically
        cls.orchestrator.register_user_history("USER_A", cls.df_user_a)
        cls.orchestrator.register_user_history("USER_B", cls.df_user_b)

        # Setup Data Access Layer
        cls.data_layer = UserDataAccessLayer()
        cls.data_layer.add_user_transactions("USER_A", cls.df_user_a)
        cls.data_layer.add_user_transactions("USER_B", cls.df_user_b)

    def test_01_transaction_isolation(self):
        """Verify transaction store returns strictly isolated partitions."""
        txns_a = self.data_layer.get_user_transactions("USER_A")
        txns_b = self.data_layer.get_user_transactions("USER_B")

        self.assertEqual(len(txns_a), len(self.df_user_a))
        self.assertEqual(len(txns_b), len(self.df_user_b))
        self.assertTrue((txns_a["user_id"] == "USER_A").all())
        self.assertTrue((txns_b["user_id"] == "USER_B").all())

        # Assert ID disjointness
        ids_a = set(txns_a["transaction_id"])
        ids_b = set(txns_b["transaction_id"])
        self.assertTrue(ids_a.isdisjoint(ids_b))

    def test_02_user_specific_baselines(self):
        """Verify personal category baselines are isolated."""
        anom_detector = self.orchestrator.anomaly_detector

        base_a = anom_detector.user_category_baselines.get(("USER_A", "Food"))
        base_b = anom_detector.user_category_baselines.get(("USER_B", "Food"))

        self.assertIsNotNone(base_a, "User A Food baseline missing")
        self.assertIsNotNone(base_b, "User B Food baseline missing")

        self.assertAlmostEqual(base_a["mean"], 500.0, places=1)
        self.assertAlmostEqual(base_b["mean"], 5000.0, places=1)
        self.assertNotEqual(base_a["mean"], base_b["mean"])

    def test_03_cross_contamination_anomaly_detection(self):
        """
        Verify that a ₹5,000 Food transaction is anomalous for User A (10x baseline)
        but NORMAL for User B (1.0x baseline).
        """
        anom_detector = self.orchestrator.anomaly_detector

        # User A with ₹5,000 Food Spend -> MUST BE ANOMALOUS
        res_a_5000 = anom_detector.predict_transaction(
            user_id="USER_A",
            merchant="Luxury Dining Restaurant",
            subcategory="Food",
            amount=5000.0
        )

        # User B with ₹5,000 Food Spend -> MUST BE NORMAL (Inlier)
        res_b_5000 = anom_detector.predict_transaction(
            user_id="USER_B",
            merchant="ITC Grand Chola Dining",
            subcategory="Food",
            amount=5000.0
        )

        # Print Isolation Test Results
        base_a_mean = anom_detector.user_category_baselines[("USER_A", "Food")]["mean"]
        base_b_mean = anom_detector.user_category_baselines[("USER_B", "Food")]["mean"]

        print("\n" + "=" * 60)
        print("USER ISOLATION TEST (CROSS-CONTAMINATION)")
        print("=" * 60)
        print(f"User A baseline:\nFood average = ₹{base_a_mean:,.2f}\n")
        print(f"User B baseline:\nFood average = ₹{base_b_mean:,.2f}\n")
        print("Test transaction:\nUser A / Food / ₹5,000\n")
        print("Used baseline:\nUSER A")
        print(f"Result:\nIs Anomaly: {res_a_5000['is_anomaly']} | Score: {res_a_5000['anomaly_score']} | Reason: {res_a_5000['reason']}")
        print(f"PASS/FAIL:\n{'PASS' if res_a_5000['is_anomaly'] else 'FAIL'}")
        print("-" * 60)
        print("Test transaction:\nUser B / Food / ₹5,000\n")
        print("Used baseline:\nUSER B")
        print(f"Result:\nIs Anomaly: {res_b_5000['is_anomaly']} | Score: {res_b_5000['anomaly_score']} | Reason: {res_b_5000['reason']}")
        print(f"PASS/FAIL:\n{'PASS' if not res_b_5000['is_anomaly'] else 'FAIL'}")
        print("=" * 60 + "\n")

        self.assertTrue(res_a_5000["is_anomaly"], "User A ₹5,000 food transaction should be anomalous")
        self.assertFalse(res_b_5000["is_anomaly"], "User B ₹5,000 food transaction should NOT be anomalous")
        self.assertGreater(res_a_5000["anomaly_score"], res_b_5000["anomaly_score"])

    def test_04_recurring_expense_isolation(self):
        """Verify recurring subscription detection operates per-user without cross-pollution."""
        rec_a = self.recurring_detector.fit_and_detect_user(self.df_user_a)
        rec_b = self.recurring_detector.fit_and_detect_user(self.df_user_b)

        self.assertTrue(len(rec_a) >= 1)
        self.assertTrue(len(rec_b) >= 1)

        amt_a = rec_a[0]["recurring_amount"]
        amt_b = rec_b[0]["recurring_amount"]

        self.assertEqual(amt_a, 649.0, "User A Netflix subscription should be ₹649")
        self.assertEqual(amt_b, 1499.0, "User B Netflix subscription should be ₹1,499")
        self.assertNotEqual(amt_a, amt_b)

    def test_05_behavioral_feature_isolation(self):
        """Verify user behavioral feature vectors are generated independently."""
        vecs_a = self.segmenter.extract_user_vectors(self.df_user_a)
        vecs_b = self.segmenter.extract_user_vectors(self.df_user_b)

        self.assertEqual(len(vecs_a), 1)
        self.assertEqual(len(vecs_b), 1)

        row_a = vecs_a.iloc[0]
        row_b = vecs_b.iloc[0]

        self.assertEqual(row_a["user_id"], "USER_A")
        self.assertEqual(row_b["user_id"], "USER_B")
        self.assertNotEqual(row_a["monthly_income"], row_b["monthly_income"])
        self.assertNotEqual(row_a["monthly_expense"], row_b["monthly_expense"])

    def test_06_llm_context_and_report_isolation(self):
        """Verify AI Orchestrator builds strictly disjoint LLM contexts and reports."""
        ctx_a = self.orchestrator.build_user_llm_context(
            user_id="USER_A",
            user_transactions_df=self.df_user_a,
            user_query="Why did I spend more this month?"
        )

        ctx_b = self.orchestrator.build_user_llm_context(
            user_id="USER_B",
            user_transactions_df=self.df_user_b,
            user_query="Why did I spend more this month?"
        )

        self.assertEqual(ctx_a["authenticated_user_id"], "USER_A")
        self.assertEqual(ctx_b["authenticated_user_id"], "USER_B")

        # Assert no User B financial values are present in User A's context
        ctx_a_str = json.dumps(ctx_a)
        ctx_b_str = json.dumps(ctx_b)

        self.assertNotIn("350000", ctx_a_str)
        self.assertNotIn("80000", ctx_a_str)
        self.assertNotIn("USER_B", ctx_a_str)

        self.assertNotIn("85000", ctx_b_str)
        self.assertNotIn("30000", ctx_b_str)
        self.assertNotIn("USER_A", ctx_b_str)

        # Generate Reports
        rep_a_txt, rep_a_json = FinancialIntelligenceReportGenerator.generate_report(
            transactions_results=self.df_user_a.to_dict(orient="records"),
            user_id="USER_A",
            save_report=False
        )

        rep_b_txt, rep_b_json = FinancialIntelligenceReportGenerator.generate_report(
            transactions_results=self.df_user_b.to_dict(orient="records"),
            user_id="USER_B",
            save_report=False
        )

        self.assertIn("USER_A", rep_a_txt)
        self.assertNotIn("USER_B", rep_a_txt)
        self.assertIn("USER_B", rep_b_txt)
        self.assertNotIn("USER_A", rep_b_txt)

    def test_07_conversation_history_isolation(self):
        """Verify multi-user chat memory partitions."""
        self.data_layer.append_user_message("USER_A", "CHAT_01", "user", "What is my food budget?")
        self.data_layer.append_user_message("USER_A", "CHAT_01", "assistant", "Your food budget is ₹500/day.")

        self.data_layer.append_user_message("USER_B", "CHAT_02", "user", "What is my food budget?")
        self.data_layer.append_user_message("USER_B", "CHAT_02", "assistant", "Your luxury dining budget is ₹5,000/day.")

        hist_a = self.data_layer.get_user_conversation_history("USER_A", "CHAT_01")
        hist_b = self.data_layer.get_user_conversation_history("USER_B", "CHAT_02")

        self.assertEqual(len(hist_a), 2)
        self.assertEqual(len(hist_b), 2)

        self.assertIn("500", hist_a[1]["content"])
        self.assertNotIn("5,000", hist_a[1]["content"])

        self.assertIn("5,000", hist_b[1]["content"])
        self.assertNotIn("500/day", hist_b[1]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
