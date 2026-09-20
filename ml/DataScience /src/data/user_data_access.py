"""
User Data Access & Context Isolation Layer.
Ensures strict multi-tenant user data boundaries across the entire system.

ARCHITECTURAL PRINCIPLE:
GLOBAL MODEL != GLOBAL USER DATA.
- Global ML models (Classifier, Anomaly Regressors, Forecaster, K-Means) share weights.
- User Financial State (Transactions, Baselines, Profiles, LLM Context, Conversations)
  is strictly partitioned by authenticated `user_id`.
"""

import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

class UserDataAccessLayer:
    """
    Data Access Layer enforcing explicit user-level data isolation.
    All read/write operations require an authenticated user_id.
    """
    def __init__(self, historical_transactions_path: Optional[str] = None):
        self._user_transactions_store: Dict[str, pd.DataFrame] = {}
        self._user_profiles_store: Dict[str, Dict[str, Any]] = {}
        self._user_conversations_store: Dict[str, Dict[str, List[Dict[str, str]]]] = {}

        if historical_transactions_path and os.path.exists(historical_transactions_path):
            self._load_and_partition_historical_data(historical_transactions_path)

    def _load_and_partition_historical_data(self, csv_path: str):
        """
        Loads and partitions transactions strictly by user_id into isolated in-memory stores.
        """
        df = pd.read_csv(csv_path)
        if "user_id" in df.columns:
            for uid, group in df.groupby("user_id"):
                self._user_transactions_store[str(uid)] = group.copy().reset_index(drop=True)

    def add_user_transactions(self, user_id: str, transactions_df: pd.DataFrame):
        """
        Stores or appends transactions strictly into the isolated partition for `user_id`.
        """
        uid = str(user_id)
        df_to_add = transactions_df.copy()
        df_to_add["user_id"] = uid
        
        if uid in self._user_transactions_store:
            self._user_transactions_store[uid] = pd.concat(
                [self._user_transactions_store[uid], df_to_add],
                ignore_index=True
            ).drop_duplicates(subset=["transaction_id"] if "transaction_id" in df_to_add.columns else None).reset_index(drop=True)
        else:
            self._user_transactions_store[uid] = df_to_add.reset_index(drop=True)

    def get_user_transactions(self, user_id: str) -> pd.DataFrame:
        """
        Retrieves ONLY the transactions belonging to `user_id`.
        Prevents full table scans or global leaks.
        """
        uid = str(user_id)
        if uid not in self._user_transactions_store:
            return pd.DataFrame()
        return self._user_transactions_store[uid].copy()

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Constructs an isolated profile summary for `user_id`.
        """
        uid = str(user_id)
        df = self.get_user_transactions(uid)
        if df.empty:
            return {
                "user_id": uid,
                "status": "new_user",
                "transaction_count": 0,
                "total_spending": 0.0,
                "total_income": 0.0
            }

        expenses = df[df["category"] == "EXPENSE"]["amount"].sum()
        income = df[df["category"] == "INCOME"]["amount"].sum()
        transfers = df[df["category"] == "TRANSFER"]["amount"].sum()

        return {
            "user_id": uid,
            "status": "active",
            "transaction_count": len(df),
            "total_expenses": float(expenses),
            "total_income": float(income),
            "total_transfers": float(transfers),
            "first_transaction_date": str(df["date"].min()) if "date" in df.columns else "N/A",
            "last_transaction_date": str(df["date"].max()) if "date" in df.columns else "N/A"
        }

    # -------------------------------------------------------------------------
    # Conversation & LLM Memory Isolation
    # -------------------------------------------------------------------------
    def append_user_message(self, user_id: str, conversation_id: str, role: str, content: str):
        """
        Appends a chat message strictly scoped to (user_id, conversation_id).
        """
        uid = str(user_id)
        cid = str(conversation_id)
        if uid not in self._user_conversations_store:
            self._user_conversations_store[uid] = {}
        if cid not in self._user_conversations_store[uid]:
            self._user_conversations_store[uid][cid] = []

        self._user_conversations_store[uid][cid].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_user_conversation_history(self, user_id: str, conversation_id: str) -> List[Dict[str, str]]:
        """
        Retrieves conversation history strictly for the requested user and conversation.
        """
        uid = str(user_id)
        cid = str(conversation_id)
        return self._user_conversations_store.get(uid, {}).get(cid, []).copy()
