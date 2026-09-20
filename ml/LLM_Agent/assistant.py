"""
Core AI Financial Intelligence Assistant Engine.
Coordinates Data Science Orchestration, Groq LLM Inference, User Isolation,
Session Memory, and Prompt Injection Defense.
"""

import os
import sys
import json
import logging
import importlib.util
from typing import Dict, Any, List, Optional, Iterator
import pandas as pd

# -----------------------------------------------------------------------------
# Path Resolution for DataScience Module & Workspace Root
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Resolve DataScience directory (handles paths with or without trailing space)
ds_candidates = [
    os.path.join(PROJECT_ROOT, "DataScience "),
    os.path.join(PROJECT_ROOT, "DataScience"),
    os.path.abspath(os.path.join(os.getcwd(), "DataScience ")),
    os.path.abspath(os.path.join(os.getcwd(), "DataScience"))
]
DS_DIR = next((p for p in ds_candidates if os.path.exists(p)), ds_candidates[0])

for p in [PROJECT_ROOT, DS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from LLM_Agent.config import AgentConfig
from LLM_Agent.groq_client import GroqClientWrapper
from LLM_Agent.memory_manager import SessionMemoryManager
from LLM_Agent.context_builder import FinancialContextBuilder
from LLM_Agent.prompts.system_prompt import (
    SYSTEM_PROMPT_CORE,
    build_complete_system_prompt,
    build_user_message_payload
)

from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class RequestContext:
    """
    Immutable request context enforcing authenticated user scope across the pipeline.
    (Section 14 & 19: Request & Session Isolation).
    """
    authenticated_user_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = "default"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# -----------------------------------------------------------------------------
# Safe Import of DataScience Modules
# -----------------------------------------------------------------------------
FinancialIntelligenceOrchestrator = None
UserDataAccessLayer = None

try:
    # pyrefly: ignore [missing-import]
    from src.orchestrator import FinancialIntelligenceOrchestrator
    # pyrefly: ignore [missing-import]
    from src.data.user_data_access import UserDataAccessLayer
except (ImportError, ModuleNotFoundError):
    # Dynamic loader fallback for various IDE/execution environments
    try:
        orch_path = os.path.join(DS_DIR, "src", "orchestrator.py")
        if os.path.exists(orch_path):
            spec = importlib.util.spec_from_file_location("orchestrator_module", orch_path)
            if spec and spec.loader:
                orch_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(orch_mod)
                FinancialIntelligenceOrchestrator = getattr(orch_mod, "FinancialIntelligenceOrchestrator", None)

        uda_path = os.path.join(DS_DIR, "src", "data", "user_data_access.py")
        if os.path.exists(uda_path):
            spec2 = importlib.util.spec_from_file_location("user_data_access_module", uda_path)
            if spec2 and spec2.loader:
                uda_mod = importlib.util.module_from_spec(spec2)
                spec2.loader.exec_module(uda_mod)
                UserDataAccessLayer = getattr(uda_mod, "UserDataAccessLayer", None)
    except Exception as err:
        logging.warning(f"Could not dynamically load DataScience modules: {err}")

logger = logging.getLogger("FinancialAssistant")


class FinancialIntelligenceAssistant:
    """
    Production-grade AI Financial Intelligence Assistant.
    Enforces that Data Science & ML is the source of truth, while Groq LLM
    converts verified facts into clear, evidence-based conversations.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        load_ds_orchestrator: bool = True
    ):
        self.config = config or AgentConfig()
        self.groq_client = GroqClientWrapper(self.config)
        self.memory_manager = SessionMemoryManager(max_history_turns=10)
        self.orchestrator = None
        self.data_access = None

        if load_ds_orchestrator and FinancialIntelligenceOrchestrator is not None and os.path.exists(self.config.models_dir):
            try:
                self.orchestrator = FinancialIntelligenceOrchestrator(self.config.models_dir)
                raw_txns_path = os.path.join(self.config.data_dir, "raw", "transactions_raw.csv")
                if UserDataAccessLayer is not None:
                    self.data_access = UserDataAccessLayer(
                        raw_txns_path if os.path.exists(raw_txns_path) else None
                    )
            except Exception as e:
                logger.warning(f"Could not load DS Orchestrator or UserDataAccessLayer: {e}")

    def chat(
        self,
        user_id: str,
        user_query: str,
        session_id: str = "default",
        custom_context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Any:
        """
        Executes a user-isolated conversational turn.
        
        Guarantees:
        1. Query and memory are scoped strictly to (user_id, session_id).
        2. Verified financial intelligence context is injected into system prompt.
        3. Prompt injection attempts inside query or data are neutralized.
        4. No secrets or API keys are ever leaked.
        """
        uid = str(user_id)
        sid = str(session_id)
        sanitized_query = build_user_message_payload(user_query)

        # 1. Retrieve or build structured financial context for current user
        if custom_context:
            context_dict = custom_context
            # Ensure user isolation within custom context
            if "user" not in context_dict:
                context_dict["user"] = {"user_id": uid}
            else:
                context_dict["user"]["user_id"] = uid
        elif self.data_access and self.orchestrator:
            u_df = self.data_access.get_user_transactions(uid)
            context_dict = FinancialContextBuilder.build_context_from_orchestrator(
                user_id=uid,
                user_transactions_df=u_df,
                orchestrator=self.orchestrator
            )
        else:
            # Minimal fallback context
            context_dict = {
                "user": {"user_id": uid, "status": "no_transactions_loaded"},
                "financial_summary": {
                    "monthly_income": 0.0,
                    "current_month_expenses": 0.0,
                    "previous_month_expenses": 0.0
                },
                "category_breakdown": [],
                "anomalies": [],
                "recurring_expenses": [],
                "forecast": {"status": "insufficient_data"}
            }

        context_json_str = FinancialContextBuilder.to_json_string(context_dict)

        # 2. Build complete System Prompt with verified current user context
        system_prompt = build_complete_system_prompt(user_context_json=context_json_str)

        # 3. Retrieve isolated session history
        history = self.memory_manager.get_history(user_id=uid, session_id=sid)

        # 4. Construct messages payload
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append(turn)
        messages.append({"role": "user", "content": sanitized_query})

        # 5. Execute LLM completion via Groq
        if stream:
            return self.groq_client.generate_chat_completion(messages=messages, stream=True)
        else:
            assistant_response = self.groq_client.generate_chat_completion(messages=messages, stream=False)
            # 6. Store turns in memory
            self.memory_manager.add_user_message(uid, sanitized_query, sid)
            self.memory_manager.add_assistant_message(uid, assistant_response, sid)
            return assistant_response

    def explain_incoming_transaction(
        self,
        user_id: str,
        merchant_raw: str,
        description: str,
        amount: float,
        payment_method: str = "UPI",
        session_id: str = "default"
    ) -> str:
        """
        Passes transaction through DS classifier and anomaly detector, then generates
        evidence-based explanation adhering to Section 6, 7, and 23.
        """
        uid = str(user_id)
        if not self.orchestrator:
            return "Financial intelligence pipeline is not initialized."

        eval_res = self.orchestrator.process_incoming_transaction(
            user_id=uid,
            merchant_raw=merchant_raw,
            description=description,
            amount=amount,
            payment_method=payment_method
        )

        cat_info = eval_res["categorization"]
        anom_info = eval_res["anomaly_detection"]

        live_context = {
            "user": {"user_id": uid},
            "evaluated_transaction": {
                "merchant": merchant_raw,
                "amount": float(amount),
                "category": cat_info.get("category"),
                "subcategory": cat_info.get("subcategory"),
                "confidence": cat_info.get("confidence", 0.0),
                "is_anomaly": anom_info.get("is_anomaly", False),
                "anomaly_score": anom_info.get("anomaly_score", 0.0),
                "anomaly_reason": anom_info.get("reason", "N/A"),
                "user_category_mean": anom_info.get("user_mean_amount", 0.0)
            }
        }

        query = f"Please explain my recent transaction of ₹{amount:,.2f} at {merchant_raw}."
        return self.chat(user_id=uid, user_query=query, session_id=session_id, custom_context=live_context)
