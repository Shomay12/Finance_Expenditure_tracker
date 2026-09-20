"""
User-Isolated Session & Conversation Memory Manager.
Guarantees strict isolation of multi-turn chat dialogues across tenant users and session IDs.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

class SessionMemoryManager:
    """
    In-memory conversation manager partitioned by (user_id, session_id).
    Enforces that conversation state cannot bleed across distinct user accounts.
    """
    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns
        # Store: {user_id: {session_id: [{"role": str, "content": str, "timestamp": str}]}}
        self._memory_store: Dict[str, Dict[str, List[Dict[str, str]]]] = {}

    def get_history(self, user_id: str, session_id: str = "default") -> List[Dict[str, str]]:
        """
        Retrieves formatted conversation turns strictly for the given user_id and session_id.
        """
        uid = str(user_id)
        sid = str(session_id)
        raw_turns = self._memory_store.get(uid, {}).get(sid, [])
        # Return last N turns without internal timestamps for LLM consumption
        return [{"role": t["role"], "content": t["content"]} for t in raw_turns[-self.max_history_turns * 2:]]

    def add_user_message(self, user_id: str, content: str, session_id: str = "default"):
        """Appends a user turn to the isolated user session."""
        self._append_message(user_id, session_id, "user", content)

    def add_assistant_message(self, user_id: str, content: str, session_id: str = "default"):
        """Appends an assistant turn to the isolated user session."""
        self._append_message(user_id, session_id, "assistant", content)

    def _append_message(self, user_id: str, session_id: str, role: str, content: str):
        uid = str(user_id)
        sid = str(session_id)
        if uid not in self._memory_store:
            self._memory_store[uid] = {}
        if sid not in self._memory_store[uid]:
            self._memory_store[uid][sid] = []

        self._memory_store[uid][sid].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def clear_session(self, user_id: str, session_id: str = "default"):
        """Clears memory for a specific session."""
        uid = str(user_id)
        sid = str(session_id)
        if uid in self._memory_store and sid in self._memory_store[uid]:
            self._memory_store[uid][sid] = []

    def get_session_count(self, user_id: str) -> int:
        """Returns number of active sessions for a user."""
        return len(self._memory_store.get(str(user_id), {}))
