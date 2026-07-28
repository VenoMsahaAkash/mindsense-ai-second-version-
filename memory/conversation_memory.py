"""
MindSense AI - Conversation Memory
=====================================
Maintains in-session turn-by-turn conversation history.
Stores the raw message log and provides formatted retrieval
for prompt injection.

This is reset when the session ends (in-memory only).

Usage::

    from memory.conversation_memory import ConversationMemory
    memory = ConversationMemory(session_id="abc123")
    memory.add_turn("user", "I feel anxious today")
    memory.add_turn("assistant", "I hear you. Anxiety can be overwhelming...")
    history = memory.get_history()
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """
    In-session conversation history manager.

    Stores raw conversation turns and provides formatted
    retrieval for prompt injection and session summarization.

    Attributes:
        session_id (str): Unique session identifier.
        max_turns (int): Maximum number of turns to retain in memory.
        _turns (List[Dict]): Internal list of conversation turns.
    """

    def __init__(self, session_id: str, max_turns: int = 20) -> None:
        """
        Initialize the conversation memory.

        Args:
            session_id: Unique identifier for this conversation session.
            max_turns: Maximum turns to keep in memory (older turns are pruned).
        """
        self.session_id: str = session_id
        self.max_turns: int = max_turns
        self._turns: List[Dict[str, Any]] = []
        self._created_at: str = datetime.now(timezone.utc).isoformat()

        logger.debug(f"ConversationMemory initialized | session={session_id}")

    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a new turn to the conversation history.

        Args:
            role: Speaker role — "user" or "assistant".
            content: The message text content.
            metadata: Optional extra context (classification, risk_level, etc.).

        Raises:
            ValueError: If role is not "user" or "assistant".
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'.")

        turn = {
            "role": role,
            "content": content.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn_index": len(self._turns),
        }

        if metadata:
            turn["metadata"] = metadata

        self._turns.append(turn)

        # Prune oldest turns if over limit
        if len(self._turns) > self.max_turns:
            pruned = len(self._turns) - self.max_turns
            self._turns = self._turns[-self.max_turns:]
            logger.debug(f"Memory pruned {pruned} old turns | session={self.session_id}")

        logger.debug(
            f"Turn added | session={self.session_id} | role={role} | "
            f"total_turns={len(self._turns)}"
        )

    def get_history(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Retrieve conversation history as a list of role/content dicts.

        Args:
            max_turns: If set, return only the N most recent turns.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts (most recent last).
        """
        turns = self._turns if max_turns is None else self._turns[-max_turns:]
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    def get_full_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve full turn records including timestamps and metadata.

        Returns:
            List of full turn dicts.
        """
        return list(self._turns)

    def get_formatted_dialogue(self, max_turns: Optional[int] = None) -> str:
        """
        Format conversation history as a readable dialogue string.

        Args:
            max_turns: Number of most recent turns to include.

        Returns:
            Formatted dialogue string.

        Example::

            "User: I feel anxious\\nAssistant: I understand..."
        """
        history = self.get_history(max_turns)
        lines = []
        for turn in history:
            role = "User" if turn["role"] == "user" else "MindSense"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def get_last_user_message(self) -> Optional[str]:
        """
        Retrieve the most recent user message.

        Returns:
            Last user message string or None if no user turns exist.
        """
        for turn in reversed(self._turns):
            if turn["role"] == "user":
                return turn["content"]
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """
        Retrieve the most recent assistant message.

        Returns:
            Last assistant message string or None if no assistant turns exist.
        """
        for turn in reversed(self._turns):
            if turn["role"] == "assistant":
                return turn["content"]
        return None

    def clear(self) -> None:
        """Clear all turns from memory."""
        self._turns.clear()
        logger.debug(f"Conversation memory cleared | session={self.session_id}")

    @property
    def turn_count(self) -> int:
        """Total number of turns stored."""
        return len(self._turns)

    @property
    def user_message_count(self) -> int:
        """Number of user messages in history."""
        return sum(1 for t in self._turns if t["role"] == "user")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory state to a dictionary (for storage/export)."""
        return {
            "session_id": self.session_id,
            "created_at": self._created_at,
            "turn_count": self.turn_count,
            "turns": self._turns,
        }
