"""
MindSense AI - Summary Agent
================================
Generates concise summaries of conversation sessions for memory persistence.
Triggered automatically by the Orchestrator after a configurable number of turns.

The summary captures:
  - Main concerns expressed by the user
  - Key emotional themes
  - Coping strategies discussed
  - Overall session arc

Summaries are stored in SessionMemory and persisted to UserMemory.

Usage::

    from agents.summary_agent import SummaryAgent
    agent = SummaryAgent()
    result = agent.run(conversation_history=[...])
"""

from typing import Any, Dict, List, Optional

from model.llm.gemini_client import gemini_client
from utils.logger import get_logger
from utils.helpers import timer
from utils.response_utils import strip_markdown

logger = get_logger(__name__)


class SummaryAgent:
    """
    Conversation summarization agent.

    Uses Gemini to produce a concise clinical-style summary
    of a conversation session for memory persistence.
    """

    def __init__(self) -> None:
        logger.debug("SummaryAgent initialized.")

    def _format_conversation(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Format conversation history into a readable dialogue string for summarization.

        Args:
            conversation_history: List of ``{"role": ..., "content": ...}`` dicts.

        Returns:
            Formatted dialogue string.
        """
        lines = []
        for turn in conversation_history:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = turn.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @timer
    def run(
        self,
        conversation_history: List[Dict[str, str]],
        existing_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a session summary from conversation history.

        Args:
            conversation_history: Full list of conversation turns.
            existing_summary: Previous summary to update (incremental summarization).

        Returns:
            Dict containing:
              - ``summary`` (str): Generated session summary.
              - ``success`` (bool): Whether summarization succeeded.
              - ``agent`` (str): Agent identifier.

        Example::

            {
                "summary": "User is experiencing anxiety about upcoming exams...",
                "success": True,
                "agent": "SummaryAgent"
            }
        """
        if not conversation_history:
            return {
                "summary": "",
                "success": False,
                "agent": "SummaryAgent",
            }

        dialogue = self._format_conversation(conversation_history)

        # Build the summarization prompt
        if existing_summary:
            prompt = (
                f"You are a clinical AI assistant. Update the following existing session summary "
                f"with new information from the continuation of the conversation.\n\n"
                f"[EXISTING SUMMARY]\n{existing_summary}\n\n"
                f"[CONVERSATION CONTINUATION]\n{dialogue}\n\n"
                f"Write an updated summary (3-5 sentences) covering: "
                f"(1) the user's main concerns, (2) key emotional themes, "
                f"(3) any coping strategies discussed, (4) the overall session arc. "
                f"Be clinical, factual, and neutral. Do not include any personal advice.\n\n"
                f"[UPDATED SUMMARY]"
            )
        else:
            prompt = (
                f"You are a clinical AI assistant. Summarize the following therapy session "
                f"conversation in 3-5 sentences. Focus on: (1) the user's main concerns, "
                f"(2) key emotional themes, (3) any coping strategies discussed, "
                f"(4) the overall session arc. Be clinical, factual, and neutral.\n\n"
                f"[CONVERSATION]\n{dialogue}\n\n"
                f"[SUMMARY]"
            )

        try:
            raw_summary = gemini_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=300,
            )

            summary = strip_markdown(raw_summary).strip()

            if not summary:
                summary = self._heuristic_summary(conversation_history)

            logger.info(
                f"[SummaryAgent] Summary generated | "
                f"turns={len(conversation_history)} | chars={len(summary)}"
            )

            return {
                "summary": summary,
                "success": True,
                "agent": "SummaryAgent",
            }

        except Exception as e:
            logger.error(f"[SummaryAgent] Error generating summary: {e}")
            fallback = self._heuristic_summary(conversation_history)
            return {
                "summary": fallback,
                "success": False,
                "agent": "SummaryAgent",
                "error": str(e),
            }

    def _heuristic_summary(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate a simple heuristic summary without LLM.
        Used as fallback when Gemini is unavailable.

        Args:
            conversation_history: Conversation turns.

        Returns:
            Short plain-text summary.
        """
        user_messages = [
            t["content"] for t in conversation_history if t.get("role") == "user"
        ]

        if not user_messages:
            return "No user messages recorded in this session."

        n_turns = len(user_messages)
        first_msg = user_messages[0][:100] if user_messages else ""
        last_msg = user_messages[-1][:100] if user_messages else ""

        return (
            f"Session with {n_turns} user messages. "
            f"First message: '{first_msg}...'. "
            f"Last message: '{last_msg}...'. "
            f"Full LLM summarization was unavailable."
        )
