"""
MindSense AI - Prompt Builder
================================
Assembles the complete LLM prompt from all available context:
  - System instructions
  - Conversation history
  - Classification & risk analysis
  - Retrieved therapy knowledge (from RAG)
  - User message

The prompt is carefully structured to ground Gemini's responses
in evidence-based psychology and prevent hallucination.

Usage::

    from rag.prompt_builder import prompt_builder
    prompt = prompt_builder.build(
        user_message="I can't stop worrying about everything.",
        conversation_history=[...],
        classification={"label": "Anxiety", "confidence": 0.88},
        risk_level="low",
        intent="vent",
        retrieved_context=[...],
        session_summary="User has been experiencing work-related anxiety.",
    )
"""

from typing import Any, Dict, List, Optional

from prompts.system_prompt import SYSTEM_PROMPT
from prompts.therapist_prompt import THERAPIST_PROMPT_TEMPLATE
from prompts.crisis_prompt import CRISIS_PROMPT
from utils.logger import get_logger
from utils.response_utils import format_sources_for_prompt, format_conversation_for_prompt
from utils.helpers import truncate_to_token_limit

logger = get_logger(__name__)


class PromptBuilder:
    """
    Assembles structured LLM prompts from multi-source context.

    The prompt follows this structure:
      [SYSTEM INSTRUCTIONS]
      [SESSION SUMMARY]
      [CONVERSATION HISTORY]
      [USER PROFILE]
      [CLINICAL ANALYSIS]
      [RETRIEVED KNOWLEDGE]
      [TASK INSTRUCTIONS]
      [USER MESSAGE]

    This ordering ensures the model sees high-level instructions first,
    then progressively more specific context.
    """

    def build(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        classification: Optional[Dict[str, Any]] = None,
        risk_level: str = "low",
        intent: str = "general",
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        session_summary: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        is_crisis: bool = False,
    ) -> str:
        """
        Build the complete prompt for the Gemini LLM.

        Args:
            user_message: The user's current message.
            conversation_history: List of previous turns as ``{"role": ..., "content": ...}`` dicts.
            classification: Mental health classification result dict.
            risk_level: Risk assessment level ("low", "moderate", "high", "critical").
            intent: Detected user intent (e.g., "vent", "seek_advice", "crisis", "info").
            retrieved_context: List of RAG-retrieved knowledge chunks.
            session_summary: Summary of the current session so far.
            user_profile: Persistent user profile data (preferences, history).
            analysis: Deep analysis dict from the analyzer agent.
            is_crisis: If True, use the crisis prompt template instead of therapist template.

        Returns:
            Complete prompt string ready for Gemini API submission.
        """
        sections: List[str] = []

        # ─── 1. System Instructions ────────────────────────────────────────
        if is_crisis or risk_level in ("high", "critical"):
            sections.append(CRISIS_PROMPT)
            logger.info("Using CRISIS prompt template due to elevated risk level.")
        else:
            sections.append(SYSTEM_PROMPT)

        # ─── 2. Session Summary ────────────────────────────────────────────
        if session_summary and session_summary.strip():
            sections.append(
                f"[SESSION CONTEXT]\n{session_summary.strip()}"
            )

        # ─── 3. User Profile ───────────────────────────────────────────────
        if user_profile and any(v for v in user_profile.values() if v):
            profile_lines = ["[USER PROFILE]"]
            if user_profile.get("preferred_name"):
                profile_lines.append(f"- Preferred Name: {user_profile['preferred_name']}")
            if user_profile.get("concerns"):
                profile_lines.append(f"- Recurring Concerns: {', '.join(user_profile['concerns'])}")
            if user_profile.get("coping_preferences"):
                profile_lines.append(
                    f"- Preferred Coping Strategies: {', '.join(user_profile['coping_preferences'])}"
                )
            sections.append("\n".join(profile_lines))

        # ─── 4. Conversation History ───────────────────────────────────────
        if conversation_history:
            history_block = format_conversation_for_prompt(
                conversation_history, max_turns=6
            )
            sections.append(history_block)

        # ─── 5. Clinical Analysis ──────────────────────────────────────────
        analysis_lines = ["[CLINICAL ANALYSIS]"]

        if classification:
            label = classification.get("label", "Unknown")
            confidence = classification.get("confidence", 0.0)
            analysis_lines.append(f"- Detected Condition: {label} (confidence: {confidence:.0%})")
            high_risk = classification.get("high_risk", False)
            if high_risk:
                analysis_lines.append("- ⚠ HIGH RISK FLAG: This classification requires careful empathetic handling.")

        analysis_lines.append(f"- Risk Level: {risk_level.upper()}")
        analysis_lines.append(f"- User Intent: {intent.replace('_', ' ').title()}")

        if analysis:
            if analysis.get("dominant_emotion"):
                analysis_lines.append(f"- Dominant Emotion: {analysis['dominant_emotion']}")
            if analysis.get("themes"):
                themes = ", ".join(analysis["themes"][:3])
                analysis_lines.append(f"- Key Themes: {themes}")
            if analysis.get("sentiment"):
                analysis_lines.append(f"- Sentiment: {analysis['sentiment']}")

        sections.append("\n".join(analysis_lines))

        # ─── 6. Retrieved Knowledge Context ───────────────────────────────
        if retrieved_context:
            knowledge_block = format_sources_for_prompt(retrieved_context)
            sections.append(knowledge_block)
        else:
            sections.append(
                "[RETRIEVED KNOWLEDGE CONTEXT]\n"
                "No specific knowledge retrieved. Rely on your training and empathetic approach."
            )

        # ─── 7. Task Instructions (Therapist Prompt) ───────────────────────
        therapist_block = THERAPIST_PROMPT_TEMPLATE.format(
            label=classification.get("label", "general concern") if classification else "general concern",
            risk_level=risk_level,
            intent=intent.replace("_", " "),
        )
        sections.append(therapist_block)

        # ─── 8. User Message ───────────────────────────────────────────────
        sections.append(f"[USER MESSAGE]\n{user_message.strip()}")

        # ─── 9. Response Instruction ───────────────────────────────────────
        sections.append(
            "[YOUR RESPONSE]\n"
            "Write your empathetic, grounded, evidence-based therapeutic response below. "
            "Do NOT include section headers, labels, or metadata in your response. "
            "Write in plain conversational prose as if speaking directly to the user."
        )

        # Assemble full prompt
        full_prompt = "\n\n".join(sections)

        # Trim to token limit to avoid API errors
        full_prompt = truncate_to_token_limit(full_prompt, max_tokens=30000)

        logger.debug(
            f"Prompt built | sections={len(sections)} | "
            f"chars={len(full_prompt)} | is_crisis={is_crisis}"
        )

        return full_prompt

    def build_validation_prompt(
        self,
        original_message: str,
        response: str,
        classification_label: str = "Unknown",
    ) -> str:
        """
        Build a prompt that asks Gemini to self-evaluate a response.

        Args:
            original_message: The user's message.
            response: The response to evaluate.
            classification_label: The predicted mental health label.

        Returns:
            Validation prompt string.
        """
        from prompts.validator_prompt import VALIDATOR_PROMPT_TEMPLATE
        return VALIDATOR_PROMPT_TEMPLATE.format(
            user_message=original_message,
            response=response,
            label=classification_label,
        )

    def build_summary_prompt(self, conversation_text: str) -> str:
        """
        Build a prompt to summarize a conversation session.

        Args:
            conversation_text: Full conversation formatted as dialogue.

        Returns:
            Summarization prompt string.
        """
        return (
            "You are a clinical AI assistant. Summarize the following therapy session "
            "conversation in 3-5 sentences. Focus on: (1) the user's main concerns, "
            "(2) emotional themes identified, (3) any coping strategies discussed. "
            "Keep the summary clinical, factual, and neutral.\n\n"
            f"[CONVERSATION]\n{conversation_text}\n\n"
            "[SUMMARY]"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
prompt_builder = PromptBuilder()
