"""
MindSense AI - Agent Orchestrator
=====================================
Coordinates all agents in the MindSense AI pipeline.
Acts as the central controller that processes each user message
through the complete RAG + Multi-Agent pipeline.

Pipeline execution order:
  1. ClassifierAgent   → Predict mental health category
  2. IntentAgent       → Detect user intent
  3. RiskAgent         → Assess crisis/safety risk
  4. AnalyzerAgent     → Deep NLP analysis
  5. RAG Retriever     → Fetch relevant knowledge chunks
  6. RAG Reranker      → Rerank retrieved chunks
  7. PromptBuilder     → Assemble complete LLM prompt
  8. TherapistAgent    → Generate response via Gemini
  9. ValidatorAgent    → Score and optionally regenerate
  10. SummaryAgent     → Update session summary (async, every N turns)
  11. Memory Update    → Update conversation, session, and user memory

Usage::

    from agents.orchestrator import Orchestrator
    orch = Orchestrator()

    # Process a user message
    result = orch.process(
        user_message="I've been feeling really anxious lately.",
        session_id="session_abc123",
        user_id="user_xyz",
    )
    print(result["response"])
"""

from typing import Any, Dict, Generator, List, Optional

from agents.classifier_agent import ClassifierAgent
from agents.intent_agent import IntentAgent
from agents.risk_agent import RiskAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.therapist_agent import TherapistAgent
from agents.validator_agent import ValidatorAgent
from agents.summary_agent import SummaryAgent

from rag.retriever import retriever
from rag.reranker import reranker
from rag.prompt_builder import prompt_builder

from memory.conversation_memory import ConversationMemory
from memory.session_memory import SessionMemory
from memory.user_memory import UserMemory

from config import settings
from utils.logger import get_logger
from utils.helpers import timer, generate_session_id, get_utc_timestamp

logger = get_logger(__name__)


class Orchestrator:
    """
    Central pipeline coordinator for MindSense AI.

    Manages a pool of sessions (each with its own conversation and
    session memory) and routes every user message through the complete
    multi-agent pipeline.

    Thread safety: Each session has isolated memory objects.
    Multiple concurrent sessions are supported.

    Attributes:
        _sessions: Dict mapping session_id → {conversation, session} memory objects.
        _agents: Dict of initialized agent instances.
    """

    def __init__(self) -> None:
        # Initialize all agents (singletons, thread-safe)
        self._classifier = ClassifierAgent()
        self._intent = IntentAgent()
        self._risk = RiskAgent()
        self._analyzer = AnalyzerAgent()
        self._therapist = TherapistAgent()
        self._validator = ValidatorAgent()
        self._summarizer = SummaryAgent()

        # Session registry: session_id → memory objects
        self._sessions: Dict[str, Dict[str, Any]] = {}

        logger.info("Orchestrator initialized with all agents.")

    # ─────────────────────────────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────────────────────────────

    def get_or_create_session(
        self, session_id: str, user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Get an existing session or create a new one.

        Args:
            session_id: Unique session identifier.
            user_id: User identifier for cross-session memory.

        Returns:
            Session dict with ``conversation``, ``session``, and ``user`` memory objects.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "conversation": ConversationMemory(
                    session_id=session_id,
                    max_turns=settings.memory.MAX_CONVERSATION_TURNS,
                ),
                "session": SessionMemory(session_id=session_id, user_id=user_id),
                "user": UserMemory(user_id=user_id),
            }
            logger.info(f"New session created | session={session_id} | user={user_id}")

        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session from memory (e.g., on logout or session reset).

        Args:
            session_id: Session to clear.

        Returns:
            True if session was found and cleared, False otherwise.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session cleared | session={session_id}")
            return True
        return False

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get the conversation history for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of turn dicts, empty list if session doesn't exist.
        """
        if session_id not in self._sessions:
            return []
        return self._sessions[session_id]["conversation"].get_history()

    # ─────────────────────────────────────────────────────────────────────
    # Core Pipeline
    # ─────────────────────────────────────────────────────────────────────

    @timer
    def process(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        user_id: str = "anonymous",
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a user message through the complete MindSense AI pipeline.

        This is the main entry point for the Flask API.

        Args:
            user_message: The user's text input.
            session_id: Session ID (auto-generated if not provided).
            user_id: User identifier for persistent memory.
            stream: Whether to use streaming response generation.

        Returns:
            Dict containing the full pipeline result:
              - ``response`` (str): Final response text.
              - ``session_id`` (str): Session identifier.
              - ``classification`` (dict): Mental health classification.
              - ``risk_level`` (str): Risk assessment level.
              - ``intent`` (str): Detected user intent.
              - ``analysis`` (dict): Deep NLP analysis.
              - ``sources`` (list): Knowledge sources used.
              - ``validation`` (dict): Response quality scores.
              - ``is_crisis`` (bool): Whether crisis protocol was active.
              - ``timestamp`` (str): UTC timestamp.
        """
        # ── Session setup ──────────────────────────────────────────────────
        if not session_id:
            session_id = generate_session_id()

        session = self.get_or_create_session(session_id, user_id)
        conv_mem: ConversationMemory = session["conversation"]
        sess_mem: SessionMemory = session["session"]
        user_mem: UserMemory = session["user"]

        # Add user message to conversation memory immediately
        conv_mem.add_turn("user", user_message)

        logger.info(
            f"[Orchestrator] Processing message | session={session_id[:8]}... | "
            f"message_chars={len(user_message)} | turn={conv_mem.turn_count}"
        )

        try:
            # ── Step 1: Classification ─────────────────────────────────────
            classification = self._classifier.run(user_message)
            label = classification.get("label", "Normal")
            confidence = classification.get("confidence", 0.0)

            # Update session memory
            sess_mem.update_classification(label, confidence)

            # ── Step 2: Intent Detection ───────────────────────────────────
            intent_result = self._intent.run(user_message, label)
            intent = intent_result.get("intent", "general")
            sess_mem.update_intent(intent)

            # ── Step 3: Risk Assessment ────────────────────────────────────
            history_for_risk = conv_mem.get_history(max_turns=10)
            risk_result = self._risk.run(
                user_message=user_message,
                classification_label=label,
                classification_confidence=confidence,
                conversation_history=history_for_risk,
            )
            risk_level = risk_result.get("risk_level", "low")
            is_crisis = risk_result.get("requires_crisis_protocol", False)
            sess_mem.update_risk_level(risk_level)

            # ── Step 4: Deep Analysis ──────────────────────────────────────
            analysis = self._analyzer.run(user_message)
            sess_mem.add_themes(analysis.get("themes", []))

            # ── Step 5: RAG Retrieval ──────────────────────────────────────
            raw_results = retriever.retrieve_by_categories(
                query=user_message,
                per_category_k=2,
            )

            # ── Step 6: Reranking ──────────────────────────────────────────
            if raw_results:
                reranked_results = reranker.rerank(
                    query=user_message,
                    results=raw_results,
                    top_k=settings.rag.TOP_K_RERANK,
                )
            else:
                reranked_results = []

            sources = [r.get("source", "") for r in reranked_results if r.get("source")]

            # ── Step 7: Prompt Assembly ────────────────────────────────────
            conversation_history = conv_mem.get_history(max_turns=6)
            user_profile = user_mem.get_session_context()
            session_summary = sess_mem.summary

            prompt = prompt_builder.build(
                user_message=user_message,
                conversation_history=conversation_history,
                classification=classification,
                risk_level=risk_level,
                intent=intent,
                retrieved_context=reranked_results,
                session_summary=session_summary,
                user_profile=user_profile,
                analysis=analysis,
                is_crisis=is_crisis,
            )

            # ── Step 8: Response Generation ────────────────────────────────
            therapist_result = self._therapist.run(prompt=prompt)
            response_text = therapist_result.get("response", "")

            # ── Step 9: Validation & Regeneration ─────────────────────────
            previous_responses = [
                t["content"]
                for t in conv_mem.get_history()
                if t.get("role") == "assistant"
            ]

            validation_result = self._validator.validate(
                user_message=user_message,
                response=response_text,
                classification_label=label,
                previous_responses=previous_responses,
                use_llm_scoring=False,  # Fast mode by default
            )

            # Regenerate if below threshold
            regen_count = 0
            while (
                not validation_result["passed"]
                and regen_count < settings.validation.MAX_REGENERATION_ATTEMPTS
            ):
                regen_count += 1
                logger.warning(
                    f"[Orchestrator] Response below threshold "
                    f"(score={validation_result['overall_score']:.3f}), "
                    f"regenerating... attempt {regen_count}"
                )
                therapist_result = self._therapist.run(
                    prompt=prompt,
                    temperature=min(0.95, 0.7 + regen_count * 0.1),  # Slightly higher temperature
                )
                response_text = therapist_result.get("response", response_text)
                validation_result = self._validator.validate(
                    user_message=user_message,
                    response=response_text,
                    classification_label=label,
                    previous_responses=previous_responses,
                )

            # ── Step 10: Memory Update ─────────────────────────────────────
            conv_mem.add_turn(
                "assistant",
                response_text,
                metadata={
                    "validation_score": validation_result["overall_score"],
                    "risk_level": risk_level,
                    "label": label,
                },
            )

            # Trigger summarization every N turns
            if conv_mem.user_message_count % settings.memory.SUMMARY_TRIGGER_TURNS == 0:
                summary_result = self._summarizer.run(
                    conversation_history=conv_mem.get_history(),
                    existing_summary=sess_mem.summary,
                )
                if summary_result.get("success"):
                    sess_mem.set_summary(summary_result["summary"])

            # ── Build final result ─────────────────────────────────────────
            result = {
                "response": response_text,
                "session_id": session_id,
                "classification": {
                    "label": label,
                    "confidence": confidence,
                    "all_scores": classification.get("all_scores", {}),
                    "explainable_words": classification.get("explainable_words", []),
                    "model_architecture": classification.get("model_architecture", "DistilBERT + BiLSTM + Attention"),
                    "high_risk": classification.get("high_risk", False),
                },
                "risk_level": risk_level,
                "risk_score": risk_result.get("risk_score", 0.0),
                "is_crisis": is_crisis,
                "intent": intent,
                "analysis": {
                    "dominant_emotion": analysis.get("dominant_emotion"),
                    "sentiment": analysis.get("sentiment"),
                    "themes": analysis.get("themes", []),
                    "cognitive_distortions": analysis.get("cognitive_distortions", []),
                },
                "sources": list(set(sources)),
                "validation": {
                    "score": validation_result["overall_score"],
                    "passed": validation_result["passed"],
                    "regenerations": regen_count,
                },
                "turn_count": conv_mem.turn_count,
                "timestamp": get_utc_timestamp(),
            }

            logger.info(
                f"[Orchestrator] Pipeline complete | "
                f"label={label} | risk={risk_level} | "
                f"validation={validation_result['overall_score']:.3f} | "
                f"regen={regen_count} | sources={len(sources)}"
            )

            return result

        except Exception as e:
            logger.error(f"[Orchestrator] Pipeline error: {e}", exc_info=True)

            # Ensure user message is still preserved in memory
            fallback_response = (
                "I'm here and I want to support you. Could you tell me a bit more "
                "about what you're experiencing right now? I'm listening."
            )
            conv_mem.add_turn("assistant", fallback_response)

            return {
                "response": fallback_response,
                "session_id": session_id,
                "classification": {"label": "Normal", "confidence": 0.0},
                "risk_level": "low",
                "is_crisis": False,
                "intent": "general",
                "sources": [],
                "validation": {"score": 0.0, "passed": False},
                "timestamp": get_utc_timestamp(),
                "error": str(e),
            }

    def process_stream(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        user_id: str = "anonymous",
    ) -> Generator[str, None, None]:
        """
        Process a user message and yield streamed response chunks.

        Runs the full pre-processing pipeline synchronously, then
        streams the therapist response token-by-token.

        Args:
            user_message: The user's text input.
            session_id: Session ID (auto-generated if not provided).
            user_id: User identifier.

        Yields:
            Text chunks from the streaming Gemini response.
        """
        if not session_id:
            session_id = generate_session_id()

        session = self.get_or_create_session(session_id, user_id)
        conv_mem: ConversationMemory = session["conversation"]
        sess_mem: SessionMemory = session["session"]

        conv_mem.add_turn("user", user_message)

        try:
            # Run all pre-generation steps synchronously
            classification = self._classifier.run(user_message)
            label = classification.get("label", "Normal")
            intent_result = self._intent.run(user_message, label)
            intent = intent_result.get("intent", "general")
            risk_result = self._risk.run(user_message, label)
            risk_level = risk_result.get("risk_level", "low")
            is_crisis = risk_result.get("requires_crisis_protocol", False)
            analysis = self._analyzer.run(user_message)

            raw_results = retriever.retrieve_by_categories(user_message, per_category_k=2)
            reranked_results = reranker.rerank(user_message, raw_results) if raw_results else []

            prompt = prompt_builder.build(
                user_message=user_message,
                conversation_history=conv_mem.get_history(max_turns=6),
                classification=classification,
                risk_level=risk_level,
                intent=intent,
                retrieved_context=reranked_results,
                session_summary=sess_mem.summary,
                analysis=analysis,
                is_crisis=is_crisis,
            )

            # Stream the therapist response
            full_response = ""
            for chunk in self._therapist.run_stream(prompt):
                full_response += chunk
                yield chunk

            # Post-stream: save to memory
            conv_mem.add_turn("assistant", full_response)
            sess_mem.update_classification(label, classification.get("confidence", 0.0))
            sess_mem.update_risk_level(risk_level)

        except Exception as e:
            logger.error(f"[Orchestrator] Streaming pipeline error: {e}", exc_info=True)
            fallback = (
                "I'm here for you. Please share what's on your mind and I'll do my best to support you."
            )
            conv_mem.add_turn("assistant", fallback)
            yield fallback

    def end_session(self, session_id: str, user_id: str = "anonymous") -> bool:
        """
        End a session: persist summary to user memory and clean up.

        Args:
            session_id: Session to end.
            user_id: User identifier.

        Returns:
            True if session was found and ended, False otherwise.
        """
        if session_id not in self._sessions:
            return False

        session = self._sessions[session_id]
        sess_mem: SessionMemory = session["session"]
        conv_mem: ConversationMemory = session["conversation"]
        user_mem: UserMemory = session["user"]

        # Generate final summary if needed
        if not sess_mem.summary and conv_mem.turn_count > 2:
            summary_result = self._summarizer.run(
                conversation_history=conv_mem.get_history()
            )
            if summary_result.get("success"):
                sess_mem.set_summary(summary_result["summary"])

        # Persist to user profile
        snapshot = sess_mem.get_session_snapshot()
        user_mem.update_from_session(snapshot)

        # Remove session from registry
        del self._sessions[session_id]
        logger.info(f"Session ended and persisted | session={session_id}")
        return True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
orchestrator = Orchestrator()
