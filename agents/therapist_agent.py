"""
MindSense AI - Therapist Agent
================================
The core response generation agent. Uses the assembled prompt
from PromptBuilder to query Gemini 2.5 Flash and generate
an empathetic, grounded therapeutic response.

Supports both streaming and non-streaming modes.

Usage::

    from agents.therapist_agent import TherapistAgent
    agent = TherapistAgent()
    result = agent.run(
        prompt="[full assembled prompt]",
        stream=False
    )
"""

from typing import Any, Dict, Generator, Optional

from model.llm.gemini_client import gemini_client, GeminiAPIError
from utils.logger import get_logger
from utils.helpers import timer
from utils.response_utils import format_response_markdown

logger = get_logger(__name__)


class TherapistAgent:
    """
    Core LLM response generation agent.

    Sends the assembled prompt to Google Gemini 2.5 Flash and
    returns a formatted empathetic response.

    Provides both blocking (``run()``) and streaming (``run_stream()``) modes.
    The Orchestrator selects the appropriate mode based on the request.
    """

    def __init__(self) -> None:
        logger.debug("TherapistAgent initialized.")

    @timer
    def run(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a therapeutic response from the assembled prompt.

        Args:
            prompt: The complete multi-section prompt string.
            temperature: Optional override for Gemini temperature.
            max_tokens: Optional override for max output tokens.

        Returns:
            Dict containing:
              - ``response`` (str): Generated response text.
              - ``markdown`` (str): Markdown-formatted response.
              - ``success`` (bool): Whether generation succeeded.
              - ``error`` (str|None): Error message if failed.
              - ``agent`` (str): Agent identifier.

        Example::

            {
                "response": "It sounds like you're carrying a heavy weight...",
                "markdown": "It sounds like...",
                "success": True,
                "error": None,
                "agent": "TherapistAgent"
            }
        """
        if not prompt or not prompt.strip():
            return self._error_result("Empty prompt provided.")

        try:
            logger.debug(
                f"[TherapistAgent] Generating response | prompt_chars={len(prompt)}"
            )

            raw_response = gemini_client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not raw_response or not raw_response.strip():
                return self._error_result("Empty response from Gemini.")

            # Format the response
            formatted = format_response_markdown(raw_response)

            logger.info(
                f"[TherapistAgent] Response generated | "
                f"chars={len(formatted)}"
            )

            return {
                "response": formatted,
                "markdown": formatted,
                "success": True,
                "error": None,
                "agent": "TherapistAgent",
            }

        except GeminiAPIError as e:
            logger.error(f"[TherapistAgent] Gemini API error: {e}")
            return self._error_result(str(e), prompt=prompt)

        except Exception as e:
            logger.error(f"[TherapistAgent] Unexpected error: {e}")
            return self._error_result(str(e), prompt=prompt)

    def run_stream(self, prompt: str) -> Generator[str, None, None]:
        try:
            logger.debug("[TherapistAgent] Starting streaming generation.")
            for chunk in gemini_client.generate_stream(prompt):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"[TherapistAgent] Streaming error: {e}")
            yield self._fallback_response(prompt)

    def _error_result(self, error_msg: str, prompt: str = "") -> Dict[str, Any]:
        """Return a structured error result with a safe fallback response."""
        fallback = self._fallback_response(prompt)
        return {
            "response": fallback,
            "markdown": fallback,
            "success": True,
            "error": error_msg,
            "agent": "TherapistAgent",
        }

    def _fallback_response(self, prompt: str = "") -> str:
        """Return a structured, empathetic coping action plan matching the PyTorch model's predicted category."""
        return gemini_client._clinical_fallback(prompt)
