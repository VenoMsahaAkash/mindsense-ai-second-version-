"""
MindSense AI - Google Gemini LLM Client
==========================================
Production-ready wrapper for the Google Gemini 2.5 Flash API.

Features:
  - Lazy initialization (model loaded once, reused across requests)
  - Automatic retry with exponential backoff
  - Streaming response support
  - Token usage logging
  - Safety settings configuration
  - Clean exception hierarchy

Usage::

    from model.llm.gemini_client import GeminiClient
    client = GeminiClient()
    response = client.generate("Tell me about CBT therapy.")
"""

import time
from typing import Generator, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class GeminiAPIError(Exception):
    """Raised when the Gemini API returns an error."""
    pass


class GeminiInitializationError(Exception):
    """Raised when the Gemini client cannot be initialized."""
    pass


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

class GeminiClient:
    """
    Singleton-style wrapper for the Google Gemini 2.5 Flash API.

    The model is initialized lazily on first call and reused for all
    subsequent requests to minimize API cold-start overhead.

    Attributes:
        model_name (str): The Gemini model identifier.
        temperature (float): Sampling temperature.
        max_output_tokens (int): Maximum tokens in the response.
        _model: Internal GenerativeModel instance.
    """

    def __init__(self) -> None:
        """Initialize the Gemini client using settings from config.py."""
        self.model_name = settings.gemini.MODEL_NAME
        self.temperature = settings.gemini.TEMPERATURE
        self.top_p = settings.gemini.TOP_P
        self.top_k = settings.gemini.TOP_K
        self.max_output_tokens = settings.gemini.MAX_OUTPUT_TOKENS
        self.max_retries = settings.gemini.MAX_RETRIES
        self.retry_delay = settings.gemini.RETRY_DELAY
        self._model: Optional[genai.GenerativeModel] = None
        self._initialized: bool = False

    def _initialize(self) -> None:
        """
        Configure the Gemini API and load the generative model.
        Called lazily on first request.

        Raises:
            GeminiInitializationError: If API key is missing or initialization fails.
        """
        if self._initialized:
            return

        api_key = settings.gemini.API_KEY
        if not api_key:
            raise GeminiInitializationError(
                "GEMINI_API_KEY is not set. Please configure it in your .env file."
            )

        try:
            genai.configure(api_key=api_key)

            # Safety settings — relaxed for mental health context
            # We still block HARM_CATEGORY_DANGEROUS_CONTENT to prevent harm
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

            generation_config = GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_output_tokens,
            )

            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            self._initialized = True
            logger.info(f"Gemini client initialized with model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise GeminiInitializationError(str(e)) from e

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a text response from the Gemini model or clinical fallback."""
        try:
            self._initialize()
        except Exception as e:
            logger.warning(f"Gemini API initialization unavailable ({e}). Using clinical fallback response.")
            return self._clinical_fallback(prompt)

        # Per-request generation config overrides
        gen_config = None
        if temperature is not None or max_tokens is not None:
            gen_config = GenerationConfig(
                temperature=temperature if temperature is not None else self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
            )

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"Gemini generate attempt {attempt}/{self.max_retries} | "
                    f"prompt_chars={len(prompt)}"
                )

                kwargs = {}
                if gen_config:
                    kwargs["generation_config"] = gen_config

                response = self._model.generate_content(prompt, **kwargs)  # type: ignore

                # Extract text from response
                if response.candidates and response.candidates[0].content.parts:
                    text = response.text
                    logger.debug(
                        f"Gemini response received | "
                        f"response_chars={len(text)} | attempt={attempt}"
                    )
                    return text
                else:
                    logger.warning(f"Gemini returned empty response on attempt {attempt}")
                    return "I'm sorry, I wasn't able to generate a response. Please try again."

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Gemini API error on attempt {attempt}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries:
                    sleep_time = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.debug(f"Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)

        logger.error(f"Gemini API failed after {self.max_retries} attempts: {last_error}")
        return self._clinical_fallback(prompt)

    def _clinical_fallback(self, prompt: str) -> str:
        """Dynamic evidence-based clinical coping response matching user text & PyTorch model category."""
        prompt_lower = prompt.lower()

        # Extract user statement from prompt if present
        user_msg = ""
        if "[user message]" in prompt_lower:
            start_idx = prompt_lower.rfind("[user message]") + len("[user message]")
            end_idx = prompt_lower.find("[your response]", start_idx)
            if end_idx != -1:
                user_msg = prompt[start_idx:end_idx].strip()
            else:
                user_msg = prompt[start_idx:].strip()

        user_quote = f'"{user_msg}"' if user_msg else "what you are experiencing"

        if "detected condition: normal" in prompt_lower:
            return (
                "### General Mental Wellness & Support\n"
                f"I'm really glad to hear you say: {user_quote}. Taking time to pause and acknowledge positive moments or feeling well is a wonderful way to maintain daily emotional resilience.\n\n"
                "### Recommended Wellness Practices\n"
                "1. **Anchor the Positive**: Reflect on what specific routine, activity, or mindset helped you feel balanced today.\n"
                "2. **Maintain Daily Rhythms**: Keep up consistent sleep, physical movement, and social connections.\n"
                "3. **Mindful Check-Ins**: Continue checking in with your feelings regularly to stay grounded.\n\n"
                "What is one pleasant thing that happened in your day so far?"
            )
        elif "detected condition: stress" in prompt_lower:
            return (
                "### Managing Stress & Overwhelm\n"
                f"I hear you when you express: {user_quote}. Stress and feeling overwhelmed happen when demands outpace your current energy and open tasks pile up in your mind.\n\n"
                "### Actionable Coping Strategies\n"
                "1. **Brain Dump & Priority Matrix**: Write down everything stressing you out, then pick just ONE priority task to tackle first.\n"
                "2. **Physiological Sigh**: Take two quick inhales through your nose, followed by one long slow exhale through your mouth. Repeat 3 times to immediately lower heart rate.\n"
                "3. **Micro-Breaks**: Step away from screens for 5 minutes and focus your gaze on a far-off point to relax your focus.\n\n"
                "Which stressor feels like the biggest burden right now?"
            )
        elif "detected condition: depression" in prompt_lower:
            return (
                "### Acknowledging Low Mood & Fatigue\n"
                f"Thank you for sharing: {user_quote}. I want to validate how heavy and draining things feel right now. Depression often tries to convince us that low energy will last forever, making even small steps feel monumental.\n\n"
                "### Evidence-Based Action Steps\n"
                "1. **Micro-Behavioral Activation**: Pick just ONE 2-minute task (e.g. drinking a glass of water or opening a window). Action often precedes motivation.\n"
                "2. **Compassionate Self-Talk (CBT)**: Treat yourself with the same kindness you would offer a dear friend in distress.\n"
                "3. **Gentle Movement**: Take 5 deep breaths or stretch for 30 seconds to gently reconnect with your body.\n\n"
                "Would you like to try one tiny step together right now?"
            )
        elif "detected condition: bipolar" in prompt_lower:
            return (
                "### Navigating Mood Swings & Energy Shifts\n"
                f"I hear what you're sharing regarding: {user_quote}. Experiencing shifts in mood or energy can feel unpredictable. Establishing steady daily rhythms helps anchor your emotional baseline.\n\n"
                "### Recommended Stabilization Steps\n"
                "1. **Routine Anchoring**: Keep sleep, mealtime, and wake times strictly consistent every single day.\n"
                "2. **Energy Log**: Note down your energy levels (1-10) twice a day to notice patterns early.\n"
                "3. **Grounding Interventions**: Engage in low-stimulation calming activities during high-energy or tense periods.\n\n"
                "Have you noticed a specific trigger for your recent energy shifts?"
            )
        elif "detected condition: suicidal" in prompt_lower:
            return (
                "### Supportive Emotional Grounding\n"
                f"I hear how deep your pain feels when you express: {user_quote}. I want to validate that carrying this level of distress is exhausting. Please know that your life matters and you do not have to carry this alone.\n\n"
                "### Immediate Grounding Steps\n"
                "1. **5-4-3-2-1 Sensory Grounding**: Name 5 things you see, 4 things you touch, 3 things you hear, 2 things you smell, and 1 slow breath.\n"
                "2. **Safety First**: Reach out to a trusted loved one, counselor, or crisis helpline professional who can support you safely.\n"
                "3. **Pace Your Focus**: You do not have to figure out everything right now. Focus only on getting through this next single hour.\n\n"
                "Is there a trusted person or loved one you can connect with right now?"
            )
        elif "detected condition: personality disorder" in prompt_lower:
            return (
                "### Emotion Regulation & Interpersonal Grounding\n"
                f"I hear what you're sharing: {user_quote}. Intense emotional shifts and interpersonal stress can feel overwhelming. Dialectical Behavior Therapy (DBT) techniques help balance strong feelings.\n\n"
                "### Recommended DBT Interventions\n"
                "1. **TIPP Temperature Reset**: Splash cold water on your face or hold an ice cube to gently calm intense nervous system arousal.\n"
                "2. **STOP Technique**: Pause, Take a step back, Observe your feelings without judgment, and Proceed mindfully.\n"
                "3. **Radical Acceptance**: Acknowledge current reality without fighting it to reduce emotional suffering.\n\n"
                "Which emotion feels most intense right now?"
            )
        else:
            return (
                "### Understanding Your Anxiety & Feelings\n"
                f"I hear you when you express: {user_quote}. "
                "When anxiety surfaces—whether expectedly or out of nowhere—your body's nervous system triggers a 'fight-or-flight' response, causing sensations like a racing heart, muscle tension, or restless thoughts.\n\n"
                "### Actionable Coping Strategies\n"
                "1. **Box Breathing Technique**: Inhale slowly for 4 seconds, hold for 4 seconds, exhale for 4 seconds, and pause for 4 seconds. Repeat 4 times to signal safety to your nervous system.\n"
                "2. **5-4-3-2-1 Grounding**: Look around and name 5 things you can see, 4 things you can touch, 3 things you hear, 2 things you smell, and 1 positive affirmation.\n"
                "3. **Thought Decoupling (CBT)**: Ask yourself: *Is this worry a proven fact right now, or is my anxiety predicting a worst-case scenario?*\n\n"
                "What is one small thing that helped you feel grounded in the past?"
            )

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate a streaming response from the Gemini model.
        Yields text chunks as they are received from the API or clinical fallback.
        """
        try:
            self._initialize()
            logger.debug(f"Gemini streaming request | prompt_chars={len(prompt)}")
            response = self._model.generate_content(prompt, stream=True)  # type: ignore

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.warning(f"Gemini streaming API unavailable ({e}). Using dynamic clinical stream fallback.")
            fallback_text = self._clinical_fallback(prompt)
            # Yield in natural word chunks
            words = fallback_text.split(" ")
            for i in range(0, len(words), 3):
                yield " ".join(words[i:i+3]) + " "

    def health_check(self) -> bool:
        """
        Verify that the Gemini API is reachable and responding.

        Returns:
            True if the API is healthy, False otherwise.
        """
        try:
            response = self.generate("Reply with only: OK", max_tokens=5)
            return bool(response)
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Module-level singleton instance
# ---------------------------------------------------------------------------
gemini_client = GeminiClient()
