"""
MindSense AI - User Memory
============================
Persistent cross-session user profile storage using JSON files.

Stores user preferences, recurring concerns, coping strategy preferences,
and session history summaries that persist between restarts.

JSON files are stored in memory/user_profiles/<user_id>.json

Usage::

    from memory.user_memory import UserMemory
    user_mem = UserMemory(user_id="user_abc")
    user_mem.update_from_session(session_snapshot)
    profile = user_mem.get_profile()
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from utils.logger import get_logger
from utils.helpers import safe_json_load, safe_json_save, get_utc_timestamp

logger = get_logger(__name__)


# Default empty user profile structure
def _default_profile(user_id: str) -> Dict[str, Any]:
    """Create a default empty user profile."""
    return {
        "user_id": user_id,
        "created_at": get_utc_timestamp(),
        "last_seen": get_utc_timestamp(),
        "session_count": 0,
        "preferred_name": None,
        "concerns": [],                  # Recurring concern keywords
        "coping_preferences": [],        # Preferred coping strategies
        "dominant_condition": None,      # Most frequently detected condition
        "condition_history": {},         # {label: count} across sessions
        "crisis_history_count": 0,       # Number of crisis events
        "session_summaries": [],         # Last N session summaries
        "total_turn_count": 0,
    }


class UserMemory:
    """
    Persistent cross-session user profile manager.

    Reads from and writes to ``memory/user_profiles/<user_id>.json``.
    Aggregates data from completed sessions to build a longitudinal
    user understanding.

    Attributes:
        user_id (str): Unique user identifier.
        profile_path (Path): Path to the JSON profile file.
    """

    MAX_SESSION_SUMMARIES: int = 10  # Keep last 10 session summaries

    def __init__(self, user_id: str = "anonymous") -> None:
        """
        Initialize user memory for the given user ID.

        Args:
            user_id: Unique user identifier. Defaults to "anonymous".
        """
        self.user_id: str = user_id
        self.profiles_dir: Path = settings.memory.USER_PROFILES_DIR
        self.profile_path: Path = self.profiles_dir / f"{user_id}.json"
        self._profile: Optional[Dict[str, Any]] = None

    def _load(self) -> None:
        """Load the user profile from disk, or create a default one."""
        if self._profile is not None:
            return

        loaded = safe_json_load(self.profile_path)
        if loaded:
            self._profile = loaded
            logger.debug(f"User profile loaded | user={self.user_id}")
        else:
            self._profile = _default_profile(self.user_id)
            logger.debug(f"New user profile created | user={self.user_id}")

    def _save(self) -> None:
        """Persist the user profile to disk."""
        if self._profile is None:
            return
        saved = safe_json_save(self._profile, self.profile_path)
        if saved:
            logger.debug(f"User profile saved | user={self.user_id}")
        else:
            logger.error(f"Failed to save user profile | user={self.user_id}")

    def get_profile(self) -> Dict[str, Any]:
        """
        Get the full user profile dictionary.

        Returns:
            User profile dict with all fields.
        """
        self._load()
        return dict(self._profile)  # type: ignore

    def set_preferred_name(self, name: str) -> None:
        """
        Set the user's preferred name.

        Args:
            name: Display name for personalized greetings.
        """
        self._load()
        self._profile["preferred_name"] = name.strip()  # type: ignore
        self._save()

    def add_concern(self, concern: str) -> None:
        """
        Add a recurring concern keyword to the user's profile.
        Deduplicates and limits to 20 concerns.

        Args:
            concern: Concern keyword or phrase.
        """
        self._load()
        concerns: List[str] = self._profile.get("concerns", [])  # type: ignore
        concern = concern.strip().lower()
        if concern and concern not in concerns:
            concerns.append(concern)
            if len(concerns) > 20:
                concerns = concerns[-20:]  # Keep most recent
            self._profile["concerns"] = concerns  # type: ignore
            self._save()

    def add_coping_preference(self, strategy: str) -> None:
        """
        Record a coping strategy the user has responded positively to.

        Args:
            strategy: Coping strategy name (e.g., "box breathing", "journaling").
        """
        self._load()
        prefs: List[str] = self._profile.get("coping_preferences", [])  # type: ignore
        strategy = strategy.strip()
        if strategy and strategy not in prefs:
            prefs.append(strategy)
            self._profile["coping_preferences"] = prefs[-10:]  # type: ignore
            self._save()

    def update_from_session(self, session_snapshot: Dict[str, Any]) -> None:
        """
        Update the user profile with data from a completed session.

        Args:
            session_snapshot: Session state dict from ``SessionMemory.get_session_snapshot()``.
        """
        self._load()
        profile = self._profile  # type: ignore

        # Update session count and last seen
        profile["session_count"] = profile.get("session_count", 0) + 1
        profile["last_seen"] = get_utc_timestamp()
        profile["total_turn_count"] = (
            profile.get("total_turn_count", 0) + session_snapshot.get("turn_count", 0)
        )

        # Track condition history
        label = session_snapshot.get("dominant_label")
        if label and label != "Normal":
            cond_hist = profile.get("condition_history", {})
            cond_hist[label] = cond_hist.get(label, 0) + 1
            profile["condition_history"] = cond_hist

            # Update dominant condition (most frequent)
            profile["dominant_condition"] = max(
                cond_hist, key=lambda k: cond_hist[k]
            )

        # Track crisis events
        if session_snapshot.get("crisis_triggered"):
            profile["crisis_history_count"] = profile.get("crisis_history_count", 0) + 1

        # Store session summary
        if session_snapshot.get("summary"):
            summaries: List[str] = profile.get("session_summaries", [])
            summaries.append(
                f"[{get_utc_timestamp()}] {session_snapshot['summary']}"
            )
            profile["session_summaries"] = summaries[-self.MAX_SESSION_SUMMARIES:]

        # Accumulate themes
        for theme in session_snapshot.get("themes", []):
            self.add_concern(theme)

        self._profile = profile
        self._save()
        logger.info(f"User profile updated from session | user={self.user_id}")

    def get_session_context(self) -> Dict[str, Any]:
        """
        Get a compact profile summary for prompt injection.

        Returns:
            Dict with the most relevant fields for the LLM context.
        """
        self._load()
        profile = self._profile  # type: ignore
        return {
            "preferred_name": profile.get("preferred_name"),
            "concerns": profile.get("concerns", [])[-5:],  # Most recent 5
            "coping_preferences": profile.get("coping_preferences", [])[-3:],
            "dominant_condition": profile.get("dominant_condition"),
            "session_count": profile.get("session_count", 0),
            "recent_summaries": profile.get("session_summaries", [])[-2:],  # Last 2 sessions
        }

    @property
    def is_returning_user(self) -> bool:
        """Whether this user has had previous sessions."""
        self._load()
        return self._profile.get("session_count", 0) > 0  # type: ignore
