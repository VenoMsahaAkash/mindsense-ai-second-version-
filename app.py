"""
MindSense AI - Flask Application
====================================
Main Flask web application entry point.

Routes:
  GET  /              → Chat UI (index.html)
  POST /api/chat      → Process message through the pipeline
  GET  /api/stream    → Streaming response (SSE)
  GET  /api/history   → Get conversation history
  POST /api/reset     → Reset/clear session
  GET  /health        → Health check endpoint
  GET  /api/status    → System status (model readiness, index status)

Usage::

    python app.py
    # or with Flask CLI:
    flask run --host=0.0.0.0 --port=5000
"""

import json
import uuid
from pathlib import Path
from typing import Generator

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    session,
    stream_with_context,
)
from flask_cors import CORS

from config import settings
from agents.orchestrator import orchestrator
from model.faiss.index_manager import index_manager
from utils.logger import get_logger
from utils.helpers import generate_session_id, get_utc_timestamp
from utils.response_utils import build_api_response, build_error_response

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = settings.app.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = settings.app.MAX_CONTENT_LENGTH

# Enable CORS for development / frontend-backend separation
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_session_id() -> str:
    """
    Get or create a session ID for the current browser session.

    Returns:
        Session ID string.
    """
    if "session_id" not in session:
        session["session_id"] = generate_session_id()
    return session["session_id"]


def get_user_id() -> str:
    """
    Get or create a persistent user ID stored in the browser session.

    Returns:
        User ID string.
    """
    if "user_id" not in session:
        session["user_id"] = f"user_{uuid.uuid4().hex[:12]}"
    return session["user_id"]


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """
    Serve the main chat UI.

    Returns:
        Rendered index.html template.
    """
    session_id = get_session_id()
    logger.info(f"Chat UI requested | session={session_id[:8]}...")
    return render_template("index.html", app_name=settings.app.APP_NAME)


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Process a user message and return a complete JSON response.

    Request body (JSON)::

        {
            "message": "I feel anxious about work",
            "session_id": "optional_override"  // optional
        }

    Returns:
        JSON response from the complete agent pipeline.
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify(build_error_response("Invalid JSON body.", status_code=400)), 400

        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify(build_error_response("Message cannot be empty.", status_code=400)), 400

        # Session management
        session_id = data.get("session_id") or get_session_id()
        user_id = get_user_id()

        logger.info(
            f"[/api/chat] Received message | "
            f"session={session_id[:8]}... | "
            f"chars={len(user_message)}"
        )

        # Process through the full agent pipeline
        pipeline_result = orchestrator.process(
            user_message=user_message,
            session_id=session_id,
            user_id=user_id,
            stream=False,
        )

        # Build structured API response
        api_response = build_api_response(
            message=pipeline_result.get("response", ""),
            session_id=session_id,
            classification=pipeline_result.get("classification"),
            risk_level=pipeline_result.get("risk_level"),
            intent=pipeline_result.get("intent"),
            validation_score=pipeline_result.get("validation", {}).get("score"),
            sources=pipeline_result.get("sources", []),
        )

        # Enrich with additional pipeline data
        api_response["analysis"] = pipeline_result.get("analysis", {})
        api_response["is_crisis"] = pipeline_result.get("is_crisis", False)
        api_response["turn_count"] = pipeline_result.get("turn_count", 0)

        return jsonify(api_response)

    except Exception as e:
        logger.error(f"[/api/chat] Unhandled error: {e}", exc_info=True)
        session_id = get_session_id()
        return jsonify(build_error_response(str(e), session_id=session_id)), 500


@app.route("/api/stream", methods=["POST"])
def chat_stream():
    """
    Process a user message and stream the response token-by-token via SSE.

    Request body (JSON)::

        {"message": "I feel anxious", "session_id": "optional"}

    Returns:
        text/event-stream response with streamed text chunks.
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        session_id = data.get("session_id") or get_session_id()
        user_id = get_user_id()

        def generate_sse() -> Generator[str, None, None]:
            """Generate Server-Sent Events from the streaming pipeline."""
            try:
                for chunk in orchestrator.process_stream(
                    user_message=user_message,
                    session_id=session_id,
                    user_id=user_id,
                ):
                    if chunk:
                        # SSE format: data: <payload>\n\n
                        payload = json.dumps({"chunk": chunk, "done": False})
                        yield f"data: {payload}\n\n"

                # Signal completion
                done_payload = json.dumps({"chunk": "", "done": True, "session_id": session_id})
                yield f"data: {done_payload}\n\n"

            except Exception as e:
                logger.error(f"[/api/stream] Streaming error: {e}")
                error_payload = json.dumps({"error": str(e), "done": True})
                yield f"data: {error_payload}\n\n"

        return Response(
            stream_with_context(generate_sse()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error(f"[/api/stream] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Retrieve the conversation history for the current session.

    Returns:
        JSON with the conversation history array.
    """
    session_id = request.args.get("session_id") or get_session_id()
    history = orchestrator.get_session_history(session_id)

    return jsonify({
        "status": "success",
        "session_id": session_id,
        "history": history,
        "turn_count": len(history),
    })


@app.route("/api/reset", methods=["POST"])
def reset_session():
    """
    Reset/clear the current session and start fresh.

    Returns:
        JSON with the new session ID.
    """
    old_session_id = get_session_id()
    user_id = get_user_id()

    # End old session (persist summary to user memory)
    orchestrator.end_session(old_session_id, user_id)

    # Generate new session
    session["session_id"] = generate_session_id()
    new_session_id = session["session_id"]

    logger.info(f"Session reset | old={old_session_id[:8]} | new={new_session_id[:8]}")

    return jsonify({
        "status": "success",
        "message": "Session reset successfully.",
        "new_session_id": new_session_id,
    })


@app.route("/health")
def health_check():
    """
    Health check endpoint for Docker and load balancers.

    Returns:
        JSON health status.
    """
    return jsonify({
        "status": "healthy",
        "app": settings.app.APP_NAME,
        "version": settings.app.APP_VERSION,
        "timestamp": get_utc_timestamp(),
    })


@app.route("/api/status")
def system_status():
    """
    Return system status including model and index readiness.

    Returns:
        JSON with component status flags.
    """
    return jsonify({
        "status": "ok",
        "timestamp": get_utc_timestamp(),
        "components": {
            "faiss_index": {
                "ready": index_manager.is_ready,
                "vectors": index_manager.size,
            },
            "gemini_api": {
                "configured": bool(settings.gemini.API_KEY),
                "model": settings.gemini.MODEL_NAME,
            },
        },
    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found", "code": 404}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "code": 405}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error", "code": 500}), 500


# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------

def initialize_app() -> None:
    """
    Run startup tasks:
      - Validate configuration
      - Pre-load the FAISS index (non-blocking; warns if not built yet)
      - Warm up the embedding model
    """
    logger.info("=" * 55)
    logger.info(f"  {settings.app.APP_NAME} v{settings.app.APP_VERSION}")
    logger.info("=" * 55)

    # Validate config (raises if GEMINI_API_KEY is missing)
    try:
        settings.validate()
    except ValueError as e:
        logger.error(str(e))

    # Lazy load notice for memory optimization (512MB RAM constraint)
    logger.info("RAG FAISS index and Embedding models will load lazily on first user chat request.")

    logger.info(f"Flask server ready on {settings.app.HOST}:{settings.app.PORT}")
    logger.info("=" * 55)


if __name__ == "__main__":
    initialize_app()
    app.run(
        host=settings.app.HOST,
        port=settings.app.PORT,
        debug=settings.app.DEBUG,
        threaded=True,
        use_reloader=False,  # Disable reloader to avoid double-init
    )
