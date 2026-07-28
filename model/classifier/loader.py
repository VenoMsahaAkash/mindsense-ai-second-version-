"""
MindSense AI - Lightweight Classifier Loader
==============================================
Provides zero-dependency mental health classification using an enhanced
rule-based lexicon covering all 7 categories.

On the free-tier deployment (512MB RAM limit), the PyTorch/DistilBERT model
(~370MB) is intentionally NOT loaded. Classification uses an expanded
keyword lexicon that matches the fallback logic with high coverage.

For local development with the full PyTorch model, set:
  CLASSIFIER_BACKEND=pytorch
in your .env file.

Author: MindSense AI Team
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from utils.logger import get_logger
from utils.preprocessing import clean_text

logger = get_logger(__name__)

DEFAULT_LABELS = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal",
]

# Expanded keyword lexicons for enhanced rule-based classification
_LEXICONS: Dict[str, List[str]] = {
    "Anxiety": [
        "anxious", "anxiety", "panic", "panic attack", "worry", "worrying", "worried",
        "fear", "nervous", "nervousness", "scared", "heart racing", "overthinking",
        "restless", "tense", "apprehensive", "dread", "phobia", "uneasy", "jittery",
        "can't breathe", "shaking", "trembling", "sweating", "social anxiety",
        "catastrophe", "what if", "doom", "terror", "helpless", "on edge",
    ],
    "Depression": [
        "depressed", "depression", "hopeless", "worthless", "meaningless", "empty",
        "numb", "sad", "sadness", "crying", "tears", "unmotivated", "no motivation",
        "can't get up", "exhausted", "drained", "nothing matters", "no point",
        "lonely", "alone", "isolated", "giving up", "lost interest", "anhedonia",
        "not sleeping", "sleeping too much", "can't eat", "no appetite", "low mood",
        "miserable", "heartbroken", "broken", "crushed", "devastated", "grief",
    ],
    "Stress": [
        "stress", "stressed", "overwhelmed", "pressure", "burnout", "burned out",
        "deadline", "workload", "too much", "can't cope", "overwhelm", "exhausted",
        "overworked", "tight schedule", "no time", "behind", "falling behind",
        "barely managing", "stretched thin", "juggling", "responsibilities",
        "demands", "performance pressure", "college stress", "exam stress",
    ],
    "Suicidal": [
        "suicidal", "suicide", "end my life", "end it all", "kill myself", "want to die",
        "don't want to live", "no reason to live", "better off dead", "can't go on",
        "self harm", "self-harm", "hurt myself", "cutting", "overdose",
        "not worth living", "world without me", "no future", "plan to die",
        "goodbyes", "saying goodbye", "final goodbye", "life insurance",
    ],
    "Bipolar": [
        "bipolar", "manic", "mania", "mood swing", "mood swings", "high and low",
        "up and down", "racing thoughts", "grandiosity", "no sleep but energetic",
        "euphoric", "impulsive", "reckless", "spending spree", "very fast",
        "episode", "depressive episode", "manic episode", "cycling",
    ],
    "Personality disorder": [
        "borderline", "bpd", "split", "identity", "unstable identity", "fear of abandonment",
        "emptiness", "impulsive behaviour", "unstable relationships", "self-image",
        "dissociation", "dissociate", "paranoia", "paranoid", "narcissistic",
        "antisocial", "no empathy", "manipulative", "volatile",
    ],
    "Normal": [
        "doing okay", "fine", "alright", "good day", "happy", "well", "positive",
        "feeling good", "great", "wonderful", "improving", "getting better", "hopeful",
        "motivated", "productive", "calm", "peaceful", "relaxed", "content",
    ],
}


class PyTorchClassifierLoader:
    """
    Mental health classifier.

    On Render free tier (512MB), uses an enhanced rule-based lexicon.
    To enable the full PyTorch/DistilBERT model locally, set:
        CLASSIFIER_BACKEND=pytorch
    in your .env file.
    """

    def __init__(self) -> None:
        self.model_dir: Path = settings.classifier.MODEL_DIR
        self.labels: List[str] = DEFAULT_LABELS
        # Determine backend: 'pytorch' only if explicitly requested AND torch is available
        self._backend: str = os.environ.get("CLASSIFIER_BACKEND", "rules")
        self._model = None
        self._tokenizer = None
        self._loaded: bool = False
        logger.info(f"ClassifierLoader backend: {self._backend}")

    def load(self) -> None:
        """Load model (only for pytorch backend; rules backend needs no loading)."""
        if self._loaded:
            return

        if self._backend == "pytorch":
            self._load_pytorch()
        else:
            # Rule-based: nothing to load, mark as ready immediately
            self._loaded = True
            logger.info("Rule-based classifier ready (no model weights needed).")

    def _load_pytorch(self) -> None:
        """Load the heavy PyTorch/DistilBERT model (local dev only)."""
        try:
            import json
            import torch
            import torch.nn.functional as F
            from transformers import AutoTokenizer
            from model.classifier.model_def import HybridMentalHealthModel, MODEL_NAME

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.set_num_threads(1)

            labels_path = self.model_dir / "mental_health_labels.json"
            if labels_path.exists():
                with open(labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)

            tokenizer_dir = self.model_dir / "hybrid_tokenizer"
            if tokenizer_dir.exists():
                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            self._model = HybridMentalHealthModel(num_classes=len(self.labels))
            weights_path = self.model_dir / "hybrid_full_model.pth"
            if weights_path.exists():
                checkpoint = torch.load(str(weights_path), map_location=device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                self._model.load_state_dict(state_dict, strict=True)
                del checkpoint, state_dict
                import gc; gc.collect()

            self._model.to(device)
            self._model.eval()
            self._loaded = True
            self._device = device
            logger.info("PyTorch DistilBERT classifier loaded.")

        except Exception as e:
            logger.error(f"Failed to load PyTorch classifier: {e}. Falling back to rules.")
            self._backend = "rules"
            self._loaded = True

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify a user message into one of 7 mental health categories.

        Uses the PyTorch model when CLASSIFIER_BACKEND=pytorch,
        otherwise uses the enhanced rule-based lexicon (default on Render).

        Args:
            text: User statement.

        Returns:
            Dict with label, confidence, all_scores, explainable_words.
        """
        if not self._loaded:
            self.load()

        if self._backend == "pytorch" and self._model is not None:
            return self._pytorch_predict(text)

        return self._rules_predict(text)

    def _pytorch_predict(self, text: str) -> Dict[str, Any]:
        """Run inference with the loaded PyTorch model."""
        try:
            import torch
            import torch.nn.functional as F
            import numpy as np

            cleaned = clean_text(text, lowercase=True) or text.lower().strip() or "normal"
            encoding = self._tokenizer(
                cleaned,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )

            device = self._device
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                logits, attn_weights = self._model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_label = self.labels[pred_idx] if pred_idx < len(self.labels) else "Normal"
            all_scores = {label: float(round(p, 4)) for label, p in zip(self.labels, probs)}

            # Token-level attention explanation
            try:
                tokens = self._tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
                weights = attn_weights[0].squeeze().cpu().tolist()
                word_weights: Dict[str, float] = {}
                stopwords = {"the", "and", "is", "in", "to", "of", "it", "that", "you", "for",
                             "on", "are", "with", "as", "this", "was", "at", "be", "have", "not"}
                for token, weight in zip(tokens, weights):
                    t = token.replace("##", "").replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip().lower()
                    if t and len(t) > 2 and t.isalpha() and t not in stopwords:
                        word_weights[t] = max(word_weights.get(t, 0.0), float(weight))
                explainable_words = [{"word": w, "score": float(round(s, 4))}
                                     for w, s in sorted(word_weights.items(), key=lambda x: x[1], reverse=True)[:5]]
            except Exception:
                explainable_words = []

            return {
                "label": pred_label,
                "confidence": round(confidence, 4),
                "all_scores": all_scores,
                "explainable_words": explainable_words,
                "model_architecture": "DistilBERT + BiLSTM + Attention (Trained PyTorch Model)",
                "backend": "pytorch",
            }

        except Exception as e:
            logger.error(f"PyTorch inference error: {e}. Falling back to rules.")
            return self._rules_predict(text)

    def _rules_predict(self, text: str) -> Dict[str, Any]:
        """
        Enhanced rule-based mental health classifier using keyword lexicons.
        Covers all 7 categories with multi-keyword scoring.
        """
        t_lower = text.lower()
        scores: Dict[str, float] = {label: 0.0 for label in self.labels}

        for label, keywords in _LEXICONS.items():
            match_count = sum(1 for kw in keywords if kw in t_lower)
            if match_count > 0:
                # Sigmoid-style scoring: more matches → higher confidence
                raw_score = min(0.97, 0.55 + (match_count * 0.12))
                if label in scores:
                    scores[label] = raw_score

        # If nothing matched, default to Normal
        if all(v == 0.0 for v in scores.values()):
            scores["Normal"] = 0.82
        else:
            # Ensure Normal has a small baseline
            if scores.get("Normal", 0.0) == 0.0:
                scores["Normal"] = 0.05

        # Suicidal always takes priority if matched
        if scores.get("Suicidal", 0.0) > 0:
            scores["Suicidal"] = min(0.97, scores["Suicidal"] + 0.15)

        best_label = max(scores, key=lambda k: scores[k])
        confidence = scores[best_label]

        # Normalize remaining scores
        total = sum(scores.values()) or 1.0
        all_scores = {k: round(v / total, 4) for k, v in scores.items()}
        all_scores[best_label] = round(confidence, 4)  # keep top score unnormalized for clarity

        # Explainable words: matched keywords from the winning category
        matched_kws = [kw for kw in _LEXICONS.get(best_label, []) if kw in t_lower]
        explainable = [{"word": kw, "score": round(confidence, 4)} for kw in matched_kws[:5]]
        if not explainable:
            explainable = [{"word": w.strip(".,!?"), "score": 0.20}
                           for w in t_lower.split() if len(w) > 3][:5]

        return {
            "label": best_label,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "explainable_words": explainable,
            "model_architecture": "Enhanced Rule-Based Lexicon Classifier",
            "backend": "rules",
            "high_risk": best_label == "Suicidal",
        }


# Singleton instance
classifier_loader = PyTorchClassifierLoader()
