"""
MindSense AI - PyTorch Hybrid Classifier Loader (Trained Model Integration)
=============================================================================
Loads the trained DistilBERT + BiLSTM + Attention PyTorch model state dict from
hybrid_full_model.pth and hybrid_tokenizer.

Provides:
  - 7-Class Mental Health Prediction:
    ["Anxiety", "Bipolar", "Depression", "Normal", "Personality disorder", "Stress", "Suicidal"]
  - Token-level LIME / Attention Feature Attribution for research explainability.

Author: MindSense AI Team
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import settings
from model.classifier.model_def import HybridMentalHealthModel, MODEL_NAME, MAX_LEN
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


class PyTorchClassifierLoader:
    """
    Inference loader for the trained PyTorch Hybrid (DistilBERT + BiLSTM + Attention) Classifier.
    Extracts class probabilities and token-level feature attributions for LIME/Attention explainability.
    """

    def __init__(self) -> None:
        self.model_dir: Path = settings.classifier.MODEL_DIR
        self.labels: List[str] = DEFAULT_LABELS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[HybridMentalHealthModel] = None
        self._tokenizer: Optional[Any] = None
        self._loaded: bool = False

    def load(self) -> None:
        """Load tokenizer and PyTorch state dict into memory."""
        if self._loaded:
            return

        logger.info(f"Loading trained PyTorch Hybrid Model on device: {self.device}")

        try:
            # 1. Load Labels if present
            labels_path = self.model_dir / "mental_health_labels.json"
            if labels_path.exists():
                with open(labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
                logger.info(f"Loaded labels: {self.labels}")

            # 2. Load Tokenizer
            tokenizer_dir = self.model_dir / "hybrid_tokenizer"
            if tokenizer_dir.exists():
                logger.info(f"Loading tokenizer from {tokenizer_dir}")
                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
            else:
                logger.info(f"Loading base tokenizer {MODEL_NAME}")
                self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            # 3. Instantiate Model
            self._model = HybridMentalHealthModel(num_classes=len(self.labels))

            # Set single thread to prevent PyTorch thread RAM amplification on 512MB limit
            torch.set_num_threads(1)

            # 4. Load weights from hybrid_full_model.pth
            weights_path = self.model_dir / "hybrid_full_model.pth"
            if weights_path.exists():
                logger.info(f"Loading weights from {weights_path.name}")
                checkpoint = torch.load(str(weights_path), map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                self._model.load_state_dict(state_dict, strict=True)
                del checkpoint, state_dict
                import gc; gc.collect()
                logger.info("Loaded trained PyTorch weights into HybridMentalHealthModel successfully!")
            else:
                logger.warning(
                    f"Weights file {weights_path.name} not found in {self.model_dir}."
                )

            self._model.to(self.device)
            self._model.eval()
            self._loaded = True

        except Exception as e:
            logger.error(f"Failed to load PyTorch classifier: {e}", exc_info=True)
            self._loaded = False

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference using DistilBERT + BiLSTM + Attention architecture.

        Args:
            text: User statement.

        Returns:
            Dict containing predicted label, confidence, per-class probabilities,
            and token-level attention/LIME feature explanations.
        """
        if not self._loaded:
            self.load()

        cleaned = clean_text(text, lowercase=True)
        if not cleaned:
            cleaned = text.lower().strip() or "normal"

        if not self._loaded or self._model is None or self._tokenizer is None:
            return self._fallback_prediction(text)

        try:
            # Tokenize input
            encoding = self._tokenizer(
                cleaned,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                logits, attn_weights = self._model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_label = self.labels[pred_idx] if pred_idx < len(self.labels) else "Normal"

            all_scores = {
                label: float(round(p, 4))
                for label, p in zip(self.labels, probs)
            }

            # Extract token attention highlights for LIME/Explainability
            explainable_words = self._extract_attention_explanation(
                cleaned, encoding, attn_weights
            )

            return {
                "label": pred_label,
                "confidence": round(confidence, 4),
                "all_scores": all_scores,
                "explainable_words": explainable_words,
                "model_architecture": "DistilBERT + BiLSTM + Attention (Trained PyTorch Model)",
                "backend": "pytorch",
            }

        except Exception as e:
            logger.error(f"Inference error: {e}. Falling back.")
            return self._fallback_prediction(text)

    def _extract_attention_explanation(
        self, text: str, encoding: Any, attn_weights: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Extract top feature tokens using model's Attention layer weights mapped to clean words."""
        try:
            tokens = self._tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            weights = attn_weights[0].squeeze().cpu().tolist()

            stopwords = {
                "the", "and", "is", "in", "to", "of", "it", "that", "you", "for",
                "on", "are", "with", "as", "this", "was", "at", "be", "have", "not",
                "but", "had", "they", "from", "by", "she", "or", "an", "were", "my",
                "which", "one", "would", "all", "will", "there", "their", "what"
            }

            word_weights: Dict[str, float] = {}
            for token, weight in zip(tokens, weights):
                clean_tok = (
                    token.replace("##", "")
                    .replace("[CLS]", "")
                    .replace("[SEP]", "")
                    .replace("[PAD]", "")
                    .strip()
                    .lower()
                )
                if clean_tok and len(clean_tok) > 2 and clean_tok.isalpha() and clean_tok not in stopwords:
                    if clean_tok not in word_weights or weight > word_weights[clean_tok]:
                        word_weights[clean_tok] = float(weight)

            sorted_words = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)
            return [{"word": w, "score": float(round(s, 4))} for w, s in sorted_words[:5]]
        except Exception as e:
            logger.warning(f"Error extracting attention tokens: {e}")
            words = [w.strip(".,!?").lower() for w in text.split() if len(w) > 2 and w.isalpha()]
            return [{"word": w, "score": 0.2} for w in list(dict.fromkeys(words))[:5]]

    def _fallback_prediction(self, text: str) -> Dict[str, Any]:
        """Rule-based fallback for mental health category detection."""
        t_lower = text.lower()
        scores = {l: 0.05 for l in self.labels}

        if any(
            w in t_lower
            for w in [
                "anxious",
                "panic",
                "worry",
                "fear",
                "nervous",
                "scared",
                "heart racing",
            ]
        ):
            scores["Anxiety"] = 0.88
        elif any(
            w in t_lower
            for w in [
                "depressed",
                "hopeless",
                "worthless",
                "crying",
                "empty",
                "sad",
                "unmotivated",
            ]
        ):
            scores["Depression"] = 0.85
        elif any(
            w in t_lower
            for w in [
                "stress",
                "overwhelmed",
                "pressure",
                "exhausted",
                "burnout",
                "deadline",
            ]
        ):
            scores["Stress"] = 0.78
        elif any(w in t_lower for w in ["swing", "manic", "high and low", "bipolar"]):
            scores["Bipolar"] = 0.75
        else:
            scores["Normal"] = 0.90

        best_label = max(scores, key=lambda k: scores[k])
        words = [w.strip(".,!?") for w in t_lower.split() if len(w) > 3][:4]
        explainable = [{"word": w, "score": 0.25} for w in words]

        return {
            "label": best_label,
            "confidence": round(scores[best_label], 4),
            "all_scores": {k: round(v, 4) for k, v in scores.items()},
            "explainable_words": explainable,
            "model_architecture": "DistilBERT + BiLSTM + Attention (Fallback)",
            "backend": "rule_based",
        }


# Singleton instance
classifier_loader = PyTorchClassifierLoader()
