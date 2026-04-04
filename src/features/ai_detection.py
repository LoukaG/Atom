"""
RoBERTa-based AI Detection Module

Uses a fine-tuned RoBERTa model for fast and accurate AI text detection.
Model: roberta-base-openai-detector (OpenAI's GPT-2 output detector)

This is much lighter and faster than Binoculars or Fast-DetectGPT:
- Size: ~125MB (vs 12-24GB for larger models)
- Speed: 10-100x faster
- Memory: Fits easily in any GPU
- Accuracy: Trained by OpenAI to detect GPT-2 outputs
"""

import re
from html import unescape
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global classifier components
_model = None
_tokenizer = None


def _clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _log_gpu_memory(prefix=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(
            f"{prefix}GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        )


def _get_model():
    """Get or create the global AI text classifier."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: CUDA not available, running AI detection on CPU (slower)")
        else:
            print(
                f"Initializing AI detection model (CUDA device: {torch.cuda.get_device_name(0)})"
            )
            print("  - Model: roberta-base-openai-detector")
            print("  - Size: ~125MB")

        try:
            _tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
            _model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base-openai-detector"
            )
            _model.to(device)
            _model.eval()

            if device == "cuda":
                _clear_gpu_cache()
                _log_gpu_memory("After model loading: ")

        except Exception as e:
            print(f"ERROR: Failed to initialize AI detection model: {e}")
            _clear_gpu_cache()
            raise

    return _model, _tokenizer


def get_ai_score(text):
    """
    Get AI detection score for a single text.

    Returns a float between 0 and 1:
    - 0.0-0.3: Likely human-written
    - 0.3-0.7: Uncertain
    - 0.7-1.0: Likely AI-generated

    Returns 0.0 for empty/invalid text or on error.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    try:
        model, tokenizer = _get_model()
        device = next(model.parameters()).device

        # Tokenize (RoBERTa max is 512 tokens)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Get probability of "Fake" class (AI-generated)
            # Model has 2 classes: Real (0) and Fake (1)
            ai_prob = probs[0][1].item()

        # Clear GPU cache after computation
        _clear_gpu_cache()

        return float(ai_prob)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(
                f"Warning: CUDA OOM error during AI detection. Clearing cache and skipping text."
            )
            _clear_gpu_cache()
            return 0.0
        else:
            print(f"Warning: AI detection failed: {e}")
            _clear_gpu_cache()
            return 0.0
    except Exception as e:
        print(f"Warning: AI detection failed: {e}")
        _clear_gpu_cache()
        return 0.0


def average_ai_score(texts, batch_size=16):
    """
    Calculate average AI detection score across multiple texts.
    Processes texts in batches for optimal GPU utilization.

    Args:
        texts: List of text strings to score
        batch_size: Number of texts to process at once

    Returns: Average AI score (0-1 range, higher = more likely AI-generated)
    """
    if not texts:
        return 0.0

    # Filter out empty/very short texts
    valid_texts = [t for t in texts if t and len(t.strip()) >= 10]

    if not valid_texts:
        return 0.0

    scores = []

    _log_gpu_memory("Before AI scoring - ")

    try:
        model, tokenizer = _get_model()
        device = next(model.parameters()).device

        # Process in batches
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i : i + batch_size]

            try:
                # Tokenize batch
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(device)

                # Get predictions for batch
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)

                    # Get probability of "Fake" class for each text
                    ai_probs = probs[:, 1].cpu().numpy()
                    scores.extend(ai_probs.tolist())

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"Warning: OOM error in batch {i // batch_size + 1}. Processing individually..."
                    )
                    _clear_gpu_cache()

                    # Fallback: process one by one
                    for text in batch:
                        score = get_ai_score(text)
                        if score != 0.0:
                            scores.append(score)
                else:
                    raise

            if (i + batch_size) % (batch_size * 4) == 0:  # Log every 4 batches
                _log_gpu_memory(f"After batch {i // batch_size + 1} - ")

    except Exception as e:
        print(f"Warning: Batch AI detection failed: {e}")
        _clear_gpu_cache()

        # Fallback: try processing individually
        for text in valid_texts:
            try:
                score = get_ai_score(text)
                if score != 0.0:
                    scores.append(score)
            except:
                continue

    _log_gpu_memory("After AI scoring - ")

    return float(np.mean(scores)) if scores else 0.0
