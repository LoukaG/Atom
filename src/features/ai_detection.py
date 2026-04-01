import torch
import numpy as np
from binoculars import Binoculars

# Initialize Binoculars model with GPU support
_binoculars = None


def _get_binoculars():
    global _binoculars
    if _binoculars is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: CUDA not available, running Binoculars on CPU (will be slow)")
        else:
            print(f"Initializing Binoculars on GPU (CUDA device: {torch.cuda.get_device_name(0)})")
        
        _binoculars = Binoculars()
        _binoculars.model.to(device)
    return _binoculars


def get_binoculars_score(text):
    """
    Get AI detection score for a single text.
    Higher scores indicate more likely AI-generated.
    
    Returns 0.0 for empty/invalid text or on error.
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    
    try:
        bino = _get_binoculars()
        score = bino.compute_score(text)
        return float(score)
    except Exception as e:
        print(f"Warning: Binoculars scoring failed: {e}")
        return 0.0


def average_ai_score(texts):
    """
    Calculate average AI detection score across multiple texts.
    Processes texts in batches for better GPU utilization.
    
    Returns 0.0 if no valid texts.
    """
    if not texts:
        return 0.0
    
    # Filter out empty/very short texts
    valid_texts = [t for t in texts if t and len(t.strip()) >= 10]
    
    if not valid_texts:
        return 0.0
    
    scores = []
    batch_size = 8  # Process in batches for better GPU efficiency
    
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        for text in batch:
            score = get_binoculars_score(text)
            if score != 0.0:  # Only include valid scores
                scores.append(score)
    
    return float(np.mean(scores)) if scores else 0.0
