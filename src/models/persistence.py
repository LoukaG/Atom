import joblib
import os


def save_model(model, path, threshold=None, vectorizer=None, tfidf_scorer=None):
    """
    Save model, optional threshold, TF-IDF vectorizer, and TF-IDF scorer.
    
    Args:
        model: Trained model
        path: Path to save model file
        threshold: Optional classification threshold (default: None)
        vectorizer: Optional TfidfVectorizer instance for features (default: None)
        tfidf_scorer: Optional TFIDFBotScorer instance for ensemble (default: None)
    
    Saves:
    - Model + threshold in main file (if threshold provided)
    - Vectorizer in separate .vectorizer.pkl file (if vectorizer provided)
    - TF-IDF scorer in separate .tfidf_scorer.pkl file (if tfidf_scorer provided)
    """
    # Save model and threshold
    if threshold is not None:
        bundle = {"model": model, "threshold": threshold}
        joblib.dump(bundle, path)
    else:
        joblib.dump(model, path)
    
    # Save vectorizer separately (for TF-IDF features)
    if vectorizer is not None:
        vectorizer_path = path + '.vectorizer.pkl'
        joblib.dump(vectorizer, vectorizer_path)
        print(f"  Vectorizer saved to {vectorizer_path}")
    
    # Save TF-IDF scorer separately (for ensemble scoring)
    if tfidf_scorer is not None:
        tfidf_scorer_path = path + '.tfidf_scorer.pkl'
        tfidf_scorer.save(tfidf_scorer_path)
        print(f"  TF-IDF scorer saved to {tfidf_scorer_path}")


def load_model(path):
    """
    Load model, threshold, TF-IDF vectorizer, and TF-IDF scorer.
    
    Args:
        path: Path to model file
    
    Returns:
        Tuple of (model, threshold, vectorizer, tfidf_scorer)
        - model: The trained model
        - threshold: Classification threshold (None if not saved)
        - vectorizer: TfidfVectorizer for features (None if not saved)
        - tfidf_scorer: TFIDFBotScorer for ensemble (None if not saved)
    
    For backward compatibility:
    - If only model saved: returns (model, None, None, None)
    - If model+threshold saved: returns (bundle["model"], bundle["threshold"], vectorizer, tfidf_scorer)
    """
    # Load main model file
    loaded = joblib.load(path)
    
    # Extract model and threshold
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        threshold = loaded.get("threshold", None)
    else:
        model = loaded
        threshold = None
    
    # Try to load vectorizer (for TF-IDF features)
    vectorizer = None
    vectorizer_path = path + '.vectorizer.pkl'
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    
    # Try to load TF-IDF scorer (for ensemble scoring)
    tfidf_scorer = None
    tfidf_scorer_path = path + '.tfidf_scorer.pkl'
    if os.path.exists(tfidf_scorer_path):
        from .tfidf_scorer import TFIDFBotScorer
        tfidf_scorer = TFIDFBotScorer.load(tfidf_scorer_path)
    
    return model, threshold, vectorizer, tfidf_scorer
