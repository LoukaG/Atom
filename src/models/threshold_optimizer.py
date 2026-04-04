import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_competition_score(y_true, y_pred):
    """
    Calculate competition score:
    - True Positive (bot correctly detected): +2
    - False Negative (bot not detected): -2
    - False Positive (non-bot flagged as bot): -6
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    score = (tp * 2) + (fn * -2) + (fp * -6)
    return score


def find_optimal_threshold(model, datasets_info, threshold_range=None):
    """
    Find optimal classification threshold by testing on multiple datasets.
    
    Args:
        model: Trained classifier with predict_proba method
        datasets_info: List of tuples (X, y) for each dataset
        threshold_range: Array of thresholds to test (default: 0.30 to 0.70 step 0.05)
    
    Returns:
        tuple: (optimal_threshold, detailed_results)
            - optimal_threshold: float, threshold with best average score
            - detailed_results: dict with scores per threshold and dataset
    """
    if threshold_range is None:
        threshold_range = np.arange(0.30, 0.71, 0.05)
    
    results = {
        "thresholds": threshold_range.tolist(),
        "scores_per_dataset": [],
        "avg_scores": []
    }
    
    for threshold in threshold_range:
        scores_for_threshold = []
        
        for X, y_true in datasets_info:
            probas = model.predict_proba(X)[:, 1]
            y_pred = (probas >= threshold).astype(int)
            score = calculate_competition_score(y_true, y_pred)
            scores_for_threshold.append(score)
        
        results["scores_per_dataset"].append(scores_for_threshold)
        avg_score = np.mean(scores_for_threshold)
        results["avg_scores"].append(avg_score)
    
    best_idx = np.argmax(results["avg_scores"])
    optimal_threshold = threshold_range[best_idx]
    
    return optimal_threshold, results
