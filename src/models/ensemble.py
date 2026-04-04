"""
Ensemble Bot Detection Model

Combines XGBoost feature-based classifier with TF-IDF content-based scorer.

Weighted combination:
    final_score = 0.3 * TF-IDF + 0.7 * XGBoost

This approach leverages:
- XGBoost: Captures behavioral patterns (temporal, volume, profile features)
- TF-IDF: Captures content patterns (vocabulary, text similarity)
"""

import numpy as np


class EnsembleBotDetector:
    """Ensemble bot detector combining XGBoost and TF-IDF."""
    
    def __init__(self, xgboost_model, tfidf_scorer, 
                 tfidf_weight=0.3, xgboost_weight=0.7):
        """
        Initialize ensemble detector.
        
        Args:
            xgboost_model: Trained XGBoost classifier
            tfidf_scorer: Trained TFIDFBotScorer
            tfidf_weight: Weight for TF-IDF score (default: 0.3)
            xgboost_weight: Weight for XGBoost score (default: 0.7)
        """
        if tfidf_weight + xgboost_weight != 1.0:
            raise ValueError(f"Weights must sum to 1.0, got {tfidf_weight + xgboost_weight}")
        
        self.xgboost_model = xgboost_model
        self.tfidf_scorer = tfidf_scorer
        self.tfidf_weight = tfidf_weight
        self.xgboost_weight = xgboost_weight
    
    def predict_proba_single(self, X_features, user_posts):
        """
        Get ensemble probability for a single user.
        
        Args:
            X_features: Feature vector for XGBoost (1D array or DataFrame row)
            user_posts: List of post dicts for TF-IDF scoring
        
        Returns:
            float: Bot probability (0-1 range)
        """
        # Get XGBoost probability
        if hasattr(X_features, 'values'):
            # DataFrame row
            X_features = X_features.values.reshape(1, -1)
        else:
            # Ensure 2D array
            X_features = np.array(X_features).reshape(1, -1)
        
        xgboost_proba = self.xgboost_model.predict_proba(X_features)[0, 1]
        
        # Get TF-IDF score
        tfidf_score = self.tfidf_scorer.score(user_posts)
        
        # Weighted combination
        ensemble_score = (
            self.tfidf_weight * tfidf_score + 
            self.xgboost_weight * xgboost_proba
        )
        
        return float(ensemble_score)
    
    def predict_proba_batch(self, X_features, all_users):
        """
        Get ensemble probabilities for multiple users.
        
        Args:
            X_features: Feature matrix for XGBoost (DataFrame or 2D array)
            all_users: List of user data dicts with "posts" field
        
        Returns:
            np.array: Bot probabilities for each user
        """
        # Get XGBoost probabilities
        xgboost_probas = self.xgboost_model.predict_proba(X_features)[:, 1]
        
        # Get TF-IDF scores
        tfidf_scores = self.tfidf_scorer.score_batch(all_users)
        
        # Weighted combination
        ensemble_scores = (
            self.tfidf_weight * tfidf_scores + 
            self.xgboost_weight * xgboost_probas
        )
        
        return ensemble_scores
    
    def predict_batch(self, X_features, all_users, threshold=0.5):
        """
        Get binary predictions for multiple users.
        
        Args:
            X_features: Feature matrix for XGBoost
            all_users: List of user data dicts with "posts" field
            threshold: Classification threshold (default: 0.5)
        
        Returns:
            np.array: Binary predictions (0=human, 1=bot)
        """
        probas = self.predict_proba_batch(X_features, all_users)
        return (probas >= threshold).astype(int)
    
    def get_individual_scores(self, X_features, user_posts):
        """
        Get individual scores from each model (for debugging/analysis).
        
        Args:
            X_features: Feature vector for XGBoost
            user_posts: List of post dicts for TF-IDF scoring
        
        Returns:
            dict: {
                'xgboost_proba': XGBoost probability,
                'tfidf_score': TF-IDF score,
                'ensemble_score': Combined score
            }
        """
        # Get XGBoost probability
        if hasattr(X_features, 'values'):
            X_features = X_features.values.reshape(1, -1)
        else:
            X_features = np.array(X_features).reshape(1, -1)
        
        xgboost_proba = self.xgboost_model.predict_proba(X_features)[0, 1]
        
        # Get TF-IDF score
        tfidf_score = self.tfidf_scorer.score(user_posts)
        
        # Calculate ensemble
        ensemble_score = (
            self.tfidf_weight * tfidf_score + 
            self.xgboost_weight * xgboost_proba
        )
        
        return {
            'xgboost_proba': float(xgboost_proba),
            'tfidf_score': float(tfidf_score),
            'ensemble_score': float(ensemble_score)
        }


def create_ensemble(xgboost_model, tfidf_scorer, tfidf_weight=0.3):
    """
    Create an ensemble bot detector.
    
    Args:
        xgboost_model: Trained XGBoost classifier
        tfidf_scorer: Trained TFIDFBotScorer (can be None to disable)
        tfidf_weight: Weight for TF-IDF (default: 0.3)
    
    Returns:
        EnsembleBotDetector or None if tfidf_scorer is None
    
    Note: If tfidf_scorer is None, returns None (caller should use XGBoost only)
    """
    if tfidf_scorer is None:
        print("Warning: TF-IDF scorer not provided, ensemble disabled")
        return None
    
    xgboost_weight = 1.0 - tfidf_weight
    return EnsembleBotDetector(
        xgboost_model, 
        tfidf_scorer, 
        tfidf_weight=tfidf_weight,
        xgboost_weight=xgboost_weight
    )
