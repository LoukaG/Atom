import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from .threshold_optimizer import find_optimal_threshold
from .tfidf_scorer import train_tfidf_scorer


def train_baseline(X, y, datasets_for_threshold=None, all_users_data=None):
    """
    Train XGBoost classifier with optional threshold optimization.
    Optionally trains TF-IDF scorer if all_users_data is provided.
    
    Args:
        X: Feature matrix
        y: Labels
        datasets_for_threshold: Optional list of (X, y) tuples for threshold optimization
        all_users_data: Optional list of user data dicts for TF-IDF training
    
    Returns:
        tuple: (model, results, importance, optimal_threshold, tfidf_scorer)
            - optimal_threshold is None if datasets_for_threshold not provided
            - tfidf_scorer is None if all_users_data not provided
    """
    scale = (y == 0).sum() / max((y == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running 5-fold cross-validation...")
    results = cross_validate(
        model, X, y, cv=cv, scoring=["f1", "precision", "recall", "accuracy"]
    )

    print("Training final model on all data...")
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    
    optimal_threshold = None
    if datasets_for_threshold is not None:
        print("Optimizing classification threshold...")
        optimal_threshold, threshold_results = find_optimal_threshold(
            model, datasets_for_threshold
        )
        print(f"Threshold optimization complete!")
    
    # Train TF-IDF scorer if data provided
    tfidf_scorer = None
    if all_users_data is not None:
        print("Training TF-IDF scorer...")
        tfidf_scorer = train_tfidf_scorer(all_users_data, y, max_features=5000)
        print("TF-IDF scorer trained!")

    return model, results, importance, optimal_threshold, tfidf_scorer
