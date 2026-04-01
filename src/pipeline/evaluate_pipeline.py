from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from src.data.loader import load_dataset, group_posts_by_user
from src.features.builder import build_feature_matrix
from src.models.persistence import load_model


def calculate_score(y_true, y_pred):
    """
    Calculate competition score based on:
    - True Positive (bot correctly detected): +2
    - False Negative (bot not detected): -2
    - False Positive (non-bot flagged as bot): -6
    - True Negative (non-bot correctly identified): 0
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    score = (tp * 2) + (fn * -2) + (fp * -6)
    return score


def evaluate(input_file, model_path, bot_ids=None):
    bundle = load_model(model_path)

    # Supporte deux formats :
    # - ancien format : le modèle directement
    # - nouveau format : dict {"model": ..., "threshold": ...}
    if isinstance(bundle, dict):
        model = bundle["model"]
        threshold = float(bundle.get("threshold", 0.5))
    else:
        model = bundle
        threshold = 0.5

    data = load_dataset(input_file)
    users = data["users"]
    posts_by_user = group_posts_by_user(data["posts"])

    all_users = []
    for user in users:
        uid = user["id"]
        label = int(uid in bot_ids) if bot_ids else 0
        all_users.append(
            {
                "user": user,
                "posts": posts_by_user.get(uid, []),
                "label": label,
            }
        )

    X, y_true = build_feature_matrix(all_users)

    # Utiliser le seuil optimal au lieu du seuil par défaut 0.5
    probas = model.predict_proba(X)[:, 1]
    y_pred = (probas >= threshold).astype(int)

    bot_user_ids = [
        all_users[i]["user"]["id"] for i in range(len(y_pred)) if y_pred[i] == 1
    ]

    metrics = None
    if bot_ids is not None:
        score = calculate_score(y_true, y_pred)
        metrics = {
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "score": score,
        }

    return bot_user_ids, metrics


def save_bot_ids(user_ids, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for uid in user_ids:
            f.write(f"{uid}\n")
