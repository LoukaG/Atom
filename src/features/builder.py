import numpy as np
import pandas as pd
from .text import lexical_richness, typo_rate
from .temporal import posting_regularity


def extract_features(user_data):
    user = user_data["user"]
    posts = user_data["posts"]

    texts = [p["text"] for p in posts]
    timestamps = [p["created_at"] for p in posts]
    n_posts = len(posts)

    lengths = [len(t) for t in texts] or [0]

    return {
        "n_posts": n_posts,
        "tweet_count": user.get("tweet_count", 0),
        "z_score": user.get("z_score", 0.0),
        "url_ratio": sum("http" in t for t in texts) / max(n_posts, 1),
        "hashtag_ratio": sum("#" in t for t in texts) / max(n_posts, 1),
        "mention_ratio": sum("@" in t for t in texts) / max(n_posts, 1),
        "duplicate_ratio": 1 - len(set(texts)) / max(n_posts, 1),
        "avg_post_length": float(np.mean(lengths)),
        "std_post_length": float(np.std(lengths)),
        "lexical_richness": lexical_richness(texts),
        "typo_rate": typo_rate(texts),
        "posting_regularity_std": posting_regularity(timestamps),
        "has_description": int(bool(user.get("description"))),
        "desc_length": len(user.get("description", "") or ""),
    }


def build_feature_matrix(all_users):
    rows = [extract_features(u) for u in all_users]
    labels = [u["label"] for u in all_users]

    return pd.DataFrame(rows), np.array(labels)
