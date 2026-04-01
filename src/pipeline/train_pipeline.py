import os
from src.data.loader import load_all_bot_ids, load_all_datasets
from src.features.builder import build_feature_matrix
from src.models.train import train_baseline
from src.models.persistence import save_model
from src.config import DATASET_FILES, BOT_IDS_FILES, MODEL_PATH


def run():
    bot_ids = load_all_bot_ids(BOT_IDS_FILES)
    users = load_all_datasets(DATASET_FILES, bot_ids)

    X, y = build_feature_matrix(users)

    model, results, importance = train_baseline(X, y)

    print(f"F1: {results['test_f1'].mean():.3f}")
    print("\nTop features:")
    print(importance.head(10))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_model(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
