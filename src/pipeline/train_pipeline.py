import os
from src.data.loader import load_all_bot_ids, load_all_datasets, load_dataset, load_bot_ids, group_posts_by_user
from src.features.builder import build_feature_matrix
from src.models.train import train_baseline
from src.models.persistence import save_model
from src.config import DATASET_FILES, BOT_IDS_FILES, MODEL_PATH


def run():
    # Load all training data
    print("Loading training datasets...")
    bot_ids = load_all_bot_ids(BOT_IDS_FILES)
    users = load_all_datasets(DATASET_FILES, bot_ids)
    print(f"Loaded {len(users)} users from {len(DATASET_FILES)} datasets")

    print("\nBuilding feature matrix...")
    X, y = build_feature_matrix(users)
    print(f"Feature matrix shape: {X.shape[0]} users × {X.shape[1]} features")

    # Prepare datasets 1-6 for threshold optimization
    print("\nPreparing datasets for threshold optimization...")
    datasets_for_threshold = []
    for i in range(6):  # Use datasets 0-5 (files 1-6)
        dataset_path = DATASET_FILES[i]
        bot_ids_path = BOT_IDS_FILES[i]
        
        data = load_dataset(dataset_path)
        bot_ids_set = load_bot_ids(bot_ids_path)
        posts_by_user = group_posts_by_user(data["posts"])
        
        dataset_users = []
        for user in data["users"]:
            uid = user["id"]
            label = int(uid in bot_ids_set)
            dataset_users.append({
                "user": user,
                "posts": posts_by_user.get(uid, []),
                "label": label,
            })
        
        X_dataset, y_dataset = build_feature_matrix(dataset_users)
        datasets_for_threshold.append((X_dataset, y_dataset))

    # Train model with threshold optimization
    print("\nTraining XGBoost model...")
    model, results, importance, optimal_threshold, tfidf_scorer = train_baseline(
        X, y, datasets_for_threshold=datasets_for_threshold, all_users_data=users
    )

    print(f"\nCross-validation F1 score: {results['test_f1'].mean():.3f}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print("\nTop features:")
    print(importance.head(10))

    # Validate on dataset 7 (held-out)
    if len(DATASET_FILES) > 6:
        print("\n" + "="*50)
        print("Validating on held-out dataset 7...")
        print("="*50)
        
        data = load_dataset(DATASET_FILES[6])
        bot_ids_set = load_bot_ids(BOT_IDS_FILES[6])
        posts_by_user = group_posts_by_user(data["posts"])
        
        validation_users = []
        for user in data["users"]:
            uid = user["id"]
            label = int(uid in bot_ids_set)
            validation_users.append({
                "user": user,
                "posts": posts_by_user.get(uid, []),
                "label": label,
            })
        
        X_val, y_val = build_feature_matrix(validation_users)
        
        # Predict using optimal threshold
        probas = model.predict_proba(X_val)[:, 1]
        y_pred = (probas >= optimal_threshold).astype(int)
        
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        from src.models.threshold_optimizer import calculate_competition_score
        
        val_score = calculate_competition_score(y_val, y_pred)
        val_f1 = f1_score(y_val, y_pred)
        val_precision = precision_score(y_val, y_pred, zero_division=0)
        val_recall = recall_score(y_val, y_pred, zero_division=0)
        val_accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Validation Score: {val_score}")
        print(f"Validation F1: {val_f1:.3f}")
        print(f"Validation Precision: {val_precision:.3f}")
        print(f"Validation Recall: {val_recall:.3f}")
        print(f"Validation Accuracy: {val_accuracy:.3f}")

    # Save model with optimal threshold and TF-IDF scorer
    print("\nSaving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_model(model, MODEL_PATH, threshold=optimal_threshold, tfidf_scorer=tfidf_scorer)
    print(f"Model saved to {MODEL_PATH}")
