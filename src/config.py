# Configuration file for the project
DATASET_FILES = [
    "data/processed/posts/posts_1.json",
    "data/processed/posts/posts_2.json",
    "data/processed/posts/posts_3.json",
    "data/processed/posts/posts_4.json",
    "data/processed/posts/posts_5.json",
    "data/processed/posts/posts_6.json",
    "data/processed/posts/posts_7.json",
]

# List of bot ID files for training and evaluation
BOT_IDS_FILES = [
    "data/processed/bots/bots_1.txt",
    "data/processed/bots/bots_2.txt",
    "data/processed/bots/bots_3.txt",
    "data/processed/bots/bots_4.txt",
    "data/processed/bots/bots_5.txt",
    "data/processed/bots/bots_6.txt",
    "data/processed/bots/bots_7.txt",
]

# Path to save the trained model
MODEL_PATH = "models/bot_classifier.pkl"
# Path to save the predicted bot IDs
PREDICTIONS_OUTPUT = "predictions.txt"
