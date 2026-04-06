# Atom

Atom is an intelligent tool designed to detect bot accounts on social media platforms. Developed for the Bot or Not Hackathon organized by McGill University, Atom fights fake profiles with precision and efficiency, helping maintain authentic engagement online. Inspired by the movie Real Steel, Atom is your knockout champion against bots.

## How It Works

Atom uses a hybrid machine learning approach combining XGBoost with TF-IDF text analysis to identify bot accounts:

1. **Feature Extraction**: Analyzes user profiles and posts to extract 14+ features including:
   - Content patterns (URL/hashtag/mention ratios, duplicates)
   - Text quality (lexical richness, typo rate, AI-generated content scores)
   - Temporal patterns (posting regularity)
   - Profile characteristics (description, tweet count)

2. **Ensemble Model**: Combines two detection approaches:
   - **XGBoost Classifier**: Trains on extracted features using gradient boosting
   - **TF-IDF Scorer**: Analyzes text content for bot-like language patterns
   - Weighted ensemble (60% XGBoost, 40% TF-IDF) for final predictions

3. **Threshold Optimization**: Finds optimal classification threshold by maximizing competition score across multiple datasets

4. **Cross-Validation**: Uses 5-fold stratified cross-validation for robust model evaluation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Atom.git
cd Atom
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

**Required packages:**
- numpy
- pandas
- scikit-learn
- xgboost
- torch
- transformers
- scipy

## Usage

### Train a New Model

To train a bot detection model on the provided datasets:

```bash
python atom.py --build
```

This command will:
- Load all training datasets (7 datasets)
- Extract features from users and their posts
- Train an XGBoost classifier with 5-fold cross-validation
- Optimize the classification threshold
- Train a TF-IDF text scorer
- Validate on held-out dataset 7
- Save the trained model to `models/bot_classifier.pkl`

**Expected output:**
```
==================================================
Building new bot detection model
==================================================
Loading training datasets...
Loading dataset 1/7...
...
Cross-validation F1 score: 0.XXX
Optimal threshold: 0.XXX
Model saved to models/bot_classifier.pkl
==================================================
Model training complete!
==================================================
```

### Predict Bot Accounts

To detect bots in a new dataset:

```bash
python atom.py --input data/dataset.posts&users.1.json --output predictions.txt
```

**Options:**
- `--input`: Path to input JSON file (required for evaluation)
- `--output`: Path to output file for predicted bot IDs (default: `predictions-en.txt`)
- `--model`: Path to trained model file (default: `models/bot_classifier.pkl`)
- `--bot-ids`: Optional path to known bot IDs file for calculating metrics

**Example with evaluation:**
```bash
python atom.py --input data/dataset.posts&users.1.json --bot-ids data/dataset.bots.1.txt --output results.txt
```

When `--bot-ids` is provided, the tool calculates and displays:
- Competition Score (TP×2 + FN×-2 + FP×-6)
- F1 Score
- Precision
- Recall
- Accuracy

### Alternative Training Method

You can also use the simplified training script:

```bash
python main.py
```

This runs the same training pipeline as `python atom.py --build`.

## Data Format

### Input Dataset Format
JSON file with the following structure:
```json
{
  "id": 1,
  "lang": "en",
  "users": [
    {
      "id": "user_123",
      "tweet_count": 456,
      "z_score": 0.5,
      "username": "example_user",
      "name": "Example User",
      "description": "User bio...",
      "location": "City, Country"
    }
  ],
  "posts": [
    {
      "id": "post_789",
      "author_id": "user_123",
      "text": "Post content...",
      "created_at": "2024-01-01T12:00:00Z",
      "lang": "en"
    }
  ]
}
```

### Bot IDs File Format
Plain text file with one user ID per line:
```
user_123
user_456
user_789
```

### Output Format
Plain text file with predicted bot user IDs (one per line):
```
user_bot_001
user_bot_042
user_bot_123
```

## Project Structure

```
Atom/
├── atom.py                      # Main CLI interface
├── main.py                      # Simplified training script
├── requirements.txt             # Python dependencies
├── data/                        # Training and test datasets
│   ├── dataset.posts&users.*.json
│   └── dataset.bots.*.txt
├── models/                      # Saved trained models
│   └── bot_classifier.pkl
├── src/
│   ├── config.py               # Configuration and paths
│   ├── data/
│   │   └── loader.py           # Dataset loading utilities
│   ├── features/
│   │   ├── builder.py          # Feature matrix construction
│   │   ├── text.py             # Text-based features
│   │   ├── temporal.py         # Time-based features
│   │   └── ai_detection.py     # AI content detection
│   ├── models/
│   │   ├── train.py            # Model training
│   │   ├── persistence.py      # Model save/load
│   │   ├── threshold_optimizer.py
│   │   ├── tfidf_scorer.py     # TF-IDF text scoring
│   │   └── ensemble.py         # Ensemble predictor
│   └── pipeline/
│       ├── train_pipeline.py   # Training orchestration
│       └── evaluate_pipeline.py # Evaluation orchestration
└── predictions-en.txt          # Default output file
```

## Dataset

For this project, Atom uses the following datasets to train and evaluate the bot detection models:
- Custom Hackathon Dataset (7 datasets with labeled bot accounts)

## Performance

The model is evaluated using:
- **Competition Score**: Custom metric (TP×2 + FN×-2 + FP×-6)
- **F1 Score**: Harmonic mean of precision and recall
- **Cross-validation**: 5-fold stratified CV on training data
- **Held-out validation**: Dataset 7 reserved for final validation

## Sources

- Feng, S., Tan, Z., Wan, H., Wang, N., Chen, Z., Zhang, B., et al. (2022). TwiBot-22: Towards Graph-Based Twitter Bot Detection. Xi'an Jiaotong University, University of Washington, Tsinghua University, University of Virginia.
- Alarfaj, F. K., Ahmad, H., Khan, H. U., Alomair, A. M., Almusallam, N., & Ahmed, M. (2023). Twitter Bot Detection Using Diverse Content Features and Applying Machine Learning Algorithms. Sustainability, 15(8), 6662.
