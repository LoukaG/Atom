# Atom
Atom is an intelligent tool designed to detect bot accounts on social media platforms. Developed for the Bot or Not Hackathon organized by McGill University, Atom fights fake profiles with precision and efficiency, helping maintain authentic engagement online. Inspired by the movie Real Steel, Atom is your knockout champion against bots.

## Features

Atom uses a sophisticated **ensemble model** combining XGBoost feature-based classification with TF-IDF content analysis:

### Ensemble Architecture
- **XGBoost (70%)**: Feature-based classifier for behavioral patterns
- **TF-IDF (30%)**: Content-based similarity to known bot/human text patterns
- **Weighted Combination**: `final_score = 0.3 × TF-IDF + 0.7 × XGBoost`

This dual approach captures both behavioral anomalies and content patterns, providing more robust bot detection.

### Core Features (Used by XGBoost)
- **Volume**: Post count, tweet count
- **Content patterns**: URL ratio, hashtag ratio, mention ratio, duplicate ratio
- **Text analysis**: Average/standard deviation of post length, lexical richness, typo rate
- **Temporal**: Posting regularity (standard deviation of time intervals)
- **Profile**: Z-score, description presence/length

### TF-IDF Features (Content Analysis)
- **Vocabulary Patterns**: Identifies characteristic words/phrases used by bots vs humans
- **Configuration**: 5000 features, unigrams + bigrams, English stopwords removed
- **Similarity Scoring**: Compares user text to prototype vectors of known bots/humans
- **Complementary**: Captures text patterns that behavioral features might miss

### AI-Generated Text Detection
- **AI Score**: Detects AI-generated content using RoBERTa-based classifier
- **Model**: roberta-base-openai-detector (OpenAI's GPT-2 output detector)
- **Ultra-lightweight**: Only ~125MB model size, ~500MB VRAM usage
- **Fast inference**: ~0.03s per text on GPU (10-100x faster than large models)
- Returns statistics (avg, std, min, max) of AI detection scores across user posts
- Higher scores indicate content more likely to be AI-generated

**Key Benefits**:
- Fits on any GPU (tested on 8GB RTX 2070)
- No CPU offloading needed - entire model runs on GPU efficiently
- Pre-trained by OpenAI specifically for AI text detection
- Supports batch processing for optimal performance

## Installation

### Requirements
- Python 3.8+
- **Optional GPU**: 1GB+ VRAM recommended for faster AI detection
  - AI detection model: ~500MB VRAM
  - Works perfectly on budget GPUs (tested on RTX 2070 8GB)
  - CPU fallback available (slower but functional)
- Internet connection (for initial model downloads)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**GPU Support**: The project uses PyTorch with CUDA for GPU-accelerated AI text detection. If you encounter issues, install PyTorch manually:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**First Run**: The AI detection model will download ~500MB on first use. This is a one-time download and requires an internet connection.

## Troubleshooting

### Model Download Issues

If you encounter errors downloading the AI detection model:

1. **Check internet connection**: Model downloads from HuggingFace Hub
2. **Clear cache**: `rm -rf ~/.cache/huggingface/` (or equivalent on Windows)
3. **Manual download**: Visit https://huggingface.co/roberta-base-openai-detector

### Performance Issues

If AI detection is slow:

1. **Use GPU**: Ensure CUDA is properly installed and PyTorch detects your GPU
2. **Reduce batch size**: Edit `batch_size` parameter in `src/features/ai_detection.py`
3. **Disable AI features**: Comment out AI features in `src/features/builder.py` if not needed

### TF-IDF Ensemble Configuration

The TF-IDF ensemble can be customized:

1. **Weights**: Modify `tfidf_weight` in `src/models/ensemble.py::create_ensemble()` (default: 0.3)
   - Higher TF-IDF weight (e.g., 0.5): More emphasis on text content patterns
   - Lower TF-IDF weight (e.g., 0.2): More emphasis on behavioral features
   
2. **Vocabulary Size**: Adjust `max_features` in `src/models/tfidf_scorer.py` (default: 5000)
   - Larger vocabulary: More detailed text analysis, slower training
   - Smaller vocabulary: Faster training, may miss nuanced patterns
   
3. **Disable Ensemble**: To train XGBoost only, set `all_users_data=None` in `train_baseline()` call

## Usage

### Training
```bash
python main.py
```

This runs the complete pipeline: data loading, feature engineering, TF-IDF training, XGBoost training, and saves both models to `models/bot_classifier.pkl` (plus `.tfidf_scorer.pkl`).

**Note**: 
- First run will download the AI detection model (~500MB)
- TF-IDF training adds ~1-2 seconds to overall training time
- The ensemble approach typically improves F1 score by 2-5% compared to XGBoost alone
- Inference on GPU takes ~0.03s per post vs ~0.5-1s on CPU

### Evaluation/Prediction
```bash
# Predict bots from a new dataset (uses ensemble if available)
python evaluate.py --input data/dataset.posts&users.1.json --output predictions.txt

# Evaluate with known labels (calculate metrics)
python evaluate.py --input data/dataset.posts&users.1.json --bot-ids data/dataset.bots.1.txt --output predictions.txt

# Use a different model
python evaluate.py --input mydata.json --model models/my_model.pkl --output results.txt
```

Outputs predicted bot IDs to a TXT file (one ID per line). If `--bot-ids` is provided, also displays F1, precision, recall, and accuracy.

**Ensemble Note**: The evaluation automatically uses ensemble scoring if both XGBoost and TF-IDF models are available. For legacy models (XGBoost only), falls back to single model prediction.

# Dataset
For this project, Atom uses the following datasets to train and evaluate the bot detection models:
- Custom Hackathon Dataset

# Sources
- Feng, S., Tan, Z., Wan, H., Wang, N., Chen, Z., Zhang, B., et al. (2022). TwiBot-22: Towards Graph-Based Twitter Bot Detection. Xi’an Jiaotong University, University of Washington, Tsinghua University, University of Virginia.
- Alarfaj, F. K., Ahmad, H., Khan, H. U., Alomair, A. M., Almusallam, N., & Ahmed, M. (2023). Twitter Bot Detection Using Diverse Content Features and Applying Machine Learning Algorithms. Sustainability, 15(8), 6662.