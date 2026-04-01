# Atom
Atom is an intelligent tool designed to detect bot accounts on social media platforms. Developed for the Bot or Not Hackathon organized by McGill University, Atom fights fake profiles with precision and efficiency, helping maintain authentic engagement online. Inspired by the movie Real Steel, Atom is your knockout champion against bots.

## Features

Atom uses a sophisticated XGBoost-based machine learning model with the following feature categories:

### Core Features
- **Volume**: Post count, tweet count
- **Content patterns**: URL ratio, hashtag ratio, mention ratio, duplicate ratio
- **Text analysis**: Average/standard deviation of post length, lexical richness, typo rate
- **Temporal**: Posting regularity (standard deviation of time intervals)
- **Profile**: Z-score, description presence/length

### AI-Generated Text Detection
- **AI Score**: Detects AI-generated content using the Binoculars library
- Runs on GPU (CUDA) for fast inference
- Returns average AI detection score across all user posts
- Higher scores indicate content more likely to be AI-generated

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for AI detection)
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

**First Run**: The Binoculars model will download ~10GB of model files (Falcon-7B) on first use. This is a one-time download and requires an internet connection.

## Usage

### Training
```bash
python main.py
```

This runs the complete pipeline: data loading, feature engineering, model training, evaluation, and saves the trained model to `models/bot_classifier.pkl`.

**Note**: First run will download Binoculars models (~1-2 GB). Inference on GPU takes ~0.05-0.1s per post vs ~1-2s on CPU.

### Evaluation/Prediction
```bash
# Predict bots from a new dataset
python evaluate.py --input data/dataset.posts&users.1.json --output predictions.txt

# Evaluate with known labels (calculate metrics)
python evaluate.py --input data/dataset.posts&users.1.json --bot-ids data/dataset.bots.1.txt --output predictions.txt

# Use a different model
python evaluate.py --input mydata.json --model models/my_model.pkl --output results.txt
```

Outputs predicted bot IDs to a TXT file (one ID per line). If `--bot-ids` is provided, also displays F1, precision, recall, and accuracy.

# Dataset
For this project, Atom uses the following datasets to train and evaluate the bot detection models:
- Custom Hackathon Dataset

# Sources
- Feng, S., Tan, Z., Wan, H., Wang, N., Chen, Z., Zhang, B., et al. (2022). TwiBot-22: Towards Graph-Based Twitter Bot Detection. Xi’an Jiaotong University, University of Washington, Tsinghua University, University of Virginia.
- Alarfaj, F. K., Ahmad, H., Khan, H. U., Alomair, A. M., Almusallam, N., & Ahmed, M. (2023). Twitter Bot Detection Using Diverse Content Features and Applying Machine Learning Algorithms. Sustainability, 15(8), 6662.