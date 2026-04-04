"""
TF-IDF Scorer for Bot Detection

Uses TF-IDF (Term Frequency-Inverse Document Frequency) to score text based on
similarity to known bot and human text patterns.

Approach:
1. Train TF-IDF vectorizer on corpus of bot/human posts
2. Create prototype vectors (centroids) for bot and human classes
3. Score new texts by comparing cosine similarity to both prototypes
4. Return probability that text is bot-generated
"""

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class TFIDFBotScorer:
    """TF-IDF-based bot detection scorer using similarity to prototype vectors."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF scorer.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams to extract (default: unigrams + bigrams)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,                 # Word must appear in at least 2 documents
            max_df=0.8,               # Ignore words in >80% of documents
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            sublinear_tf=True         # Use log scaling for term frequency
        )
        self.bot_prototype = None
        self.human_prototype = None
        self.is_fitted = False
    
    def _preprocess_text(self, text):
        """Clean text before TF-IDF vectorization."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags but keep the text
        text = re.sub(r'#', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _combine_user_posts(self, posts):
        """Combine all user posts into a single document."""
        if not posts:
            return ""
        
        texts = [p.get("text", "") for p in posts]
        cleaned_texts = [self._preprocess_text(t) for t in texts]
        return " ".join(cleaned_texts)
    
    def fit(self, all_users, labels):
        """
        Fit TF-IDF vectorizer and compute prototype vectors.
        
        Args:
            all_users: List of user data dicts with "posts" field
            labels: Array of labels (0=human, 1=bot)
        
        Returns:
            self
        """
        # Combine all posts for each user into single documents
        documents = [self._combine_user_posts(user["posts"]) for user in all_users]
        
        # Fit vectorizer on all documents
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Convert to dense array for easier manipulation
        tfidf_dense = tfidf_matrix.toarray()
        
        # Compute prototypes (centroids) for each class
        labels = np.array(labels)
        bot_indices = np.where(labels == 1)[0]
        human_indices = np.where(labels == 0)[0]
        
        if len(bot_indices) > 0:
            self.bot_prototype = tfidf_dense[bot_indices].mean(axis=0).reshape(1, -1)
        else:
            # Fallback: zero vector
            self.bot_prototype = np.zeros((1, tfidf_dense.shape[1]))
        
        if len(human_indices) > 0:
            self.human_prototype = tfidf_dense[human_indices].mean(axis=0).reshape(1, -1)
        else:
            # Fallback: zero vector
            self.human_prototype = np.zeros((1, tfidf_dense.shape[1]))
        
        self.is_fitted = True
        
        print(f"TF-IDF scorer fitted:")
        print(f"  - Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  - Bot samples: {len(bot_indices)}")
        print(f"  - Human samples: {len(human_indices)}")
        
        return self
    
    def score(self, user_posts):
        """
        Score a user's posts for bot probability.
        
        Args:
            user_posts: List of post dicts with "text" field
        
        Returns:
            float: Probability that user is a bot (0-1 range)
        """
        if not self.is_fitted:
            raise ValueError("TFIDFBotScorer must be fitted before scoring")
        
        # Combine user posts into single document
        document = self._combine_user_posts(user_posts)
        
        if not document or len(document.strip()) < 10:
            # Not enough text, return neutral score
            return 0.5
        
        # Transform to TF-IDF vector
        tfidf_vector = self.vectorizer.transform([document])
        
        # Calculate cosine similarity to both prototypes
        bot_sim = cosine_similarity(tfidf_vector, self.bot_prototype)[0][0]
        human_sim = cosine_similarity(tfidf_vector, self.human_prototype)[0][0]
        
        # Handle edge case where both similarities are 0
        if bot_sim == 0 and human_sim == 0:
            return 0.5
        
        # Convert to probability: bot_sim / (bot_sim + human_sim)
        # This gives us P(bot | text) assuming equal priors
        bot_probability = bot_sim / (bot_sim + human_sim + 1e-10)
        
        return float(bot_probability)
    
    def score_batch(self, all_users):
        """
        Score multiple users at once.
        
        Args:
            all_users: List of user data dicts with "posts" field
        
        Returns:
            np.array: Bot probabilities for each user
        """
        if not self.is_fitted:
            raise ValueError("TFIDFBotScorer must be fitted before scoring")
        
        scores = []
        for user in all_users:
            score = self.score(user["posts"])
            scores.append(score)
        
        return np.array(scores)
    
    def save(self, filepath):
        """Save fitted scorer to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted TFIDFBotScorer")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'bot_prototype': self.bot_prototype,
            'human_prototype': self.human_prototype,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        print(f"TF-IDF scorer saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load fitted scorer from file."""
        model_data = joblib.load(filepath)
        
        # Create new instance
        scorer = TFIDFBotScorer()
        scorer.vectorizer = model_data['vectorizer']
        scorer.bot_prototype = model_data['bot_prototype']
        scorer.human_prototype = model_data['human_prototype']
        scorer.is_fitted = model_data['is_fitted']
        
        print(f"TF-IDF scorer loaded from {filepath}")
        return scorer


def train_tfidf_scorer(all_users, labels, max_features=5000):
    """
    Train a TF-IDF scorer on user data.
    
    Args:
        all_users: List of user data dicts with "posts" field
        labels: Array of labels (0=human, 1=bot)
        max_features: Maximum vocabulary size
    
    Returns:
        Fitted TFIDFBotScorer
    """
    scorer = TFIDFBotScorer(max_features=max_features)
    scorer.fit(all_users, labels)
    return scorer
