#!/usr/bin/env python
"""Quick integration test for Binoculars AI detection feature."""

from src.features.builder import extract_features

print("Testing Binoculars integration...")

# Create test data
test_data = {
    'user': {
        'id': 'test_user',
        'tweet_count': 10,
        'z_score': 0.5,
        'description': 'Test user for validation'
    },
    'posts': [
        {
            'text': 'This is a test post that is long enough for AI detection to process.',
            'created_at': '2024-01-01T12:00:00Z'
        },
        {
            'text': 'Another test post to verify the feature extraction works correctly.',
            'created_at': '2024-01-01T13:00:00Z'
        }
    ]
}

print("Extracting features...")
features = extract_features(test_data)

print(f"\nExtracted {len(features)} features:")
for key, value in features.items():
    print(f"  {key}: {value}")

print(f"\n✓ AI score feature present: {'avg_ai_score' in features}")
print(f"✓ AI score value: {features.get('avg_ai_score', 'N/A')}")
print("\nIntegration test complete!")
