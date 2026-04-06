import argparse
from src.pipeline.evaluate_pipeline import evaluate, save_bot_ids
from src.pipeline.train_pipeline import run as train_model
from src.data.loader import load_bot_ids
from src.config import MODEL_PATH, PREDICTIONS_OUTPUT


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bot detection model on a dataset or build a new model"
    )
    parser.add_argument(
        "--input",
        help="Path to input JSON file (dataset.posts&users format)",
    )
    parser.add_argument(
        "--build",
        "--train",
        action="store_true",
        dest="build",
        help="Build/train a new model and save it to the configured model path",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=f"Path to trained model file (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        default=PREDICTIONS_OUTPUT,
        help=f"Path to output TXT file for bot IDs (default: {PREDICTIONS_OUTPUT})",
    )
    parser.add_argument(
        "--bot-ids",
        help="Optional: Path to bot IDs file for evaluation metrics",
    )

    args = parser.parse_args()

    if args.build:
        print("="*50)
        print("Building new bot detection model")
        print("="*50)
        train_model()
        print("\n" + "="*50)
        print("Model training complete!")
        print("="*50)
        return

    if not args.input:
        parser.error("--input is required unless --build/--train is used")

    bot_ids = None
    if args.bot_ids:
        bot_ids = load_bot_ids(args.bot_ids)

    print("="*50)
    print("Bot Detection Evaluation")
    print("="*50)
    predicted_bot_ids, metrics = evaluate(args.input, args.model, bot_ids)

    print("\nSaving predictions...")
    save_bot_ids(predicted_bot_ids, args.output)
    print(f"Predicted {len(predicted_bot_ids)} bots")
    print(f"Results saved to {args.output}")

    if metrics:
        print("\nEvaluation Metrics:")
        print(f"  Score:     {metrics['score']}")
        print(f"  F1:        {metrics['f1']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
