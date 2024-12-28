import os
import joblib
from audio_classifier import extract_features  # Import from the previous script
import pandas as pd


def predict_audio(audio_path: str, model_path: str, scaler_path: str) -> float:
    """
    Make a prediction for a single audio file.

    Args:
        audio_path (str): Path to the audio file
        model_path (str): Path to the saved model
        scaler_path (str): Path to the saved scaler

    Returns:
        float: Prediction probability for the positive class
    """
    # Load the model and scaler
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Extract features
    features = extract_features(audio_path)

    # Convert to DataFrame
    X = pd.DataFrame([features])

    # Scale features
    X_scaled = scaler.transform(X)

    # Make prediction
    prob = clf.predict_proba(X_scaled)[0][1]

    return prob


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Predict using trained audio classifier')
    parser.add_argument('--audio_path', required=True,
                        help='Path to audio file or directory')
    parser.add_argument('--model_path', default='audio_classifier.joblib',
                        help='Path to saved model')
    parser.add_argument('--scaler_path', default='audio_scaler.joblib',
                        help='Path to saved scaler')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')

    args = parser.parse_args()

    if os.path.isfile(args.audio_path):
        # Single file prediction
        prob = predict_audio(args.audio_path, args.model_path, args.scaler_path)
        prediction = "Positive" if prob >= args.threshold else "Negative"
        print(f"\nFile: {args.audio_path}")
        print(f"Prediction: {prediction} (probability: {prob:.3f})")

    elif os.path.isdir(args.audio_path):
        # Directory prediction
        results = []
        for filename in os.listdir(args.audio_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(args.audio_path, filename)
                try:
                    prob = predict_audio(file_path, args.model_path, args.scaler_path)
                    prediction = "Positive" if prob >= args.threshold else "Negative"
                    results.append({
                        'file': filename,
                        'prediction': prediction,
                        'probability': prob
                    })
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

        # Create and save results DataFrame
        df = pd.DataFrame(results)
        output_path = 'predictions.csv'
        df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")

        # Print summary
        pos_count = sum(df['prediction'] == 'Positive')
        total = len(df)
        print(f"\nSummary:")
        print(f"Total files: {total}")
        print(f"Positive predictions: {pos_count}")
        print(f"Negative predictions: {total - pos_count}")


if __name__ == "__main__":
    main()