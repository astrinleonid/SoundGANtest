import os
import librosa
import soundfile as sf
import pandas as pd
from typing import List, Tuple


def process_audio_files(input_dir: str, output_dir: str, label: str, target_duration: float = 0.5) -> List[
    Tuple[str, str]]:
    """
    Process all WAV files in the input directory, truncate them to target duration,
    and save them to the output directory.

    Args:
        input_dir (str): Directory containing input WAV files
        output_dir (str): Directory where processed files will be saved
        label (str): Label to assign to all files
        target_duration (float): Target duration in seconds (default: 0.5)

    Returns:
        List[Tuple[str, str]]: List of tuples containing (filename, label)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    dataset = []

    # Process each WAV file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.wav'):
            # Load the audio file
            audio_path = os.path.join(input_dir, filename)
            try:
                # Load audio with a target sample rate of 22050 Hz
                y, sr = librosa.load(audio_path, sr=22050)

                # Calculate target number of samples
                target_samples = int(target_duration * sr)

                # Truncate or pad the audio to target duration
                if len(y) > target_samples:
                    y = y[:target_samples]
                else:
                    # Pad with zeros if audio is shorter than target duration
                    padding = target_samples - len(y)
                    y = np.pad(y, (0, padding), mode='constant')

                # Generate output filename
                output_filename = f"processed_{filename}"
                output_path = os.path.join(output_dir, output_filename)

                # Save the processed audio
                sf.write(output_path, y, sr)

                # Add to dataset
                dataset.append((output_filename, label))

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    return dataset


def create_dataset_csv(dataset: List[Tuple[str, str]], output_csv: str):
    """
    Create a CSV file from the processed dataset.

    Args:
        dataset (List[Tuple[str, str]]): List of (filename, label) tuples
        output_csv (str): Path to save the CSV file
    """
    df = pd.DataFrame(dataset, columns=['filename', 'label'])
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    data_path = "D:/repos/Data/PianoTriads1/piano_triads"
    dest_path = "D:/repos/Data/PianoDataset05/Pos1"
    process_audio_files(data_path, dest_path, 1, target_duration = 0.5)