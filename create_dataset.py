from Pianoid.HarmonicSimulator import HarmonicSimulation

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
from typing import List, Dict


def generate_harmonic_params(base_freq: float, num_overtones: int = 15) -> List[Dict]:
    """
    Generate parameters for base harmonic and its overtones.

    Args:
        base_freq (float): Frequency of the base harmonic
        num_overtones (int): Number of overtone harmonics to generate

    Returns:
        List[Dict]: List of parameter dictionaries for all harmonics
    """
    params = []

    # Base harmonic parameters
    base_params = {
        'frequency': base_freq,
        'amplitude': 1.0,
        'phase': 0.0,
        'decay': 0.1,
        'delay': 0.0
    }
    params.append(base_params)

    # Generate overtone parameters
    for _ in tqdm(range(num_overtones)):
        # Random frequency between base_freq and 10kHz
        freq = np.random.uniform(base_freq, 10000)
        # Random amplitude between 0 and 1
        amp = np.random.uniform(0, 1)
        # Random decay greater than 0.1
        decay = np.random.uniform(0.1, 1.0)

        overtone_params = {
            'frequency': freq,
            'amplitude': amp,
            'phase': 0.0,
            'decay': decay,
            'delay': 0.0
        }
        params.append(overtone_params)

    return params


def generate_samples(
        num_samples: int,
        freq_range: tuple,
        output_dir: str,
        duration: float = 0.5,
        sample_rate: int = 48000
):
    """
    Generate multiple harmonic sound samples and save them as WAV files.

    Args:
        num_samples (int): Number of samples to generate
        freq_range (tuple): (min_freq, max_freq) for base harmonic
        output_dir (str): Directory to save WAV files
        duration (float): Duration of each sample in seconds
        sample_rate (int): Sample rate for audio generation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    min_freq, max_freq = freq_range

    for i in range(num_samples):
        # Generate random base frequency
        base_freq = np.random.uniform(min_freq, max_freq)

        # Generate parameters for all harmonics
        params = generate_harmonic_params(base_freq)

        # Create harmonic simulation
        sim = HarmonicSimulation(params=params)

        # Generate sound
        sound = sim.generate(duration=duration, sample_rate=sample_rate)

        # Normalize the sound to prevent clipping
        sound = sound / np.max(np.abs(sound))

        # Save as WAV file
        filename = os.path.join(output_dir, f'sample_{i:04d}_{base_freq:.1f}Hz.wav')
        sf.write(filename, sound, sample_rate)

        print(f"Generated sample {i + 1}/{num_samples}: {filename}")

if __name__ == "__main__":
    generate_samples(
        100,
        [55, 2000],
        "D:/repos/Data/PianoDataset05/Neg1",
        duration = 0.5,
        sample_rate = 48000
    )
