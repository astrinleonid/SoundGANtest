import numpy as np
import sounddevice as sd

from Pianoid import HarmonicSimulator

def generate_samples(number_of_samples, frequency_range):

    sim = HarmonicSimulator()