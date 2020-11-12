import librosa
import librosa.display
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

def visualize_mel_spectrogram(mel_spectrogram, title="Mel Spec"):
    """Visualize Mel Spectrogram\n
    Inputs:
      mel_spectrogram(tensor/array of shape: time, n_mels): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram.cpu(), ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()