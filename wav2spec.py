import io

import numpy as np
from PIL import Image
import pydub
from scipy.io import wavfile
import torch
import torchaudio
import argparse
import easygui as eg
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",
                    help="Input file to process, anything that FFMPEG supports, but wav and mp3 are recommended")
parser.add_argument("-o", "--output", help="Output Image")
parser.add_argument("-m", "--maxvol", default=100, help="Max Volume, 255 for identical results")
parser.add_argument("-p", "--powerforimage", default=0.25, help="Power for Image")
parser.add_argument("-n", "--nmels", default=512,
                    help="n_mels to use for Image, basically width. Higher = more fidelity")
args = parser.parse_args()


def spectrogram_image_from_wav(wav_bytes: io.BytesIO, max_volume: float = 50, power_for_image: float = 0.25,
                               ms_duration: int = 5119) -> Image.Image:
    """
    Generate a spectrogram image from a WAV file.
    """
    # Read WAV file from bytes
    sample_rate, waveform = wavfile.read(wav_bytes)

    # sample_rate = 44100  # [Hz]
    clip_duration_ms = ms_duration  # [ms]

    bins_per_image = 512
    n_mels = int(args.nmels)
    mel_scale = True

    # FFT parameters
    window_duration_ms = 100  # [ms]
    padded_duration_ms = 400  # [ms]
    step_size_ms = 10  # [ms]

    # Derived parameters
    num_samples = int(512 / float(bins_per_image) * clip_duration_ms) * sample_rate
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    # Compute spectrogram from waveform
    Sxx = spectrogram_from_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        mel_scale=mel_scale,
        n_mels=n_mels,
    )

    # Convert spectrogram to image
    image = image_from_spectrogram(Sxx, max_volume=max_volume, power_for_image=power_for_image)

    return image


def spectrogram_from_waveform(
        waveform: np.ndarray,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        mel_scale: bool = True,
        n_mels: int = 512,
) -> np.ndarray:
    """
    Compute a spectrogram from a waveform.
    """

    spectrogram_func = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        power=None,
        hop_length=hop_length,
        win_length=win_length,
    )

    waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).reshape(1, -1)
    Sxx_complex = spectrogram_func(waveform_tensor).numpy()[0]

    Sxx_mag = np.abs(Sxx_complex)

    if mel_scale:
        mel_scaler = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
        )

        Sxx_mag = mel_scaler(torch.from_numpy(Sxx_mag)).numpy()

    return Sxx_mag


def image_from_spectrogram(
        data: np.ndarray,
        max_volume: float = 50,
        power_for_image: float = 0.25
) -> Image.Image:
    data = np.power(data, power_for_image)
    data = data / (max_volume / 255)
    data = 255 - data
    data = data[::-1]
    image = Image.fromarray(data.astype(np.uint8))
    return image


def spectrogram_image_from_file(filename, max_volume: float = 50, power_for_image: float = 0.25) -> Image.Image:
    """
    Generate a spectrogram image from an MP3 file.
    """

    max_volume = int(max_volume)
    power_for_image = float(args.powerforimage)

    # Load MP3 file into AudioSegment object
    audio = pydub.AudioSegment.from_file(filename)

    # Convert to mono and set frame rate
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(44100)

    length_in_ms = len(audio)
    print("ORIGINAL AUDIO LENGTH IN MS:", length_in_ms)
    # Extract first 5 seconds of audio data
    audio = audio[:5119]
    length_in_ms = len(audio)
    print("CROPPED AUDIO LENGTH IN MS:", length_in_ms)

    # Convert to WAV and save as BytesIO object
    wav_bytes = io.BytesIO()
    audio.export("clip.wav", format="wav")
    audio.export(wav_bytes, format="wav")
    wav_bytes.seek(0)

    # Generate spectrogram image from WAV file
    return spectrogram_image_from_wav(wav_bytes, max_volume=max_volume, power_for_image=power_for_image,
                                      ms_duration=length_in_ms)


def convert(audio):
    filetype = audio.split(".")[-1]
    if filetype == "mp3":
        print("MP3")
        image = spectrogram_image_from_file(audio,50)
    elif filetype == "wav":
        print("WAV")
        image = spectrogram_image_from_wav(audio,100)

    return image


audio = eg.fileopenbox('Select a file', 'Select a file')
image = convert(audio)
filename = f'spec_{time.time()}.png'
filedir = os.path.join('logs','music','spectrograms')
if not os.path.exists(filedir):
    os.makedirs(filedir)
image.save(os.path.join(filedir,filename))

