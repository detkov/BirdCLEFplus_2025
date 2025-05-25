import argparse
import datetime
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import basename, dirname, exists, join, splitext

import cv2
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import Config
from src.utils import set_seed

warnings.filterwarnings("ignore")


def get_melspec_id(cfg):
    melspec_cfg_to_str = "-".join(map(lambda x: str(x).replace(" ", ""), [
            cfg.N_FFT, cfg.HOP_LENGTH, cfg.N_MELS, cfg.FMIN, cfg.FMAX, cfg.MINMAX_NORM,
            cfg.TARGET_DURATION, cfg.TARGET_SHAPE, cfg.FS
        ])
    )
    
    return melspec_cfg_to_str


def audio2melspec(audio_data, cfg):
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if cfg.MINMAX_NORM:
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_db


def process_single_sample(audio_path, cfg, crop_position="random"):
    audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
    target_samples = int(cfg.TARGET_DURATION * cfg.FS)

    if len(audio_data) < target_samples:
        audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)), mode='constant')
    
    # Calculate valid start index range
    max_start_idx = len(audio_data) - target_samples
    
    if crop_position == "start":
        start_idx = 0
    elif crop_position == "middle":
        start_idx = max(0, min(max_start_idx, int(len(audio_data) / 2 - target_samples / 2)))
    elif crop_position == "random":
        start_idx = np.random.randint(0, max_start_idx + 1)
    else:
        raise ValueError(f"Invalid crop_position: {crop_position}. Must be one of: start, middle, random")

    end_idx = start_idx + target_samples
    audio_data = audio_data[start_idx:end_idx]

    mel_spec = audio2melspec(audio_data, cfg)    

    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

    return mel_spec.astype(np.float32)


def process_data(cfg, num_threads=None, save=True):
    melspec_id = get_melspec_id(cfg)
    melspec_path = join(cfg.precomputerd_datadir, melspec_id, cfg.precomputed_filename)
    if exists(melspec_path):
        print(f"Mel spectrograms already exist at {melspec_path}. Skipping processing.")
        return np.load(melspec_path, allow_pickle=True).item()

    if num_threads is None:
        # Use number of CPU cores if not specified
        num_threads = os.cpu_count()
    
    df = pd.read_csv(cfg.train_csv)
    df['filepath'] = df['filename'].map(lambda x: join(cfg.train_datadir, x))
    df['samplename'] = df['filename'].map(lambda x: f'{dirname(x)}-{splitext(basename(x))[0]}' )

    print(f"Starting audio processing using {num_threads} threads...")
    start_time = time.time()

    spectrograms = {}
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks to the executor
        future_to_sample = {
            executor.submit(process_single_sample, row.filepath, cfg): (row.filepath, row.samplename)
            for _, row in df.iterrows()
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_sample), total=len(df)):
            try:
                audio_path, samplename = future_to_sample[future]
                mel_spec = future.result()
                spectrograms[samplename] = mel_spec
            except Exception as e:
                audio_path, _ = future_to_sample[future]
                print(f"Error processing {audio_path}: {str(e)}")

    end_time = time.time()
    print(f"Processing is completed in {end_time - start_time:.2f} seconds")
    
    if save:
        print(f"Saving mel spectrograms to {melspec_path}...")
        os.makedirs(dirname(melspec_path), exist_ok=True)
        np.save(melspec_path, spectrograms)
        print(f"Mel spectrograms are saved to {melspec_path}")

    return spectrograms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BirdCLEF model")
    parser.add_argument('-c', '--config', type=str, default='configs/debug.yaml', help='Path to config file')
    args = parser.parse_args()
    cfg = Config(args.config)
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%d-%m_%H-%M-%S")
    cfg.exp_name = f"{splitext(basename(args.config))[0]}_{now_str}"

    set_seed(cfg.seed)
    spectrograms = process_data(cfg)
