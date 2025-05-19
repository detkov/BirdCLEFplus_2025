import inspect
import math
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join

import cv2
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


class CFG:    
    OUTPUT_DIR = './input/birdclef-2025-precomputed/01'
    DATA_ROOT = './input/birdclef-2025'
    FS = 32000
    
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)  
    MINMAX_NORM = True

    def to_dict(self):
        pr = {}
        for name in dir(self):
            value = getattr(self, name)
            if not name.startswith('__') and not inspect.ismethod(value):
                pr[name] = value
        return pr


def audio2melspec(audio_data):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)
        raise ValueError("NaN values found in audio data. Replaced with mean value.")

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=config.FS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if config.MINMAX_NORM:
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_db


def process_single_sample(row, config):
    audio_data, _ = librosa.load(row.filepath, sr=config.FS)
    target_samples = int(config.TARGET_DURATION * config.FS)

    if len(audio_data) < target_samples:
        n_copy = math.ceil(target_samples / len(audio_data))
        if n_copy > 1:
            audio_data = np.concatenate([audio_data] * n_copy)

    start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
    end_idx = min(len(audio_data), start_idx + target_samples)
    center_audio = audio_data[start_idx:end_idx]

    if len(center_audio) < target_samples:
        center_audio = np.pad(center_audio, 
                          (0, target_samples - len(center_audio)), 
                          mode='constant')

    mel_spec = audio2melspec(center_audio)

    if mel_spec.shape != config.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

    return row.samplename, mel_spec.astype(np.float32)

def process_data(config, working_df, total_samples, num_threads=None):
    if num_threads is None:
        # Use number of CPU cores if not specified
        num_threads = os.cpu_count()
    
    print(f"Starting audio processing using {num_threads} threads...")
    start_time = time.time()

    all_bird_data = {}
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks to the executor
        future_to_sample = {
            executor.submit(process_single_sample, row, config): row 
            for _, row in working_df.iterrows()
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_sample), total=total_samples):
            try:
                samplename, mel_spec = future.result()
                all_bird_data[samplename] = mel_spec
            except Exception as e:
                row = future_to_sample[future]
                print(f"Error processing {row.filepath}: {str(e)}")

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    return all_bird_data


if __name__ == "__main__":
    config = CFG()

    taxonomy_df = pd.read_csv(f'{config.DATA_ROOT}/taxonomy.csv')
    species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))

    train_df = pd.read_csv(f'{config.DATA_ROOT}/train.csv')

    label_list = sorted(train_df['primary_label'].unique())
    label_id_list = list(range(len(label_list)))
    label2id = dict(zip(label_list, label_id_list))
    id2label = dict(zip(label_id_list, label_list))

    working_df = train_df[['primary_label', 'rating', 'filename']].copy()
    working_df['target'] = working_df.primary_label.map(label2id)
    working_df['filepath'] = config.DATA_ROOT + '/train_audio/' + working_df.filename
    working_df['samplename'] = working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    working_df['class'] = working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))
    total_samples = len(working_df)

    all_bird_data = process_data(config, working_df, total_samples)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_filename = join(config.OUTPUT_DIR, "melspec.npy")
    config_filename = join(config.OUTPUT_DIR, "config.txt")

    with open(config_filename, 'w') as f:
        for key, value in config.to_dict().items():
            f.write(f"{key} = {value}\n")
    np.save(output_filename, all_bird_data)
    print(f"Mel spectrograms saved to {output_filename}")
    print(f"Config saved to {config_filename}")
