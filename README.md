# BirdCLEFplus_2025
Solution for https://www.kaggle.com/competitions/birdclef-2025


## Installation
This repository uses [uv](https://github.com/astral-sh/uv) package manager to install the required packages. For specific installation, please look [docs](https://docs.astral.sh/uv/getting-started/installation/#installation-methods). On Linux you can install and synchronize it by running:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Usage

Place your Kaggle API token in the `~/.kaggle` directory. You can find instructions on how to do it [here](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).

```bash
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Dataset download and processing

```bash
kaggle competitions download -p input -c birdclef-2025
unzip input/birdclef-2025.zip -d input/birdclef-2025/
```

```bash
uv run python scripts/data_precomputation.py
```

## Journal

### 2025-05-18
Lookead at different `librosa.feature.melspectrogram` params:
* Changed `N_FFT` to `2048` from `1024`
* Changed `HOP_LENGTH` to `512` from `256`
* Changed `FMIN` from `20` to `50`
* `FMAX` seems to be better at `15000`, not `14000`
These changes were made due to the seemingly clearer and full image. However, we have to investigate it with training and validation. 

### 2025-05-19
* Tested mel spectrograms precomputation — indeed, it increased the training speed.  

## Hypotheses

[] Remove human voice https://www.kaggle.com/code/kdmitrie/bc25-separation-voice-from-data https://www.kaggle.com/code/timothylovett/human-voice-removal-caution-around-ruther1  
[] Do we need `mel_spec_norm`?  
[] Shall we pad audio if it is less than 5s instead of copying it?  
