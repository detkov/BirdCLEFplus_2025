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
uv run python src/data_precomputation.py
```

### Training

```bash
uv run python src/train.py -c configs/default.yaml
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

[] Remove human voice [link1](https://www.kaggle.com/code/kdmitrie/bc25-separation-voice-from-data) [link2](https://www.kaggle.com/code/timothylovett/human-voice-removal-caution-around-ruther1)  
[] Remove `mel_spec_norm`  
[] Test padding audio if it is less than 5s instead of copying it [link](https://www.kaggle.com/code/shionao7/bird-25-submission-regnety008-v1)  
[] Maybe use TTA [link](https://www.kaggle.com/code/salmanahmedtamu/labels-tta-efficientnet-b0-pytorch-inference)  
[] Test `HOP_LENGTH` up to 16  
[] Test `FMIN` up to 20  
[] Test `FMAX` up to 16000    
[] Test `N_MELS` up to 128
[] Test model's `drop_rate` to something other than `0.2`  
[] Test model's `drop_path_rate` to something other than `0.2`  
[] Testa different model's classifier [link](https://www.kaggle.com/code/midcarryhz/lb-0-784-efficientnet-b0-pytorch-cpu)
```python
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
```  
[] Test audio denoising [link](https://www.kaggle.com/code/midcarryhz/lb-0-784-efficientnet-b0-pytorch-cpu/notebook)  
[] Test `FocalLossBCE` [link](https://www.kaggle.com/code/hideyukizushi/bird25-onlyinf-v2-s-focallossbce-cv-962-lb-829)  
[] Make prediction based on all 5s segments of the audio [link](https://www.kaggle.com/code/stefankahl/birdclef-2025-sample-submission)  
[] Add albumentations [link](https://www.kaggle.com/code/gopidurgaprasad/audio-augmentation-albumentations)  
[] Test extracting not the center 5 seconds, bu the first 5 seconds