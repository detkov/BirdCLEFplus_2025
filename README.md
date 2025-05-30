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
kaggle datasets download kdmitrie/bc25-separation-voice-from-data-by-silero-vad -p input/voice_data --unzip
```

If you wish to do manual data precomputation, you can run the following command:
```bash
uv run python -m src.audio_processing
```

### Training

```bash
uv run python -m src.train -c configs/your_config.yaml
```

### Upload models to Kaggle 
If you do not have the dataset `birdclefplus-2025-models` on Kaggle, you can create it by running the following commands.
```bash
kaggle datasets init -p birdclefplus-2025-models
```
Then fix the `birdclefplus-2025-models/dataset-metadata.json`. After that, run the following commands to upload your models: 
```bash
kaggle datasets create -p birdclefplus-2025-models
kaggle datasets version -p birdclefplus-2025-models -m "Update"
```  

Or, if the dataset already exists, you can use the following command to update it:
```bash
kaggle datasets download <your_username>/birdclefplus-2025-models -p ./birdclefplus-2025-models --unzip
kaggle datasets init -p birdclefplus-2025-models
```
Then fix the `birdclefplus-2025-models/dataset-metadata.json`. After that, run the following commands to upload your models: 
```bash
kaggle datasets version -p birdclefplus-2025-models -m "Update"
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

### 2025-05-21
* Conducted experiments: `001`, `002`, `003`, `004`.  
### 2025-05-22
* Conducted experiments: `005`, `006`.
### 2025-05-25
* `N_MELS` has to be 256!!! Because it translates into the height of the meplspec image.  
* Changed short audio processing from copying to constant padding  

## Experiments 

* `001-1.yaml` — `001-8.yaml` are related to the melspec settings. The best ones are: 
```
N_FFT: 2048
HOP_LENGTH: 1024 (but 512 on Public LB)
N_MELS: 128
FMIN: 50 (but 20 on Public LB)
FMAX: 14000 (but 16000 on Public LB)
MINMAX_NORM: true
```
* `002-1.yaml` — `002-7.yaml` are related to the optimizer scheduler settings and batch size. 
  * Larger BS (32 -> 128) can yield better results — need to investigate further
  * `weight_decay: 1.0e-2` is better than `weight_decay: 1.0e-5`
  * `OneCycleLR` is really bad
  * `ReduceLROnPlateau` can also get good results
* `003-13.yaml`, `003-17.yaml`, `003-18.yaml` are changing resolution of `001-3`, `001-7`, `001-8` experiments from `256x256` to `224x224`
  * Based on both Local AUC and Public AUC, `256x256` is a better choice
* `004-13.yaml`, `004-17.yaml`, `004-18.yaml` are changing number of epochs from 10 to 15 and `min_lr: 1.0e-6` to `min_lr: 1.0e-7`.
  * Based on both Local AUC and Public AUC, it is not clear whether these changes actually improve generalization ability, but it's clear that they overfit much worse
* `005-1.yaml` — `005-10.yaml` are changing `lr` from `1.0e-3` to `1.0e-2` with the step of `0.1`.
  * `lr: 3.0e-3` is the best one
* `006-1.yaml` — `006-4.yaml` are changing `in_channels` from `1` to `3` and `pretrained` from `True` to `False`
  * `in_channels: 3` with ImageNet normaliation qorks fine
* `011-1.yaml` — `011-3.yaml` are changing voice processing. `1` is not filtering out human voice, `2` is filtering out human voice and leave only the longest segment without voice, `3` is filtering out human voice and leaving concatenated segments without voice.
  * `1` is the best one based on Local AUC.
* `012-*` are changing the base model from `EfficientNet0` to `1`, `2`, `3` and `4` with `1` and `3` input channels. Other training parameters are the same.
  * Only `EfficientNet3` with 3 channels is better than `EfficientNet0` with 1 channel. 
* `013-a*` are changing the `aug_prob` and `mixup_alpha` parameters. 
  * Turns out that different values of `mixup_alpha` do not change anything. Only `0` value can change, but I have never tested it.
  * Also, `aug_prob` of `0.5` is the best.
* `013-d*` are changing the `drop_rate` and `drop_path_rate` parameters. 
  * `drop_rate: 0.5` and `drop_path_rate: 0.2` are the best ones. The next is `drop_rate: 0.2` and `drop_path_rate: 0.35`.
* `014-*` are changing the `precompute_data` parameter and adds all 5 folds to the training.
  * `precompute_data` param does not change anything, but the speed (x30) of training.
  * All 5 folds have 5 different results, from `0.94544` to `0.95592`.
* `015-*` are changing the `BCEWithLogitsLoss` to `FocalLossBCE` with different params.
  * `BCEWithLogitsLoss` seems to be the best single.
  * `FocalLossBCE`'s reduction `sum` is worse than `mean`.
  * `FocalLossBCE`'s gamma `3` is worse than `2`.
### Results

| Experment name, fold | Local AUC | Public AUC | Details |
|---|---|---|---|
| 001-1, 0  | 0.94536 | 0.747 | - |
| 001-2, 0  | 0.94777 | - | - |
| 001-3, 0  | 0.95217 | 0.751 | - |
| 001-4, 0  | 0.94892 | - | - |
| 001-5, 0  | 0.94621 | - | - |
| 001-6, 0  | 0.94896 | - | - |
| 001-7, 0  | 0.95190 | 0.779 | - |
| 001-8, 0  | 0.95055 | - | - |
| 002-1, 0  | 0.94536 | - | - |
| 002-2, 0  | 0.94487 | - | - |
| 002-3, 0  | 0.94789 | - | - |
| 002-4, 0  | 0.94842 | - | - |
| 002-5, 0  | 0.94152 | - | - |
| 002-6, 0  | 0.94143 | - | - |
| 002-7, 0  | 0.94771 | - | - |
| 003-13, 0 | 0.95079 | - | - |
| 003-17, 0 | 0.95095 | 0.774 | - |
| 003-18, 0 | 0.94993 | - | - |
| 004-13, 0 | 0.95005 | - | - |
| 004-17, 0 | 0.95103 | 0.765 | - |
| 004-18, 0 | 0.95146 | - | - |
| 005-1, 0  | 0.94534 | - | - |
| 005-2, 0  | 0.94883 | - | - |
| 005-3, 0  | 0.95130 | - | - |
| 005-4, 0  | 0.94950 | - | - |
| 005-5, 0  | 0.94731 | - | - |
| 005-6, 0  | 0.93399 | - | - |
| 005-7, 0  | 0.93703 | - | - |
| 005-8, 0  | 0.92751 | - | - |
| 005-9, 0  | 0.91582 | - | - |
| 005-10, 0 | 0.91446 | - | - |
| 006-1, 0 | 0.95190 | - | - |
| 006-2, 0 | 0.92699 | - | - |
| 006-3, 0 | 0.92410 | - | - |
| 006-4, 0 | 0.95110 | - | - |
| 011-1, 0 | 0.95403 | 0.732 | - |
| 011-2, 0 | 0.95004 | 0.762 | - |
| 011-3, 0 | 0.94818 | 0.763 | - |
| 012-10, 0 | 0.95403 | - | - |
| 012-11, 0 | 0.85695 | - | - |
| 012-12, 0 | 0.94916 | - | - |
| 012-13, 0 | 0.95021 | - | - |
| 012-14, 0 | 0.94993 | - | - |
| 012-30, 0 | 0.94919 | - | - |
| 012-31, 0 | 0.90809 | - | - |
| 012-32, 0 | 0.95438 | 0.746 | - |
| 012-33, 0 | 0.95227 | - | - |
| 012-34, 0 | 0.95086 | 0.742 | - |
| 013-a0, 0 | 0.95162 | - | - |
| 013-a5, 0 | 0.94826 | - | - |
| 013-a10, 0 | 0.95403 | - | - |
| 013-a15, 0 | 0.95085 | - | - |
| 013-a20, 0 | 0.95222 | - | - |
| 013-d0, 0 | 0.94865 | - | - |
| 013-d1, 0 | 0.94600 | - | - |
| 013-d2, 0 | 0.95319 | - | - |
| 013-d3, 0 | 0.95376 | - | - |
| 013-d4, 0 | 0.95186 | - | - |
| 013-d5, 0 | 0.94825 | - | - |
| 013-d6, 0 | 0.94824 | - | - |
| 013-d7, 0 | 0.95255 | - | - |
| 013-d8, 0 | 0.95196 | - | - |
| 013-d9, 0 | 0.95023 | - | - |
| 013-d10, 0 | 0.94682 | - | - |
| 013-d11, 0 | 0.95443 | 0.759 | - |
| 013-d12, 0 | 0.95529 | 0.764 | - |
| 013-d13, 0 | 0.95204 | - | - |
| 013-d14, 0 | 0.94606 | - | - |
| 013-d15, 0 | 0.94986 | - | - |
| 013-d16, 0 | 0.95272 | - | - |
| 013-d17, 0 | 0.95022 | - | - |
| 013-d18, 0 | 0.95159 | - | - |
| 013-d19, 0 | 0.94326 | - | - |
| 013-d20, 0 | 0.95247 | - | - |
| 013-d21, 0 | 0.95592 | 0.773 | - |
| 013-d22, 0 | 0.95044 | - | - |
| 013-d23, 0 | 0.95028 | - | - |
| 015-1, 0 | 0.95592 | - | - |
| 015-2, 0 | 0.94985 | - | - |
| 015-3, 0 | 0.9462 | - | - |
| 015-4, 0 | error | - | - |
| 015-5, 0 | 0.94688 | - | - |
| 015-6, 0 | 0.95063 | - | - |
| 015-7, 0 | 0.94945 | - | - |
| 015-8, 0 | 0.95122 | - | - |
| 015-9, 0 |  | - | - |

### Hypotheses

- [x] Remove human voice [link1](https://www.kaggle.com/code/kdmitrie/bc25-separation-voice-from-data) [link2](https://www.kaggle.com/code/timothylovett/human-voice-removal-caution-around-ruther1)  
- [ ] Remove `mel_spec_norm`  
- [x] Test padding audio if it is less than 5s instead of copying it [link](https://www.kaggle.com/code/shionao7/bird-25-submission-regnety008-v1)  
- [ ] Maybe use TTA [link](https://www.kaggle.com/code/salmanahmedtamu/labels-tta-efficientnet-b0-pytorch-inference)  
- [ ] Test `HOP_LENGTH` up to 16  
- [x] Test `FMIN` up to 20  
- [x] Test `FMAX` up to 16000    
- [ ] Test `N_MELS` up to 128
- [ ] Test 
- [x] Test model's `drop_rate` to something other than `0.2`  
- [x] Test model's `drop_path_rate` to something other than `0.2`  
- [ ] Testa different model's classifier [link](https://www.kaggle.com/code/midcarryhz/lb-0-784-efficientnet-b0-pytorch-cpu)
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
- [ ] Test audio denoising [link](https://www.kaggle.com/code/midcarryhz/lb-0-784-efficientnet-b0-pytorch-cpu/notebook)  
- [x] Test `FocalLossBCE` [link](https://www.kaggle.com/code/hideyukizushi/bird25-onlyinf-v2-s-focallossbce-cv-962-lb-829)  
- [ ] Make prediction based on all 5s segments of the audio [link](https://www.kaggle.com/code/stefankahl/birdclef-2025-sample-submission)  
- [ ] Add albumentations  
- [x] Test extracting not the center 5 seconds, but the first 5 seconds
- [x] Test 3 channels
- [x] Test ImageNet normalization for 3 channels if the weights are pretrained `T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),`
- [ ] Test melspec more thoroughly (`N_MELS`, `HOP_LENGTH`)
- [ ] 
