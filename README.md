# BirdCLEFplus_2025
Solution for https://www.kaggle.com/competitions/birdclef-2025


## Installation

Place your Kaggle API token in the `~/.kaggle` directory. You can find instructions on how to do it [here](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).

```bash
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

This repository uses [uv](https://github.com/astral-sh/uv) package manager to install the required packages. For specific installation, please look [docs](https://docs.astral.sh/uv/getting-started/installation/#installation-methods). On Linux you can install it by running:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, you can install the required packages by running:

```bash
uv sync
source .venv/bin/activate
```

## Usage

### Dataset downloading

```bash
kaggle competitions download -p data  -c birdclef-2025
unzip data/birdclef-2025.zip -d data
```