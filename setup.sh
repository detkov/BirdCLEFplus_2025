curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
mkdir ~/.kaggle
mv test_file.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
mkdir input
kaggle competitions download -p input -c birdclef-2025
unzip input/birdclef-2025.zip -d input/birdclef-2025/
rm -rf ~/.kaggle