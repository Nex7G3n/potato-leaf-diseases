# Potato Leaf Disease Training

This repository provides a simple script to train a convolutional neural network on the [Potato Leaf Disease Dataset](https://www.kaggle.com/datasets/warcoder/potato-leaf-disease-dataset).

## Setup

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Authenticate the Kaggle API by placing your `kaggle.json` credentials file under `~/.kaggle/`.
3. Run the training script:
   ```bash
   python train.py --data-dir data --epochs 10
   ```

The dataset will be downloaded automatically the first time you run the script.
