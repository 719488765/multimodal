# Machine Learning Scripts User Guide

This directory contains the scripts generated for the Multimodal Driver Emotion Analysis project. Follow the instructions below to use them.

## 1. Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```
(Make sure `torch`, `transformers`, `librosa`, `opencv-python`, `scipy`, `pandas`, `matplotlib`, `seaborn` are included)

## 2. Directory Structure

```
project/
├── data/
│   ├── train.csv           # Training set index
│   ├── val.csv             # Validation set index
│   ├── test.csv            # Test set index
│   ├── video/              # Video data
│   ├── audio/              # Audio data
│   ├── physiological/      # Physiological data
│   ├── text/               # Text data
│   ├── labels/             # Label data
│   ├── preprocess.py       # Data preprocessing script
│   └── dataset.py          # Dataset loader
├── scripts/
│   ├── train.py            # Main training/fine-tuning script
│   └── run_baselines.py    # Automation script for baseline experiments
└── utils/
    ├── recorder.py         # Experiment logging
    └── visualization.py    # Analysis visualization
```

## 3. Usage

### 3.1 Data Preprocessing

Use `data/preprocess.py` to inspect or test the preprocessing logic. The actual preprocessing is integrated into the `MultimodalDataset` class (in `data/dataset.py`).

To test preprocessing on a single file (requires modifying the `__main__` block in `preprocess.py`):
```bash
python data/preprocess.py
```

### 3.2 Model Training (Pre-training)

To train the model from scratch (pre-training) using the default configuration:

```bash
python scripts/train.py --config config/config.yaml --mode pretrain
```

**Arguments:**
- `--config`: Path to the YAML configuration file.
- `--mode`: `pretrain` or `finetune`.
- `--resume`: (Optional) Path to a checkpoint `.pth` file to resume training.

### 3.3 Model Fine-tuning

To fine-tune the model on a specific dataset (e.g., driver dataset), first ensure you have a pre-trained checkpoint, then run:

```bash
python scripts/train.py --config config/config.yaml --mode finetune --resume checkpoints/checkpoint_pretrain_best.pth
```
In fine-tuning mode, the script handles freezing backbone layers if configured in `config.yaml`.

### 3.4 Running Baselines

To automatically run a series of baseline experiments (e.g., single-modal vs multi-modal):

```bash
python scripts/run_baselines.py --config config/config.yaml --output_dir results/baselines
```

This will:
1. Generate temporary config files for each experiment.
2. Run training for each configuration.
3. Save a summary JSON and individual experiment logs.

## 4. Visualization

Visualization tools are located in `utils/visualization.py`. You can import them in your analysis notebooks or scripts to plot confusion matrices and training curves.

Example usage in a python script:
```python
from utils.visualization import plot_training_curves

plot_training_curves('logs/training_log.csv', 'results/plots/')
```

## 5. Troubleshooting

- **OOM Errors**: Reduce `batch_size` in `config.yaml` or `num_frames` for video.
- **Path Errors**: Ensure you are running scripts from the project root directory (e.g., `python scripts/train.py` not `cd scripts && python train.py`).
