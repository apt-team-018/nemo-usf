# Parakeet-TDT-0.6B v2 Fine-tuning Guide

Complete guide for fine-tuning NVIDIA's `parakeet-tdt-0.6b-v2` ASR model using NeMo on single or multiple GPUs.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
  - [Single GPU Training](#single-gpu-training)
  - [Multi-GPU Training (4× H100)](#multi-gpu-training-4-h100)
- [Monitoring](#monitoring)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware
- **Recommended**: H100 80GB, A100 80GB, or similar high-end GPU
- **Minimum**: GPU with 24GB+ VRAM
- **Multi-GPU**: 4× H100 recommended for optimal performance

### Software
- Python 3.10 or 3.11 (avoid 3.12)
- CUDA 12.1 or 12.4
- Ubuntu 20.04 / 22.04 or similar Linux distribution

---

## Installation

### 1. Create Python Environment

Using **conda** (recommended):

```bash
conda create -n nemo-asr python=3.10 -y
conda activate nemo-asr
```

Or using **venv**:

```bash
python3.10 -m venv nemo-asr
source nemo-asr/bin/activate
```

### 2. Install PyTorch with CUDA

For CUDA 12.4:

```bash
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

For CUDA 12.1:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```bash
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

### 3. Install NeMo and Dependencies

```bash
pip install 'pytorch-lightning>=2.3.0,<=2.4.0'
pip install 'nemo_toolkit[asr]'
pip install datasets==2.19.0 soundfile librosa
pip install fsspec==2024.12.0
```

Verify NeMo installation:

```bash
python -c "import nemo; print('NeMo version:', nemo.__version__)"
```

Expected output:
```
NeMo version: 2.5.3
```

---

## Dataset Preparation

### Option 1: Using HuggingFace Datasets

#### Method A: Direct from Jupyter/Python (Used in this tutorial)

Run this in a Jupyter notebook or Python script:

```python
# First, downgrade datasets to avoid torchcodec issues
import subprocess
# subprocess.run(["pip", "install", "datasets==2.19.0", "--quiet"])

from datasets import load_dataset
import soundfile as sf
import json
from pathlib import Path
import numpy as np

# Load dataset
dataset_name = "librispeech_asr"  # Replace with your dataset
print("Loading dataset...")
dataset = load_dataset(dataset_name, "clean", split="train.100")

# Output directories
base_dir = Path("hf_nemo_demo")  # Change to your preferred path
audio_dir = base_dir / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

train_manifest = base_dir / "train_manifest.json"
val_manifest = base_dir / "val_manifest.json"

# Split dataset (800 train, 100 val)
train_samples = dataset.select(range(0, 800))
val_samples = dataset.select(range(800, 900))

def write_manifest(split_ds, manifest_path):
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i in range(len(split_ds)):
            ex = split_ds[i]
            
            # Get audio data
            audio = np.array(ex["audio"]["array"])
            sr = ex["audio"]["sampling_rate"]
            text = ex["text"]

            # Save as WAV
            wav_path = audio_dir / f"{manifest_path.stem}_{i}.wav"
            sf.write(wav_path, audio, sr)

            # Calculate duration
            duration = float(len(audio) / sr)
            
            # Create manifest entry
            entry = {
                "audio_filepath": str(wav_path.resolve()),
                "duration": duration,
                "text": text,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            if i % 100 == 0:
                print(f"Processed {i} samples")

print("Creating train manifest...")
write_manifest(train_samples, train_manifest)
print("Creating val manifest...")
write_manifest(val_samples, val_manifest)

print(f"\nTrain manifest: {train_manifest}")
print(f"Val manifest: {val_manifest}")
print(f"Total WAV files: {len(list(audio_dir.glob('*.wav')))}")
```

**Output:**
```
Loading dataset...
Creating train manifest...
Processed 0 samples
Processed 100 samples
Processed 200 samples
...
Creating val manifest...
Processed 0 samples

Train manifest: hf_nemo_demo/train_manifest.json
Val manifest: hf_nemo_demo/val_manifest.json
Total WAV files: 900
```

#### Method B: Standalone Script

Create a file `prepare_data_hf.py`:

```python
from datasets import load_dataset
import soundfile as sf
import json
from pathlib import Path
import numpy as np

# Load dataset
dataset_name = "librispeech_asr"  # Replace with your dataset
dataset = load_dataset(dataset_name, "clean", split="train.100")

# Output directories
base_dir = Path("data/nemo_format")
audio_dir = base_dir / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

train_manifest = base_dir / "train_manifest.json"
val_manifest = base_dir / "val_manifest.json"

# Split dataset
train_samples = dataset.select(range(0, 800))
val_samples = dataset.select(range(800, 900))

def write_manifest(split_ds, manifest_path):
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i in range(len(split_ds)):
            ex = split_ds[i]
            audio = np.array(ex["audio"]["array"])
            sr = ex["audio"]["sampling_rate"]
            text = ex["text"]

            # Save as WAV
            wav_path = audio_dir / f"{manifest_path.stem}_{i}.wav"
            sf.write(wav_path, audio, sr)

            duration = float(len(audio) / sr)
            entry = {
                "audio_filepath": str(wav_path.resolve()),
                "duration": duration,
                "text": text,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            if i % 100 == 0:
                print(f"Processed {i} samples")

print("Creating train manifest...")
write_manifest(train_samples, train_manifest)
print("Creating val manifest...")
write_manifest(val_samples, val_manifest)

print(f"\nTrain manifest: {train_manifest}")
print(f"Val manifest: {val_manifest}")
```

Run:

```bash
python prepare_data_hf.py
```

### Option 2: Medical Dataset (5techlab-research/asr_60)

#### Quick Test (100 samples for testing)

Use this for quick testing with 80 train + 20 val samples:

```python
from datasets import load_dataset
import soundfile as sf
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

print("datasets version:", __import__('datasets').__version__)

print("\nLoading 5techlab-research/asr_60 in streaming mode...")
dataset = load_dataset(
    "5techlab-research/asr_60",
    split="train",
    streaming=True
)

# Setup
base_dir = Path("medical_nemo_data")
audio_dir = base_dir / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

train_file = open(base_dir / "train_manifest.json", "w")
val_file = open(base_dir / "val_manifest.json", "w")

# Counters
train_count = 0
val_count = 0
skip_empty = 0
skip_audio = 0
skip_error = 0

# TARGETS
TARGET_TRAIN = 80
TARGET_VAL = 20
TARGET_TOTAL = TARGET_TRAIN + TARGET_VAL

# Progress bar
pbar = tqdm(
    total=TARGET_TOTAL,
    desc="Creating test dataset",
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
)

print(f"\nTarget: {TARGET_TRAIN} train + {TARGET_VAL} val = {TARGET_TOTAL} samples")
print("Processing will stop as soon as targets are reached\n")

samples_checked = 0

for i, ex in enumerate(dataset):
    # Update status
    status = f"train:{train_count}/{TARGET_TRAIN} val:{val_count}/{TARGET_VAL} skip:{skip_empty+skip_audio+skip_error}"
    pbar.set_postfix_str(status)
    
    try:
        samples_checked += 1
        
        # Stop if we have enough
        if train_count >= TARGET_TRAIN and val_count >= TARGET_VAL:
            tqdm.write("✓ Targets reached! Stopping...")
            break
        
        # Check text
        text = ex.get("transcription", "")
        if not text or not text.strip():
            skip_empty += 1
            tqdm.write(f"⚠ Sample {samples_checked}: skipped (empty text)")
            continue
        
        # Check audio
        audio = np.array(ex["audio"]["array"])
        sr = ex["audio"]["sampling_rate"]
        
        if len(audio) == 0 or sr == 0:
            skip_audio += 1
            tqdm.write(f"⚠ Sample {samples_checked}: skipped (invalid audio)")
            continue
        
        # Assign to train or val
        if train_count < TARGET_TRAIN:
            wav_path = audio_dir / f"train_{train_count}.wav"
            file = train_file
            train_count += 1
            label = "TRAIN"
        elif val_count < TARGET_VAL:
            wav_path = audio_dir / f"val_{val_count}.wav"
            file = val_file
            val_count += 1
            label = "VAL"
        else:
            continue
        
        # Save
        sf.write(wav_path, audio, sr)
        file.write(json.dumps({
            "audio_filepath": str(wav_path.resolve()),
            "duration": float(len(audio) / sr),
            "text": text.strip()
        }, ensure_ascii=False) + "\n")
        
        # Update progress
        pbar.update(1)
        
        # Log every 10 samples
        if (train_count + val_count) % 10 == 0:
            tqdm.write(f"✓ {label} sample #{train_count if label=='TRAIN' else val_count} saved")
    
    except Exception as e:
        skip_error += 1
        tqdm.write(f"⚠ Sample {samples_checked}: ERROR - {str(e)[:60]}")
        continue

pbar.close()
train_file.close()
val_file.close()

# Final summary
print("\n" + "="*70)
print("✓ DATASET CREATION COMPLETE")
print("="*70)
print(f"{'Samples checked:':<25} {samples_checked:>6}")
print(f"{'Valid train samples:':<25} {train_count:>6} / {TARGET_TRAIN}")
print(f"{'Valid val samples:':<25} {val_count:>6} / {TARGET_VAL}")
print("-"*70)
print(f"{'Skipped (empty text):':<25} {skip_empty:>6}")
print(f"{'Skipped (bad audio):':<25} {skip_audio:>6}")
print(f"{'Skipped (errors):':<25} {skip_error:>6}")
print(f"{'Total skipped:':<25} {skip_empty+skip_audio+skip_error:>6}")
print("="*70)
print(f"Train manifest: {base_dir / 'train_manifest.json'}")
print(f"Val manifest:   {base_dir / 'val_manifest.json'}")
print(f"Audio files:    {audio_dir}/")
print("="*70)
```

#### Full Dataset (98% train, 2% validation)

For the complete ~292k samples dataset with 98/2 split:

```python
from datasets import load_dataset
import soundfile as sf
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

print("Loading 5techlab-research/asr_60 (FULL DATASET)...")
print("⏳ Split: 98% train, 2% validation (dynamic)\n")

dataset = load_dataset(
    "5techlab-research/asr_60",
    split="train",
    streaming=True
)

print("✓ Starting to process ALL samples with 98/2 split...\n")

base_dir = Path("medical_nemo_data_full")
audio_dir = base_dir / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

train_file = open(base_dir / "train_manifest.json", "w")
val_file = open(base_dir / "val_manifest.json", "w")

train_count = 0
val_count = 0
skip_count = 0
total_processed = 0

# No fixed target - process all samples
pbar = tqdm(desc="Processing samples", unit=" samples")

for i, ex in enumerate(dataset):
    try:
        # Get text
        text = ex.get("transcription", "").strip()
        if not text:
            skip_count += 1
            continue
        
        # Get audio
        audio = np.array(ex["audio"]["array"])
        sr = ex["audio"]["sampling_rate"]
        
        if len(audio) == 0 or sr == 0:
            skip_count += 1
            continue
        
        # Dynamic 98/2 split: Every 50th valid sample goes to val
        if total_processed % 50 == 49:  # 2% (1 out of 50)
            wav_path = audio_dir / f"val_{val_count}.wav"
            file = val_file
            val_count += 1
        else:  # 98% (49 out of 50)
            wav_path = audio_dir / f"train_{train_count}.wav"
            file = train_file
            train_count += 1
        
        total_processed += 1
        
        # Save
        sf.write(wav_path, audio, sr)
        file.write(json.dumps({
            "audio_filepath": str(wav_path.resolve()),
            "duration": float(len(audio) / sr),
            "text": text
        }, ensure_ascii=False) + "\n")
        
        # Update progress
        pbar.update(1)
        pbar.set_postfix(
            total=f"{total_processed:,}",
            train=f"{train_count:,}",
            val=f"{val_count:,}",
            skip=skip_count,
            train_pct=f"{100*train_count/max(total_processed,1):.1f}%"
        )
        
        # Save checkpoints every 10k samples
        if total_processed % 10000 == 0:
            train_file.flush()
            val_file.flush()
        
    except Exception as e:
        skip_count += 1
        continue

pbar.close()
train_file.close()
val_file.close()

# Calculate actual percentages
train_pct = 100 * train_count / total_processed if total_processed > 0 else 0
val_pct = 100 * val_count / total_processed if total_processed > 0 else 0

print(f"\n{'='*70}")
print(f"✓ FULL DATASET PROCESSING COMPLETE!")
print(f"{'='*70}")
print(f"Total processed:  {total_processed:,}")
print(f"Train samples:    {train_count:,} ({train_pct:.2f}%)")
print(f"Val samples:      {val_count:,} ({val_pct:.2f}%)")
print(f"Skipped:          {skip_count:,}")
print(f"{'='*70}")
print(f"Train manifest: {base_dir / 'train_manifest.json'}")
print(f"Val manifest:   {base_dir / 'val_manifest.json'}")
print(f"Audio files:    {audio_dir}/")
print(f"{'='*70}")
```

**Notes:**
- Test dataset takes ~5-10 minutes to create (100 samples)
- Full dataset takes ~2-3 hours to create (~292k samples)
- Run full dataset preparation overnight or in a `tmux`/`screen` session
- First load checks 1433 data files (slow), subsequent loads are cached

### Option 3: Custom Dataset

Your manifest files should be in JSON Lines format:

```json
{"audio_filepath": "/absolute/path/audio1.wav", "duration": 5.3, "text": "transcription text"}
{"audio_filepath": "/absolute/path/audio2.wav", "duration": 2.1, "text": "another transcription"}
```

Requirements:
- **audio_filepath**: absolute path to 16kHz mono WAV files
- **duration**: audio duration in seconds
- **text**: ground truth transcription

---

## Configuration

### 1. Download Training Script

```bash
wget https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/asr/speech_to_text_finetune.py
```

### 2. Create Config Directory

```bash
mkdir -p conf/asr_finetune
```

### 3. Create Config File

Save as `conf/asr_finetune/speech_to_text_finetune.yaml`:

```yaml
name: "Parakeet_Finetune"

init_from_pretrained_model: null

model:
  sample_rate: 16000

  train_ds:
    manifest_filepath: ???
    sample_rate: ${model.sample_rate}
    batch_size: 16
    batch_duration: null
    shuffle: true
    num_workers: 2
    pin_memory: true
    max_duration: 40
    min_duration: 0.1
    text_field: text
    is_tarred: false
    tarred_audio_filepaths: null
    shuffle_n: 2048
    bucketing_strategy: "fully_randomized"
    bucketing_batch_size: null

  validation_ds:
    manifest_filepath: ???
    sample_rate: ${model.sample_rate}
    batch_size: 16
    shuffle: false
    text_field: text
    use_start_end_token: false
    num_workers: 2
    pin_memory: true

  char_labels:
    update_labels: false
    labels: null

  tokenizer:
    update_tokenizer: false
    dir: null
    type: bpe

  optim:
    name: adamw
    lr: 1e-4
    betas: [0.9, 0.98]
    weight_decay: 1e-3
    sched:
      name: CosineAnnealing
      warmup_steps: 1000
      warmup_ratio: null
      min_lr: 5e-6

trainer:
  devices: 1
  num_nodes: 1
  max_epochs: 10
  max_steps: -1
  val_check_interval: 1.0
  accelerator: auto
  precision: 32
  log_every_n_steps: 10
  enable_progress_bar: true
  num_sanity_val_steps: 0
  check_val_every_n_epoch: 1
  sync_batchnorm: true
  enable_checkpointing: false
  logger: false
  benchmark: false

exp_manager:
  exp_dir: null
  name: ${name}
  create_tensorboard_logger: true
  create_checkpoint_callback: true
  checkpoint_callback_params:
    monitor: "val_wer"
    mode: "min"
    save_top_k: 3
    always_save_nemo: true
  resume_if_exists: false
  resume_ignore_no_checkpoint: false
```

---

## Training

### Single GPU Training

For a single H100 / A100:

```bash
python speech_to_text_finetune.py \
  init_from_pretrained_model=nvidia/parakeet-tdt-0.6b-v2 \
  model.train_ds.manifest_filepath=data/nemo_format/train_manifest.json \
  model.train_ds.batch_duration=200.0 \
  model.validation_ds.manifest_filepath=data/nemo_format/val_manifest.json \
  trainer.devices=1 \
  trainer.max_epochs=50 \
  trainer.precision=bf16-mixed \
  exp_manager.exp_dir=experiments/parakeet_finetune \
  exp_manager.name=single_gpu_run
```

**Batch Size Tuning:**
- H100 80GB: `batch_duration=200-300`
- A100 80GB: `batch_duration=200-250`
- A100 40GB: `batch_duration=100-150`
- Reduce if you encounter OOM errors

### Multi-GPU Training (4× H100)

```bash
python speech_to_text_finetune.py \
  init_from_pretrained_model=nvidia/parakeet-tdt-0.6b-v2 \
  model.train_ds.manifest_filepath=medical_nemo_data/train_manifest.json \
  model.train_ds.batch_duration=200.0 \
  model.validation_ds.manifest_filepath=medical_nemo_data/val_manifest.json \
  trainer.devices=4 \
  trainer.max_epochs=50 \
  trainer.precision=bf16-mixed \
  exp_manager.exp_dir=experiments/parakeet_finetune \
  exp_manager.name=multi_gpu_run
```

**Note:** `batch_duration` is per GPU, so total effective batch = 200 × 4 = 800 seconds of audio.

### Key Hyperparameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `batch_duration` | Seconds of audio per batch per GPU | 150-300 |
| `max_duration` | Max audio duration (seconds) | 20 |
| `trainer.precision` | Mixed precision | `bf16-mixed` (H100/A100) |
| `trainer.max_epochs` | Training epochs | 20-50 |
| `optim.lr` | Learning rate | 1e-4 to 5e-5 |

---

## Monitoring

### TensorBoard

Start TensorBoard:

```bash
tensorboard --logdir experiments/parakeet_finetune
```

Open browser: `http://localhost:6006`

Monitor:
- **train_loss**: Should decrease steadily
- **val_wer**: Word Error Rate (lower is better)
- **val_loss**: Validation loss

### Checkpoints

Saved at: `experiments/parakeet_finetune/<run_name>/<timestamp>/checkpoints/`

Files:
- `*.nemo`: NeMo model checkpoints (use for inference)
- Top-k checkpoints based on `val_wer`

---

## Inference

### Load Fine-tuned Model

```python
from nemo.collections.asr.models import ASRModel

# Load best checkpoint
model_path = "experiments/parakeet_finetune/multi_gpu_run/2025-11-15_02-12-11/checkpoints/Parakeet_Finetune--val_wer=0.05-epoch=10.nemo"

asr_model = ASRModel.restore_from(model_path)
asr_model.eval()

# Transcribe
audio_files = ["test1.wav", "test2.wav"]
transcripts = asr_model.transcribe(audio_files)

for file, text in zip(audio_files, transcripts):
    print(f"{file}: {text}")
```

### Batch Inference

```python
# Process large dataset
manifest_path = "test_manifest.json"
output_path = "predictions.json"

predictions = asr_model.transcribe(
    paths2audio_files=manifest_path,
    batch_size=32,
    return_hypotheses=False
)

# Save results
import json
with open(output_path, 'w') as f:
    json.dump(predictions, f, indent=2)
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce `batch_duration`: try 100, 150
2. Reduce `max_duration`: try 15, 10
3. Use `precision=bf16-mixed` instead of `32`
4. Reduce `num_workers`

### Version Conflicts

**Issue:** `TypeError: model must be a LightningModule`

**Fix:**
```bash
pip install 'pytorch-lightning>=2.3.0,<=2.4.0' --force-reinstall
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install fsspec==2024.12.0
```

### Multi-GPU DDP Issues

**Issue:** DDP not initializing properly

**Fix:** Ensure:
- `trainer.logger=false` in config
- `trainer.enable_checkpointing=false` in config
- Run from script, not Jupyter (for multi-GPU)

### Slow Training

**Solutions:**
1. Use `batch_duration` instead of `batch_size` (Lhotse dynamic bucketing)
2. Increase `num_workers` (but watch CPU usage)
3. Use tarred datasets for very large datasets
4. Enable `pretokenize=false` in dataloader config if tokenizer is large

### Low WER Not Improving

**Solutions:**
1. Increase `max_epochs`
2. Reduce learning rate: `optim.lr=5e-5` or `1e-5`
3. Check data quality and transcriptions
4. Enable spec augmentation
5. Ensure `text_field` matches your manifest

---

## Advanced Configuration

### Using Custom Tokenizer

```yaml
model:
  tokenizer:
    update_tokenizer: true
    dir: /path/to/tokenizer
    type: bpe  # or wpe
```

### Spec Augmentation

```yaml
model:
  spec_augment:
    _target_: nemo.collections.asr.modules.SpectrogramAugmentation
    freq_masks: 2
    time_masks: 10
    freq_width: 27
    time_width: 0.05
```

### Learning Rate Scheduling

```yaml
model:
  optim:
    sched:
      name: CosineAnnealing
      warmup_steps: 1000
      min_lr: 1e-6
```

---

## Performance Benchmarks

| Setup | GPU | Batch Duration | Throughput | Time/Epoch |
|-------|-----|----------------|------------|------------|
| Single | H100 80GB | 250 | ~400 audio-sec/s | ~2 hours |
| 4× | H100 80GB | 200 | ~1400 audio-sec/s | ~30 min |
| Single | A100 80GB | 200 | ~300 audio-sec/s | ~3 hours |

*Based on LibriSpeech 100h subset

---

## References

- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Parakeet-TDT Model Card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NeMo ASR Tutorial](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/intro.html)

---

## Support

For issues:
- NeMo GitHub: https://github.com/NVIDIA/NeMo/issues
- NeMo Discussions: https://github.com/NVIDIA/NeMo/discussions

---

**Last Updated:** November 15, 2025
