# Scripts - Training and Analysis Tools

Collection of scripts for training YOLO pose detection models, running experiments, and analyzing results for the lizard head pose estimation project.

## 📁 Scripts Overview

### Training Scripts
- **`pogona_pipeline_cfg.py`**: Main training pipeline with standard configuration
- **`pogona_pipeline_cfg_optuna.py`**: Training pipeline with Optuna hyperparameter optimization
- **`train_pose.sh`**: Shell wrapper for pose training
- **`val_pose.sh`**: Validation script

### HPC/SLURM Scripts
- **`slurm_train.sbatch`**: SLURM batch job for training on HPC clusters
- **`compare_runs_slurm.sbatch`**: SLURM batch job for comparing training runs

### Analysis Scripts  
- **`compare_runs_full.py`**: Comprehensive comparison of training runs and metrics
- **`README_compare_full.md`**: Documentation for comparison tools

### Utility Scripts
- **`run_train_gpu.sh`**: Local GPU training wrapper
- **`export_pose.sh`**: Model export utilities

## 🚀 Usage

### Local Training
```bash
# Standard training
cd /path/to/lizard-tracking
python scripts/pogona_pipeline_cfg.py

# With Optuna optimization  
python scripts/pogona_pipeline_cfg_optuna.py

# GPU training wrapper
./scripts/run_train_gpu.sh
```

### HPC Cluster Training
```bash
# Configure your SLURM account first
nano scripts/slurm_train.sbatch
# Uncomment and set: #SBATCH --account=YOUR_ACCOUNT_HERE

# Submit training job
sbatch scripts/slurm_train.sbatch

# Submit comparison job
sbatch scripts/compare_runs_slurm.sbatch
```

### Analysis & Comparison
```bash
# Compare training runs
python scripts/compare_runs_full.py

# Export trained models
./scripts/export_pose.sh
```

## ⚙️ Configuration

### SLURM Account Setup - only if needed
The SLURM scripts require account configuration before use:

1. **Copy example scripts** (first time setup):
   ```bash
   cp scripts/slurm_train.sbatch.example scripts/slurm_train.sbatch
   cp scripts/compare_runs_slurm.sbatch.example scripts/compare_runs_slurm.sbatch
   ```

2. **Edit your local copies** with your account info:
   ```bash
   nano scripts/slurm_train.sbatch
   nano scripts/compare_runs_slurm.sbatch
   ```

3. **Set your SLURM account**:
   ```bash
   #SBATCH --account=YOUR_ACCOUNT_HERE
   ```

4. **Adjust resources as needed**:
   - GPU allocation: `--gpus-per-task=1`
   - Memory: `--mem=64G` (training) or `--mem=32G` (comparison)
   - Time limits: `--time=12:00:00` or `--time=8:00:00`

**Note**: The `.sbatch` files are gitignored (like `.env` files) to keep sensitive account information private. Only the sanitized `.sbatch.example` files are tracked in git.

### Environment Setup
Scripts expect conda environment named `LizardPose`:
```bash
# Create environment (if not exists)
conda create -n LizardPose python=3.8
conda activate LizardPose
pip install -r requirements.txt
```

### Path Configuration
Scripts automatically detect project root and adjust paths:
- **Project Root**: Auto-detected via `$PWD/sandbox/lizard-tracking`
- **Script Paths**: Updated to use `src/scripts/` location
- **Output Paths**: Configured for scratch storage on HPC

## 📊 Training Pipeline Features

### Standard Training (`pogona_pipeline_cfg.py`)
- YOLO pose detection for lizard head keypoints
- 3 keypoints: nose, left ear, right ear
- Configurable epochs, batch size, learning rate
- Automatic model checkpointing and validation

### Optuna Optimization (`pogona_pipeline_cfg_optuna.py`)
- Hyperparameter optimization using Optuna
- Optimizes learning rate, batch size, augmentation parameters
- Automatic pruning of poor-performing trials
- Best configuration selection and model training

### HPC Integration
- SLURM batch job submission
- GPU allocation and CUDA configuration
- Conda environment activation
- Scratch storage for outputs and logs
- Job monitoring and error handling

## 📈 Analysis Tools

### Run Comparison (`compare_runs_full.py`)
- Compare metrics across multiple training runs
- Validation accuracy, loss curves, mAP scores
- Model performance analysis
- Export comparison reports and visualizations

### Validation Tools
- Model evaluation on test sets
- Keypoint detection accuracy metrics
- Pose estimation error analysis
- Export validation results

## 🔧 Notes

### Sensitive Information Removed
- **Account Info**: Real `.sbatch` scripts are gitignored (like `.env` files)
- **Example Files**: `.sbatch.example` versions provided as templates
- **User Setup**: Copy examples and add your account information locally
- **Security**: No sensitive cluster information in version control


---

*Part of the lizard-tracking project for bearded dragon head pose estimation and behavioral analysis.*