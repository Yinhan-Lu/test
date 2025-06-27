# 🚀 Google Colab Training Guide - Keep Python Project Structure

## Overview

This guide shows how to run your **complete Python project** on Google Colab while maintaining the `.py` file structure, rather than converting everything to notebooks.

### 🎯 Benefits of This Approach
- ✅ **Keep your Python project structure** intact
- ✅ **Use Colab's free GPU/TPU** resources
- ✅ **No code conversion** required
- ✅ **Version control friendly** (still using `.py` files)
- ✅ **Easy debugging** and development

---

## 📋 Prerequisites

1. **Google Account** with Google Drive access
2. **Google Colab** (colab.research.google.com)
3. **Your project files** ready for upload

---

## 🚀 Step-by-Step Setup

### Step 1: Create Colab Setup Notebook

Create a new notebook in Google Colab with this setup code:

```python
# ============================================================================
# COLAB SETUP FOR BERT TRAINING PROJECT
# ============================================================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install transformers>=4.21.0
!pip install tokenizers>=0.13.0
!pip install torch>=1.12.0
!pip install datasets
!pip install accelerate

# Check GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Create project directory structure
import os
os.makedirs('/content/McGill_Kernel', exist_ok=True)
os.makedirs('/content/McGill_Kernel/scripts', exist_ok=True)
os.makedirs('/content/McGill_Kernel/data', exist_ok=True)
os.makedirs('/content/McGill_Kernel/models', exist_ok=True)
os.makedirs('/content/McGill_Kernel/logs', exist_ok=True)
os.makedirs('/content/McGill_Kernel/tokenizer', exist_ok=True)
os.makedirs('/content/McGill_Kernel/checkpoints', exist_ok=True)
os.makedirs('/content/McGill_Kernel/results', exist_ok=True)

print("✅ Colab environment setup complete!")
```

### Step 2: Clone Project from GitHub

```python
# Clone your GitHub repository
GITHUB_REPO_URL = "https://github.com/yourusername/mcgill-kernel.git"  # Replace with your repo URL
PROJECT_PATH = "/content/McGill_Kernel"

print("📥 Cloning project from GitHub...")

# Remove existing directory if it exists
import shutil
import os
if os.path.exists(PROJECT_PATH):
    print(f"🗑️ Removing existing directory: {PROJECT_PATH}")
    shutil.rmtree(PROJECT_PATH)

# Clone the repository
!git clone {GITHUB_REPO_URL} {PROJECT_PATH}

# Change to project directory
os.chdir(PROJECT_PATH)

# Verify project structure
print("✅ Project cloned successfully!")
print("\n📁 Project structure:")
!ls -la
!ls -la scripts/

# Check if required files exist
required_files = [
    'scripts/hansard_pipeline.py',
    'scripts/hybrid_nunavut_parser.py', 
    'scripts/tokenizer_trainer.py',
    'requirements.txt'
]

print("\n🔍 Checking required files:")
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ MISSING: {file_path}")
```

### Step 3: Setup Data Processing

```python
# Since datasets are large and not included in repo, we'll generate them using the pipeline
print("📊 Setting up data processing...")

# Check if we have the raw data files needed
print("🔍 Checking for raw data files:")
raw_data_files = [
    'data/preprocessed_nunavut_hansard.txt',
    'data/cleaned_canadian_hansard.csv'
]

for file_path in raw_data_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ MISSING: {file_path}")
        print(f"   You'll need to add this file to your GitHub repo or generate it")

print("\n💡 Note: Large datasets are excluded from GitHub due to size limits.")
print("   You can either:")
print("   1. Use Git LFS for large files")
print("   2. Process smaller sample datasets")
print("   3. Generate datasets using the pipeline scripts")
```

### Step 4: Install Dependencies and Setup Environment

```python
# Install project dependencies
print("📦 Installing project dependencies...")

# Install from requirements.txt if it exists
if os.path.exists('requirements.txt'):
    !pip install -r requirements.txt
else:
    # Install essential packages for the project
    !pip install pandas numpy tokenizers transformers torch

print("✅ Dependencies installed!")

# Verify Python environment
import sys
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Test import of key modules
try:
    import pandas as pd
    import numpy as np
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaConfig, RobertaForMaskedLM
    print("✅ All required modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

---

## 🎯 Training Execution on Colab

### Quick Training Setup

```python
# Ensure we're in the project directory  
os.chdir('/content/McGill_Kernel')

# Run the data processing pipeline to create datasets
print("📊 Running data processing pipeline...")
!python scripts/hansard_pipeline.py

# Create BPE tokenizer
print("🔤 Creating BPE tokenizer...")
!python scripts/tokenizer_trainer.py

# Verify setup
print("✅ Checking setup...")
print("\n📁 Data files created:")
!ls -la data/dataset_*.csv

print("\n🔤 Tokenizer files:")
!ls -la tokenizer/hansard-bpe-tokenizer/

print("\n📋 Validation reports:")
!ls -la data/*report*.json
```

### Complete Training Execution

```python
# ============================================================================
# SPECIFICATION-COMPLIANT TRAINING ON COLAB
# ============================================================================

import subprocess
import time
from datetime import datetime

def run_training(dataset, description):
    """Run training with progress monitoring"""
    print(f"\n{'='*70}")
    print(f"🚀 STARTING {description}")
    print(f"{'='*70}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Run training
    cmd = f"python scripts/bert_trainer_specification.py {dataset} --balanced --research-optimized --spec-compliant"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/content/McGill_Kernel')
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Duration: {duration/3600:.2f} hours")
        
        if result.returncode == 0:
            print(f"✅ {description} COMPLETED SUCCESSFULLY!")
            print("📊 Training Summary:")
            print(result.stdout[-1000:])  # Last 1000 chars of output
        else:
            print(f"❌ {description} FAILED!")
            print("Error output:")
            print(result.stderr[-1000:])
            
    except Exception as e:
        print(f"❌ Training error: {e}")
    
    return result.returncode == 0

# Execute training for all datasets
training_results = {}

# Train Dataset A (Nunavut)
training_results['A'] = run_training('a', 'DATASET A (NUNAVUT) TRAINING')

# Train Dataset B (Canadian) 
training_results['B'] = run_training('b', 'DATASET B (CANADIAN) TRAINING')

# Train Dataset C (Mixed)
training_results['C'] = run_training('c', 'DATASET C (MIXED) TRAINING')

# Final summary
print(f"\n{'='*70}")
print("🎯 FINAL TRAINING SUMMARY")
print(f"{'='*70}")
for dataset, success in training_results.items():
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"Dataset {dataset}: {status}")
```

### Monitor Training Progress

```python
# Monitor training in real-time
import os
import time

def monitor_training():
    """Monitor training progress"""
    log_dirs = [
        '/content/McGill_Kernel/logs/model_a_balanced_research_spec',
        '/content/McGill_Kernel/logs/model_b_balanced_research_spec', 
        '/content/McGill_Kernel/logs/model_c_balanced_research_spec'
    ]
    
    while True:
        print(f"\n📊 Training Progress - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)
        
        for log_dir in log_dirs:
            dataset = log_dir.split('_')[1].upper()
            if os.path.exists(log_dir):
                log_files = os.listdir(log_dir)
                if log_files:
                    print(f"Dataset {dataset}: Training active ({len(log_files)} log files)")
                else:
                    print(f"Dataset {dataset}: No logs yet")
            else:
                print(f"Dataset {dataset}: Not started")
        
        # Check model outputs
        model_dirs = [d for d in os.listdir('/content/McGill_Kernel/models/') 
                     if 'balanced_research_spec' in d]
        print(f"\n📁 Models created: {len(model_dirs)}")
        for model_dir in model_dirs:
            print(f"  - {model_dir}")
        
        time.sleep(300)  # Check every 5 minutes

# Run monitoring (comment out when not needed)
# monitor_training()
```

---

## 🔧 Colab-Specific Optimizations

### GPU Memory Management

```python
# Optimize for Colab GPU memory
import torch
import gc

def optimize_gpu_memory():
    """Optimize GPU memory usage for Colab"""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check memory
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU Memory:")
        print(f"  Allocated: {allocated:.2f}GB")
        print(f"  Reserved: {reserved:.2f}GB") 
        print(f"  Total: {total:.2f}GB")
        print(f"  Free: {total - reserved:.2f}GB")
        
        # Suggest optimizations
        if total < 16:
            print("\n💡 Colab GPU optimizations:")
            print("  - Use batch_size=4 instead of 8 if OOM occurs")
            print("  - Consider shorter sequences (max_length=256)")
            print("  - Enable gradient checkpointing if available")

optimize_gpu_memory()
```

### Session Management

```python
# Prevent Colab timeout during long training
from IPython.display import Javascript
import asyncio

def prevent_colab_timeout():
    """Prevent Colab from timing out during training"""
    display(Javascript('''
        function ClickConnect(){
            console.log("Working"); 
            document.querySelector("colab-toolbar-button#connect").click() 
        }
        setInterval(ClickConnect,60000)
    '''))
    print("🔄 Auto-refresh enabled to prevent timeout")

prevent_colab_timeout()
```

### Save Results to Drive

```python
# Automatically save results to Google Drive
def save_to_drive():
    """Save training results to Google Drive"""
    drive_backup_path = '/content/drive/MyDrive/McGill_Kernel_Results'
    os.makedirs(drive_backup_path, exist_ok=True)
    
    # Copy models
    if os.path.exists('/content/McGill_Kernel/models'):
        !cp -r /content/McGill_Kernel/models/* /content/drive/MyDrive/McGill_Kernel_Results/
    
    # Copy logs  
    if os.path.exists('/content/McGill_Kernel/logs'):
        !cp -r /content/McGill_Kernel/logs /content/drive/MyDrive/McGill_Kernel_Results/
    
    # Copy results
    if os.path.exists('/content/McGill_Kernel/results'):
        !cp -r /content/McGill_Kernel/results /content/drive/MyDrive/McGill_Kernel_Results/
    
    print("✅ Results backed up to Google Drive!")

# Save results after training
save_to_drive()
```

---

## 🎮 Demo Training on Colab

### Quick Demo Setup

```python
# Quick demo training for testing
def run_demo_training():
    """Run quick demo training on Colab"""
    print("🎮 Starting Demo Training on Colab")
    
    # Create minimal demo script
    demo_script = '''
import pandas as pd
import torch
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer

# Load sample data
df = pd.read_csv('/content/McGill_Kernel/data/dataset_a_train.csv').head(1000)
texts = df['speechtext'].astype(str).tolist()

# Load tokenizer
tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename='/content/McGill_Kernel/tokenizer/hansard-bpe-tokenizer/vocab.json',
    merges_filename='/content/McGill_Kernel/tokenizer/hansard-bpe-tokenizer/merges.txt'
)
tokenizer.add_special_tokens(['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

print(f"Demo data: {len(texts)} samples")
print(f"Vocabulary: {tokenizer.get_vocab_size()} tokens")
print("🚀 Demo training ready!")
'''
    
    with open('/content/McGill_Kernel/demo_colab.py', 'w') as f:
        f.write(demo_script)
    
    print("✅ Demo script created!")
    print("Run: !python /content/McGill_Kernel/demo_colab.py")

run_demo_training()
```

---

## 📊 Training Results Management

### Check Training Status

```python
def check_training_status():
    """Check current training status"""
    print("📊 TRAINING STATUS CHECK")
    print("=" * 50)
    
    # Check models
    model_dir = '/content/McGill_Kernel/models'
    if os.path.exists(model_dir):
        models = [d for d in os.listdir(model_dir) if 'balanced_research_spec' in d]
        print(f"✅ Models completed: {len(models)}/3")
        for model in models:
            print(f"  - {model}")
    
    # Check logs
    log_dir = '/content/McGill_Kernel/logs'
    if os.path.exists(log_dir):
        logs = os.listdir(log_dir)
        print(f"📝 Log directories: {len(logs)}")
    
    # Check disk usage
    !df -h /content/
    
    # GPU status
    if torch.cuda.is_available():
        print(f"\n🎮 GPU Status:")
        print(f"  Memory used: {torch.cuda.memory_allocated()/1e9:.2f}GB")

check_training_status()
```

### Save Results to GitHub

```python
# Save results back to your GitHub repository
def save_results_to_github():
    """Save training results back to GitHub repository"""
    
    print("📤 Saving results to GitHub...")
    
    # Configure git (replace with your details)
    !git config --global user.email "your.email@example.com"
    !git config --global user.name "Your Name"
    
    # Add trained models and results (if they're not too large)
    !git add models/ logs/ results/
    
    # Create commit
    !git commit -m "Add trained models and results from Colab training"
    
    # Push to GitHub (you may need to authenticate)
    print("🔐 You may need to authenticate with GitHub...")
    print("Use a personal access token if prompted for password")
    !git push origin main
    
    print("✅ Results saved to GitHub repository!")

# Alternative: Create a results summary instead of uploading large files
def create_results_summary():
    """Create a summary of training results"""
    
    summary = f"""
# Training Results Summary

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Models Trained
"""
    
    # Check which models were created
    if os.path.exists('/content/McGill_Kernel/models'):
        models = os.listdir('/content/McGill_Kernel/models')
        for model in models:
            summary += f"- {model}\n"
    
    # Add log information
    if os.path.exists('/content/McGill_Kernel/logs'):
        summary += "\n## Training Logs\n"
        logs = os.listdir('/content/McGill_Kernel/logs')
        for log in logs:
            summary += f"- {log}\n"
    
    # Write summary file
    with open('/content/McGill_Kernel/TRAINING_RESULTS.md', 'w') as f:
        f.write(summary)
    
    print("✅ Results summary created!")
    print("📄 Check TRAINING_RESULTS.md for details")

# Call when training is complete
# create_results_summary()
# save_results_to_github()
```

---

## 🚨 Troubleshooting Colab Issues

### Common Problems and Solutions

#### 1. Runtime Disconnection
```python
# If runtime disconnects, reconnect and run:
from google.colab import drive
drive.mount('/content/drive')

# Restore from backup
!cp -r /content/drive/MyDrive/McGill_Kernel_Results/* /content/McGill_Kernel/

# Resume training from checkpoint
!python scripts/bert_trainer_specification.py a --balanced --research-optimized --spec-compliant
```

#### 2. GPU Memory Issues
```python
# If you get CUDA OOM errors:
# 1. Restart runtime
# 2. Modify batch size in your script
!sed -i 's/per_device_train_batch_size=8/per_device_train_batch_size=4/g' /content/McGill_Kernel/scripts/bert_trainer_specification.py

# 3. Or use CPU (slower but works)
!sed -i 's/fp16=self.device.type == "cuda"/fp16=False/g' /content/McGill_Kernel/scripts/bert_trainer_specification.py
```

#### 3. File Not Found Errors
```python
# Verify all files are in correct locations
def verify_setup():
    required_files = [
        '/content/McGill_Kernel/scripts/bert_trainer_specification.py',
        '/content/McGill_Kernel/scripts/create_research_optimized_datasets.py',
        '/content/McGill_Kernel/data/dataset_a_train.csv',
        '/content/McGill_Kernel/tokenizer/hansard-bpe-tokenizer/vocab.json'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ MISSING: {file_path}")

verify_setup()
```

---

## 🎯 Complete Colab Workflow Summary

### 1. Setup Phase (5-10 minutes)
```python
# Mount Drive → Install packages → Upload files → Update paths
```

### 2. Training Phase (3-8 hours per model)
```python
# Create tokenizer → Create datasets → Train models (A, B, C)
```

### 3. Results Phase (5 minutes)
```python
# Backup to Drive → Download results → Verify completion
```

---

## 💡 Pro Tips for Colab Training

1. **Use Colab Pro** for longer runtimes and better GPUs
2. **Save frequently** to Google Drive during training
3. **Monitor GPU usage** to avoid memory issues
4. **Use smaller batches** if you hit memory limits
5. **Keep browser tab active** to prevent disconnections
6. **Upload datasets to Drive first** for faster access
7. **Use checkpoint resuming** if training gets interrupted

---

## 🎯 Final Checklist

Before starting training on Colab:

- [ ] ✅ Google Colab account ready
- [ ] ✅ Project files uploaded/synced
- [ ] ✅ Dataset files available
- [ ] ✅ GPU runtime selected
- [ ] ✅ All dependencies installed
- [ ] ✅ Paths updated for Colab
- [ ] ✅ Auto-save to Drive enabled
- [ ] ✅ Timeout prevention active

**Your Python project will run seamlessly on Colab while maintaining the original `.py` structure!**