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

### Step 2: Upload Project Files

#### Option A: Direct Upload (Recommended for Small Projects)
```python
# Upload project files directly to Colab
from google.colab import files
import shutil
import zipfile

print("📁 Upload your project files...")

# Method 1: Upload individual Python files
uploaded = files.upload()

# Move uploaded files to correct locations
for filename in uploaded.keys():
    if filename.endswith('.py'):
        if 'trainer' in filename.lower():
            shutil.move(filename, f'/content/McGill_Kernel/scripts/{filename}')
        else:
            shutil.move(filename, f'/content/McGill_Kernel/scripts/{filename}')
    elif filename.endswith('.csv'):
        shutil.move(filename, f'/content/McGill_Kernel/data/{filename}')

print("✅ Files uploaded and organized!")
```

#### Option B: Google Drive Integration (Recommended for Large Projects)
```python
# Sync project from Google Drive
import shutil

# Assuming you've uploaded your project to Google Drive
DRIVE_PROJECT_PATH = '/content/drive/MyDrive/McGill_Kernel'  # Adjust path
COLAB_PROJECT_PATH = '/content/McGill_Kernel'

# Copy entire project from Drive to Colab
if os.path.exists(DRIVE_PROJECT_PATH):
    shutil.copytree(DRIVE_PROJECT_PATH, COLAB_PROJECT_PATH, dirs_exist_ok=True)
    print("✅ Project synced from Google Drive!")
else:
    print("❌ Project not found in Google Drive. Please upload it first.")

# Verify project structure
!ls -la /content/McGill_Kernel/
!ls -la /content/McGill_Kernel/scripts/
```

#### Option C: GitHub Integration
```python
# Clone from GitHub repository
!git clone https://github.com/yourusername/mcgill-kernel.git /content/McGill_Kernel

# Or download specific files
!wget -O /content/McGill_Kernel/scripts/bert_trainer_specification.py \
    https://raw.githubusercontent.com/yourusername/repo/main/scripts/bert_trainer_specification.py

print("✅ Project downloaded from GitHub!")
```

### Step 3: Upload Dataset Files

```python
# Upload dataset files (if not already in Drive)
print("📊 Upload dataset files...")
print("Expected files:")
print("- dataset_a_train.csv, dataset_a_val.csv, dataset_a_test.csv")
print("- dataset_b_train.csv, dataset_b_val.csv, dataset_b_test.csv") 
print("- dataset_c_train.csv, dataset_c_val.csv, dataset_c_test.csv")

# Upload datasets
uploaded_data = files.upload()

# Move to data directory
for filename in uploaded_data.keys():
    if filename.endswith('.csv'):
        shutil.move(filename, f'/content/McGill_Kernel/data/{filename}')

# Verify datasets
!ls -la /content/McGill_Kernel/data/*.csv
```

### Step 4: Create Colab-Optimized Training Scripts

```python
# Create Colab-optimized versions of your training scripts
os.chdir('/content/McGill_Kernel/scripts')

# Modify paths for Colab environment
!sed -i 's|../data/|/content/McGill_Kernel/data/|g' *.py
!sed -i 's|../models/|/content/McGill_Kernel/models/|g' *.py  
!sed -i 's|../logs/|/content/McGill_Kernel/logs/|g' *.py
!sed -i 's|../tokenizer/|/content/McGill_Kernel/tokenizer/|g' *.py
!sed -i 's|../checkpoints/|/content/McGill_Kernel/checkpoints/|g' *.py
!sed -i 's|../results/|/content/McGill_Kernel/results/|g' *.py

print("✅ Paths updated for Colab environment!")
```

---

## 🎯 Training Execution on Colab

### Quick Training Setup

```python
# Change to project directory
os.chdir('/content/McGill_Kernel')

# Create BPE tokenizer first
print("🔤 Creating BPE tokenizer...")
!python scripts/create_bpe_tokenizer.py

# Create research-optimized balanced datasets
print("📊 Creating research-optimized datasets...")
!python scripts/create_research_optimized_datasets.py --approach research --balanced

# Verify setup
print("✅ Checking setup...")
!ls -la tokenizer/hansard-bpe-tokenizer/
!ls -la data/dataset_*_balanced_research_*.csv
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

### Download Trained Models

```python
# Download trained models to local machine
from google.colab import files
import zipfile

def download_results():
    """Prepare and download training results"""
    
    # Create zip file of results
    !cd /content/McGill_Kernel && zip -r training_results.zip models/ logs/ results/
    
    # Download zip file
    files.download('/content/McGill_Kernel/training_results.zip')
    
    print("✅ Results downloaded!")
    print("Extract the zip file to continue working locally.")

# Call when training is complete
# download_results()
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