# 🎯 Complete Specification-Compliant BERT Training Guide

## Overview

This guide provides step-by-step instructions for training BERT models with **exact specification compliance** as required by the project documentation.

### 📋 Project Specifications (REQUIRED)
- ✅ **Masked Language Modeling (MLM)** with 15% random masking
- ✅ **AdamW optimizer** with learning rate 2×10⁻⁵
- ✅ **Batch size**: 8 sequences per device
- ✅ **Training epochs**: 3 epochs
- ✅ **Weight decay**: 0.01
- ✅ **Mixed precision**: FP16 enabled
- ✅ **Checkpoints**: Every 5000 optimization iterations

---

## 🚀 Quick Start (One Command)

For immediate training with balanced datasets and research optimization:

```bash
cd /Users/admin/Study/McGill\ Kernel/
./train_balanced_specification.sh
```

---

## 📖 Detailed Step-by-Step Instructions

### Step 1: Environment Preparation

```bash
# Navigate to project directory
cd "/Users/admin/Study/McGill Kernel/"

# Verify required directories exist
ls -la data/
ls -la tokenizer/
ls -la scripts/

# Check GPU availability (recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Create BPE Tokenizer (if not exists)

```bash
# Check if tokenizer exists
if [ -f "tokenizer/hansard-bpe-tokenizer/vocab.json" ]; then
    echo "✅ Tokenizer already exists"
else
    echo "🔄 Creating BPE tokenizer..."
    cd scripts/
    python create_bpe_tokenizer.py
    cd ..
fi
```

### Step 3: Create Research-Optimized Balanced Datasets

This step creates datasets with equal sample sizes for fair bias comparison and research-optimized splits (85% train, 15% test):

```bash
cd scripts/
python create_research_optimized_datasets.py --approach research --balanced

# Verify datasets were created
ls -la ../data/dataset_*_balanced_research_*.csv
```

**Expected Output:**
```
../data/dataset_a_balanced_research_train.csv  (103,603 samples)
../data/dataset_a_balanced_research_test.csv   (18,283 samples)
../data/dataset_b_balanced_research_train.csv  (103,603 samples)
../data/dataset_b_balanced_research_test.csv   (18,283 samples)
../data/dataset_c_balanced_research_train.csv  (103,603 samples)
../data/dataset_c_balanced_research_test.csv   (18,283 samples)
```

### Step 4: Train Specification-Compliant Models

Train all three models with exact project specifications:

#### 4a. Train Dataset A (Nunavut-only Model)
```bash
python bert_trainer_specification.py a --balanced --research-optimized --spec-compliant
```

#### 4b. Train Dataset B (Canadian-only Model)
```bash
python bert_trainer_specification.py b --balanced --research-optimized --spec-compliant
```

#### 4c. Train Dataset C (Mixed 50-50 Model)
```bash
python bert_trainer_specification.py c --balanced --research-optimized --spec-compliant
```

### Step 5: Monitor Training Progress

Each training run will show:

```
🚀 SPECIFICATION-COMPLIANT BERT TRAINING
============================================================
Dataset: A
Type: balanced
Optimization: research-optimized
Specification compliance: ENFORCED

📊 Using balanced datasets for rigorous bias detection
🔬 Using research-optimized approach:
  • 85% training data (21% more cultural signal)
  • 15% test data (sufficient for bias evaluation)
  • No validation set (optimized for bias detection)

⏱️  Estimated training time: 3-8 hours (depends on GPU)
```

### Step 6: Verify Training Completion

After each model training, check:

```bash
# Verify model directories were created
ls -la models/model_*_balanced_research_spec/

# Check training logs
ls -la logs/model_*_balanced_research_spec/

# Review training summaries
cat models/model_a_balanced_research_spec/specification_training_summary.json
```

### Step 7: Training Results Structure

Each completed model will have:

```
models/model_[a|b|c]_balanced_research_spec/
├── config.json                           # Model configuration
├── pytorch_model.bin                     # Trained model weights
├── specification_training_summary.json   # Training metrics
├── vocab.json                            # Tokenizer vocabulary
└── merges.txt                            # BPE merges

logs/model_[a|b|c]_balanced_research_spec/
├── training_logs.log                     # Detailed training logs
└── events.out.tfevents.*                # TensorBoard logs (if enabled)

checkpoints/model_[a|b|c]_balanced_research_spec/
├── checkpoint-5000/                     # Every 5000 steps
├── checkpoint-10000/
└── ...
```

---

## 🔧 Training Configuration Details

### Exact Hyperparameters Used

```python
training_config = {
    "model_architecture": "BERT-base (768 hidden, 12 layers, 110M parameters)",
    "objective": "Masked Language Modeling with 15% random masking",
    "optimizer": "AdamW",
    "learning_rate": 2e-5,
    "batch_size_per_device": 8,
    "num_epochs": 3,
    "weight_decay": 0.01,
    "mixed_precision": "FP16",
    "checkpoint_frequency": 5000,
    "warmup_steps": "min(1000, total_steps // 10)",
    "max_sequence_length": 512,
    "vocab_size": 30522
}
```

### Dataset Configuration

```python
dataset_config = {
    "approach": "research-optimized",
    "balance": "equal_samples",
    "split_ratio": "85% train / 15% test",
    "samples_per_dataset": 121886,
    "train_samples": 103603,
    "test_samples": 18283,
    "cultural_signal_gain": "+21% vs traditional splits"
}
```

---

## ⏱️ Expected Training Times

| Hardware | Time per Model | Total Time (3 models) |
|----------|---------------|-----------------------|
| RTX 4090 | 2-3 hours     | 6-9 hours            |
| RTX 3080 | 3-4 hours     | 9-12 hours           |
| RTX 2080 | 4-6 hours     | 12-18 hours          |
| CPU only | 24-48 hours   | 72-144 hours         |

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Error
```bash
# If you get CUDA OOM, the training will automatically suggest:
# "Consider using gradient accumulation if OOM occurs"
# 
# The batch size is fixed at 8 per specification, but you can:
# - Close other GPU applications
# - Use a smaller GPU if available
# - Training will attempt automatic memory optimization
```

#### 2. Tokenizer Not Found
```bash
❌ Error: Custom tokenizer not found at ../tokenizer/hansard-bpe-tokenizer
# Solution:
cd scripts/
python create_bpe_tokenizer.py
```

#### 3. Dataset Files Missing
```bash
❌ Error: Research-optimized balanced dataset not found
# Solution:
python create_research_optimized_datasets.py --approach research --balanced
```

#### 4. Training Interrupted
```bash
# Training can be resumed from last checkpoint:
python bert_trainer_specification.py a --balanced --research-optimized --spec-compliant
# The trainer will automatically detect and resume from the latest checkpoint
```

---

## 📊 Training Validation

### Verify Specification Compliance

Each training run validates compliance:

```
🔍 VALIDATING SPECIFICATION COMPLIANCE
============================================================
  MLM Objective: ✅ Implemented with 15% random masking
  AdamW Optimizer: ✅ HuggingFace Transformers default
  Learning Rate: ✅ Set to 2×10⁻⁵
  Batch Size: ✅ Fixed at 8 sequences per device
  Epochs: ✅ Set to 3 epochs
  Weight Decay: ✅ Set to 0.01
  Mixed Precision: ✅ FP16 enabled for cuda
  Checkpoints: ✅ Every 5000 optimization iterations
============================================================
✅ ALL SPECIFICATIONS VALIDATED
```

### Check Training Summary

```bash
# View final training metrics
python -c "
import json
with open('models/model_a_balanced_research_spec/specification_training_summary.json') as f:
    summary = json.load(f)
    print(f'Model: {summary[\"model_name\"]}')
    print(f'Final Loss: {summary[\"final_loss\"]:.4f}')
    print(f'Perplexity: {summary[\"final_perplexity\"]:.2f}')
    print(f'Training Time: {summary[\"training_time\"]}')
    print('Specifications:')
    for spec, value in summary['specification_compliance'].items():
        print(f'  ✅ {spec}: {value}')
"
```

---

## 🎯 Success Criteria

Training is successful when:

1. ✅ All three models complete without errors
2. ✅ Each model has ~110M parameters (BERT-base size)
3. ✅ Final perplexity is reasonable (typically 5-15 for parliamentary text)
4. ✅ All specification requirements are validated
5. ✅ Training summaries show exact hyperparameter compliance
6. ✅ Model files are saved in expected directories

---

## 🔬 Research-Optimized Benefits

This training approach provides:

- **21% more cultural data** per model (103,603 vs 85,320 training samples)
- **Enhanced bias detection capability** through richer cultural representation
- **Research-focused methodology** optimized for cross-evaluation analysis
- **Specification compliance** with exact project requirements
- **Balanced datasets** for fair bias comparison without data quantity confounds

---

## 📝 Next Steps After Training

Once all models are trained:

1. **Cross-Evaluation**: Test each model on all datasets
2. **Bias Analysis**: Compare perplexity differences across cultural contexts
3. **Statistical Testing**: Quantify bias significance
4. **Report Generation**: Document findings for COMP550 project

---

## 🆘 Getting Help

If you encounter issues:

1. Check the training logs in `logs/model_*_balanced_research_spec/`
2. Review the specification validation output
3. Ensure GPU memory availability
4. Verify all prerequisite files exist
5. Consider running demo training first to test environment

**The complete training represents the rigorous, specification-compliant approach required for your COMP550 cultural bias detection research.**