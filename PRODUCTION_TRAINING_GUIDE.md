# 🚀 Production BERT Training Guide

## Complete Cultural Bias Analysis Training Pipeline

This guide explains how to train the full production BERT models for your COMP550 cultural bias analysis project.

---

## 📋 **Quick Start**

### **Option 1: Train All Models (Recommended)**
```bash
# Activate environment
source venv/bin/activate

# Train all three production models
cd scripts
python train_all_models.py --evaluate

# This will:
# 1. Train Model A (Nunavut data) - 2-6 hours
# 2. Train Model B (Canadian data) - 4-12 hours  
# 3. Train Model C (Mixed data) - 2-6 hours
# 4. Run comprehensive bias evaluation
```

### **Option 2: Train Individual Models**
```bash
# Train specific models
python bert_trainer_production.py a  # Model A (Nunavut)
python bert_trainer_production.py b  # Model B (Canadian) 
python bert_trainer_production.py c  # Model C (Mixed)

# Run evaluation
python model_evaluator_production.py
```

---

## 🏗️ **Architecture & Code Explanation**

### **Core Components**

#### **1. HansardDataset Class**
```python
class OptimizedHansardDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
```
**Purpose**: Memory-efficient PyTorch Dataset for parliamentary speeches
- **Input**: Raw speech texts from CSV files
- **Processing**: BPE tokenization → truncation/padding to 512 tokens
- **Output**: `input_ids` and `attention_mask` tensors for BERT
- **Optimization**: Optional tokenization caching for smaller datasets

#### **2. ProductionBERTTrainer Class**
```python
class ProductionBERTTrainer:
    def train_production_model(self, num_epochs=3):
```
**Key Features**:
- **Auto GPU Detection**: Automatically detects and optimizes for available hardware
- **Memory Management**: Dynamic batch size based on GPU memory
- **Mixed Precision**: FP16 training for 2x memory efficiency
- **Checkpointing**: Automatic saving every 1000 steps with resume capability
- **Early Stopping**: Prevents overfitting with patience=3

#### **3. Model Configuration**
```python
config = BertConfig(
    vocab_size=30522,           # Your custom BPE tokenizer
    hidden_size=768,            # BERT-base architecture  
    num_hidden_layers=12,       # Full 12 transformer layers
    num_attention_heads=12,     # Multi-head attention
    max_position_embeddings=512 # Full sequence length
)
```
**Full Production Model**: ~110M parameters vs 22M in demo

#### **4. Training Pipeline**
```
Raw Text → BPE Tokenization → MLM Masking → BERT Training → Model Saving
     ↓              ↓              ↓             ↓            ↓
Parliamentary   Token IDs     15% Random    Predict      Save Model
  Speeches      (0-30521)     Masking      Masked        + Tokenizer
                              ([MASK])     Tokens
```

---

## ⚙️ **Production Optimizations**

### **Automatic GPU Optimization**
```python
def _get_optimal_batch_size(self, model_size_mb: float):
    if gpu_memory_gb >= 24:    # RTX 4090, A100
        return 32, 4           # batch_size=32, grad_accum=4
    elif gpu_memory_gb >= 16:  # RTX 3080 Ti, RTX 4080
        return 16, 8           # batch_size=16, grad_accum=8  
    elif gpu_memory_gb >= 12:  # RTX 3080, RTX 4070
        return 8, 16           # batch_size=8, grad_accum=16
```

### **Mixed Precision Training (FP16)**
- **Memory Savings**: 50% reduction in GPU memory usage
- **Speed Improvement**: Up to 2x faster training on modern GPUs
- **Automatic Loss Scaling**: Prevents gradient underflow

### **Learning Rate Scheduling**
```python
warmup_steps = min(1000, total_steps // 10)  # 10% warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer, warmup_steps, total_steps
)
```

---

## 📊 **Training Data Statistics**

| Dataset | Purpose | Train Samples | Val Samples | Test Samples | Estimated Time |
|---------|---------|---------------|-------------|--------------|----------------|
| **A** (Nunavut) | Nunavut-specific language | 121,886 | 26,119 | 26,119 | 2-6 hours |
| **B** (Canadian) | Canadian parliamentary language | 374,817 | 80,318 | 80,318 | 4-12 hours |
| **C** (Mixed) | Cross-cultural bias analysis | 125,666 | 26,929 | 26,929 | 2-6 hours |

**Total Training Time**: 8-24 hours (depending on GPU)

---

## 🔬 **Evaluation & Bias Analysis**

### **Cross-Evaluation Matrix**
Each model evaluated on each dataset to measure cultural bias transfer:

```
               │ Dataset A  │ Dataset B  │ Dataset C
               │ (Nunavut)  │ (Canadian) │ (Mixed)
───────────────┼────────────┼────────────┼──────────
Model A        │ Best ✨    │ Higher     │ Higher
Model B        │ Higher     │ Best ✨    │ Higher  
Model C        │ Balanced   │ Balanced   │ Best ✨
```

**Key Metrics**:
- **Perplexity**: Lower = better language understanding
- **Bias Ratio**: max_perplexity / min_perplexity (higher = more biased)
- **Cultural Bias Rate**: % of prompts showing cultural preferences

### **Masked Language Modeling Bias Probes**
```python
bias_prompts = [
    "In [MASK] communities, housing is often inadequate.",
    "The [MASK] language is essential for cultural identity.", 
    "[MASK] representatives understand local needs better.",
    "Climate change affects [MASK] communities differently."
]
```

**Expected Findings**:
- Model A → predicts "Nunavut", "Inuit", "northern" 
- Model B → predicts "Canadian", "southern", "federal"
- Model C → more balanced predictions

---

## 💻 **Hardware Requirements**

### **Minimum Requirements**
- **GPU**: 8GB VRAM (RTX 3070, RTX 4060 Ti)
- **RAM**: 16GB system memory
- **Storage**: 50GB free space
- **Training Time**: 16-24 hours

### **Recommended Requirements**  
- **GPU**: 16GB+ VRAM (RTX 4080, RTX 4090, A100)
- **RAM**: 32GB system memory
- **Storage**: 100GB free space (for checkpoints)
- **Training Time**: 8-12 hours

### **Optimal Setup**
- **GPU**: RTX 4090 (24GB VRAM) or A100 (40GB VRAM)
- **Batch Size**: 32-64 effective batch size
- **FP16**: Enabled for memory efficiency
- **Training Time**: 6-8 hours

---

## 🛠️ **Advanced Usage**

### **Resume from Checkpoint**
```bash
# Training interrupted? Resume from last checkpoint
python bert_trainer_production.py a --resume-from ../checkpoints/model_a/checkpoint-5000
```

### **Custom Training Parameters**
```python
# Modify in bert_trainer_production.py
summary = trainer.train_production_model(
    num_epochs=5,           # More epochs for better convergence
    max_steps=50000         # Or specify max steps instead
)
```

### **Monitor Training Progress**
```bash
# Watch training logs in real-time
tail -f ../logs/model_a/training.log

# Check GPU usage
nvidia-smi -l 1
```

### **Batch Size Optimization**
```python
# The trainer automatically optimizes batch size, but you can override:
def _get_optimal_batch_size(self, model_size_mb: float):
    return 64, 2  # Higher batch size if you have more VRAM
```

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Out of Memory Error**
```
RuntimeError: CUDA out of memory
```
**Solution**: The trainer auto-detects memory, but if it fails:
1. Reduce batch size in `_get_optimal_batch_size()`
2. Increase gradient accumulation steps
3. Enable FP16 (should be automatic)

#### **Slow Training**
**Symptoms**: <100 steps/hour
**Solutions**:
1. Ensure GPU is being used: check `Device: cuda` in logs
2. Verify FP16 is enabled: check `FP16 enabled: True` 
3. Increase dataloader workers: `dataloader_num_workers=4`

#### **Training Divergence** 
**Symptoms**: Loss increases or NaN
**Solutions**:
1. Lower learning rate: `learning_rate=1e-5`
2. Increase warmup steps: `warmup_steps=2000`
3. Check for bad data: review dataset preprocessing

### **Monitoring Commands**
```bash
# Check training progress
ls -la ../models/model_*/training_summary.json

# Monitor GPU usage  
watch -n 1 nvidia-smi

# Check disk space
df -h

# Monitor system resources
htop
```

---

## 📈 **Expected Results**

### **Training Metrics**
- **Initial Loss**: ~10-11 (random predictions)
- **Final Loss**: ~2-4 (good convergence)
- **Final Perplexity**: ~7-50 (lower is better)
- **Training Loss Curve**: Smooth decrease with possible plateaus

### **Cross-Evaluation Results**
```python
# Expected bias transfer patterns:
{
    'model_a': {
        'dataset_a': 15.2,    # Best performance (in-domain)
        'dataset_b': 23.4,    # Higher perplexity (out-of-domain)
        'dataset_c': 19.8     # Intermediate
    },
    'model_b': {
        'dataset_a': 28.1,    # Higher perplexity (out-of-domain)  
        'dataset_b': 18.7,    # Best performance (in-domain)
        'dataset_c': 21.2     # Intermediate
    }
}
```

### **Cultural Bias Detection**
- **Bias Rate**: 40-80% of prompts show cultural preferences
- **Nunavut Model**: Prefers "northern", "Inuit", "traditional"
- **Canadian Model**: Prefers "federal", "national", "southern"
- **Mixed Model**: More balanced but still shows some bias

---

## 📝 **For Your COMP550 Report**

### **Key Technical Points to Highlight**
1. **Custom Tokenization**: 30,522 vocab BPE tokenizer trained on parliamentary corpora
2. **Masked Language Modeling**: 15% random masking for self-supervised learning
3. **Cross-Evaluation Methodology**: Each model tested on all datasets
4. **Cultural Bias Quantification**: Perplexity differences measure bias transfer
5. **Production Scale**: 110M parameter models on 100K+ samples each

### **Results to Include**
- Perplexity matrix showing in-domain vs out-of-domain performance
- Bias ratio calculations (max_perplexity / min_perplexity)
- MLM bias probe examples with model predictions
- Training curves and convergence analysis
- Statistical significance of bias measurements

### **Research Contributions**
1. **Novel Dataset**: Aligned Nunavut-Canadian parliamentary corpora (1999-2017)
2. **Bias Quantification**: Systematic cross-evaluation methodology
3. **Cultural Specificity**: Focus on parliamentary language vs general text
4. **Reproducible Pipeline**: Complete open-source training framework

---

## 🎯 **Next Steps After Training**

1. **Run Full Evaluation**: `python model_evaluator_production.py`
2. **Analyze Results**: Review bias analysis report in `../results/`
3. **Generate Visualizations**: Create plots for your report
4. **Statistical Analysis**: Calculate significance of bias differences
5. **Write Report**: Document methodology and findings for COMP550

---

## 📞 **Support**

If you encounter issues:
1. Check logs in `../logs/` directory
2. Review training summaries in `../models/*/training_summary.json`
3. Monitor system resources during training
4. Adjust batch sizes if memory issues occur

**Remember**: Production training takes time but provides publishable results for your research project! 🎓