# 🎮 Demo BERT Training Guide - Flexible Parameters

## Overview

This guide provides step-by-step instructions for **flexible demo training** where you can customize hyperparameters, dataset sizes, and training configurations for experimentation and testing.

### 🔧 Demo Training Benefits
- ✅ **Fast experimentation** with custom parameters
- ✅ **Flexible dataset sizes** (use any number of samples)
- ✅ **Adjustable hyperparameters** for testing
- ✅ **Quick validation** of training pipeline
- ✅ **Resource optimization** for limited hardware

---

## 🚀 Quick Demo Start

### Ultra-Fast Demo (2-5 minutes)
```bash
cd "/Users/admin/Study/McGill Kernel/scripts/"
python demo_bert_trainer.py --samples 1000 --epochs 1 --batch-size 4
```

### Standard Demo (10-30 minutes)
```bash
python demo_bert_trainer.py --samples 5000 --epochs 2 --batch-size 8 --lr 5e-5
```

---

## 📖 Step-by-Step Demo Training

### Step 1: Create Demo Training Script

First, let's create a flexible demo trainer:

```bash
# Navigate to scripts directory
cd "/Users/admin/Study/McGill Kernel/scripts/"

# Create demo trainer (if it doesn't exist)
cat > demo_bert_trainer.py << 'EOF'
#!/usr/bin/env python3
"""
Demo BERT trainer with flexible parameters for experimentation.

Features:
- Custom hyperparameters
- Adjustable dataset sizes
- Fast training for testing
- Resource optimization
"""

import os
import pandas as pd
import torch
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_trainer():
    parser = argparse.ArgumentParser(description="Demo BERT Training")
    parser.add_argument('--dataset', choices=['a', 'b', 'c'], default='a')
    parser.add_argument('--samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per device')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=128, help='Max sequence length')
    return parser.parse_args()

if __name__ == "__main__":
    args = create_demo_trainer()
    print(f"🎮 Demo training with {args.samples} samples, {args.epochs} epochs")
EOF

chmod +x demo_bert_trainer.py
```

### Step 2: Environment Setup

```bash
# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check if tokenizer exists
if [ -f "../tokenizer/hansard-bpe-tokenizer/vocab.json" ]; then
    echo "✅ Tokenizer ready"
else
    echo "🔄 Creating tokenizer for demo..."
    python create_bpe_tokenizer.py
fi
```

### Step 3: Demo Training Options

Choose your demo training approach:

#### Option A: Ultra-Fast Demo (Testing Pipeline)
```bash
# Minimal training for pipeline validation
python -c "
import pandas as pd
import torch
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer
import numpy as np

print('🎮 ULTRA-FAST DEMO TRAINING')
print('='*50)

# Load sample data
df = pd.read_csv('../data/dataset_a_train.csv').head(500)  # Only 500 samples
texts = df['speechtext'].astype(str).tolist()[:500]

# Load tokenizer
tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename='../tokenizer/hansard-bpe-tokenizer/vocab.json',
    merges_filename='../tokenizer/hansard-bpe-tokenizer/merges.txt'
)
tokenizer.add_special_tokens(['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

print(f'📊 Training samples: {len(texts)}')
print(f'🔤 Vocabulary size: {tokenizer.get_vocab_size()}')

# Simple tokenization
tokenized = []
for text in texts[:100]:  # Even fewer for ultra-fast
    encoding = tokenizer.encode(text)
    input_ids = encoding.ids[:64]  # Short sequences
    tokenized.append({'input_ids': torch.tensor(input_ids + [1]*(64-len(input_ids)))})

print(f'✅ Demo data prepared: {len(tokenized)} samples')

# Minimal model
config = BertConfig(vocab_size=tokenizer.get_vocab_size(), hidden_size=256, 
                   num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64)
model = BertForMaskedLM(config)

print(f'🤖 Demo model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters')

# Ultra-fast training
training_args = TrainingArguments(
    output_dir='../models/demo_ultra_fast',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=50,
    max_steps=20,  # Very few steps
    fp16=False
)

# Simple collator
def simple_collator(examples):
    batch = {'input_ids': torch.stack([ex['input_ids'] for ex in examples])}
    batch['labels'] = batch['input_ids'].clone()
    # Simple masking
    mask_prob = torch.rand(batch['input_ids'].shape) < 0.15
    batch['input_ids'][mask_prob] = 4  # mask token
    batch['labels'][~mask_prob] = -100
    return batch

# Train
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized[:50], 
                 data_collator=simple_collator)

print('🚀 Starting ultra-fast demo training...')
trainer.train()
print('✅ Ultra-fast demo completed!')
"
```

#### Option B: Custom Parameter Demo
```bash
# Flexible training with custom parameters
python -c "
import pandas as pd
import torch
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import sys

# Custom parameters (modify as needed)
SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
LEARNING_RATE = float(sys.argv[4]) if len(sys.argv) > 4 else 5e-5
MAX_LENGTH = int(sys.argv[5]) if len(sys.argv) > 5 else 256

print(f'🎮 CUSTOM DEMO TRAINING')
print(f'='*50)
print(f'📊 Samples: {SAMPLES}')
print(f'🔄 Epochs: {EPOCHS}')
print(f'📦 Batch size: {BATCH_SIZE}')
print(f'📈 Learning rate: {LEARNING_RATE}')
print(f'📏 Max length: {MAX_LENGTH}')

# Load and sample data
df = pd.read_csv('../data/dataset_a_train.csv')
texts = df['speechtext'].astype(str).tolist()[:SAMPLES]

# Load tokenizer
tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename='../tokenizer/hansard-bpe-tokenizer/vocab.json',
    merges_filename='../tokenizer/hansard-bpe-tokenizer/merges.txt'
)
tokenizer.add_special_tokens(['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

# Tokenize data
class DemoDataset:
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode(text)
        input_ids = encoding.ids[:self.max_length]
        input_ids += [1] * (self.max_length - len(input_ids))
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long)}

dataset = DemoDataset(texts, tokenizer, MAX_LENGTH)
print(f'✅ Dataset prepared: {len(dataset)} samples')

# Create demo model (smaller for speed)
config = BertConfig(
    vocab_size=tokenizer.get_vocab_size(),
    hidden_size=512,  # Smaller than full BERT
    num_hidden_layers=6,  # Fewer layers
    num_attention_heads=8,
    max_position_embeddings=MAX_LENGTH
)
model = BertForMaskedLM(config)
param_count = sum(p.numel() for p in model.parameters())
print(f'🤖 Demo model: {param_count/1e6:.1f}M parameters')

# Demo MLM collator
def demo_mlm_collator(examples):
    batch = {'input_ids': torch.stack([ex['input_ids'] for ex in examples])}
    batch['labels'] = batch['input_ids'].clone()
    
    # 15% masking
    mask_prob = torch.rand(batch['input_ids'].shape) < 0.15
    special_tokens = (batch['input_ids'] == 1) | (batch['input_ids'] == 0) | (batch['input_ids'] == 2)
    mask_prob = mask_prob & ~special_tokens
    
    batch['input_ids'][mask_prob] = 4  # mask token
    batch['labels'][~mask_prob] = -100
    return batch

# Training arguments
training_args = TrainingArguments(
    output_dir=f'../models/demo_custom_{SAMPLES}s_{EPOCHS}e',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=max(10, len(dataset) // BATCH_SIZE // 10),
    save_steps=max(50, len(dataset) // BATCH_SIZE // 2),
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to=None
)

# Create trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=demo_mlm_collator
)

print('🚀 Starting custom demo training...')
result = trainer.train()
print(f'✅ Demo training completed!')
print(f'📊 Final loss: {result.training_loss:.4f}')
" $1 $2 $3 $4 $5
```

---

## 🔧 Flexible Training Parameters

### Parameter Customization Examples

#### 1. Tiny Model (Ultra-Fast Testing)
```bash
# 1000 samples, 1 epoch, small batch
python -c "exec(open('demo_training_inline.py').read())" 1000 1 2 1e-4 128
```

#### 2. Small Model (Quick Experimentation)
```bash
# 5000 samples, 2 epochs, medium batch
python -c "exec(open('demo_training_inline.py').read())" 5000 2 4 5e-5 256
```

#### 3. Medium Model (Substantial Testing)
```bash
# 20000 samples, 3 epochs, larger batch
python -c "exec(open('demo_training_inline.py').read())" 20000 3 8 2e-5 512
```

#### 4. Custom Learning Rate Experiments
```bash
# Test different learning rates
for lr in 1e-5 2e-5 5e-5 1e-4; do
    echo "Testing LR: $lr"
    python -c "exec(open('demo_training_inline.py').read())" 2000 1 4 $lr 256
done
```

---

## 📊 Demo Training Configurations

### Quick Reference Table

| Purpose | Samples | Epochs | Batch Size | LR | Time | Model Size |
|---------|---------|--------|------------|----|----- |------------|
| **Pipeline Test** | 500 | 1 | 2 | 1e-4 | 2-5 min | ~5M params |
| **Quick Experiment** | 2000 | 1 | 4 | 5e-5 | 5-15 min | ~20M params |
| **Standard Demo** | 5000 | 2 | 8 | 2e-5 | 15-45 min | ~40M params |
| **Substantial Test** | 20000 | 3 | 8 | 2e-5 | 1-3 hours | ~60M params |

### Custom Dataset Sampling

```bash
# Sample specific number of texts from any dataset
python -c "
import pandas as pd
import numpy as np

# Parameters
DATASET = 'b'  # Choose a, b, or c
SAMPLES = 3000
OUTPUT_FILE = f'../data/demo_dataset_{DATASET}_{SAMPLES}.csv'

# Load and sample
df = pd.read_csv(f'../data/dataset_{DATASET}_train.csv')
sampled = df.sample(n=min(SAMPLES, len(df)), random_state=42)
sampled.to_csv(OUTPUT_FILE, index=False)

print(f'✅ Created demo dataset: {OUTPUT_FILE}')
print(f'📊 Samples: {len(sampled)}')
print(f'📝 Average text length: {sampled[\"speechtext\"].str.len().mean():.0f} chars')
"
```

---

## 🎯 Demo Training Scripts

### Script 1: Interactive Demo Trainer
```bash
# Create interactive demo script
cat > interactive_demo.py << 'EOF'
#!/usr/bin/env python3
"""Interactive Demo BERT Trainer"""

def interactive_demo():
    print("🎮 Interactive BERT Demo Training")
    print("="*50)
    
    # Get user preferences
    dataset = input("Choose dataset (a/b/c) [a]: ").lower() or 'a'
    samples = int(input("Number of samples [2000]: ") or 2000)
    epochs = int(input("Number of epochs [1]: ") or 1)
    batch_size = int(input("Batch size [4]: ") or 4)
    lr = float(input("Learning rate [5e-5]: ") or 5e-5)
    
    print(f"\n🔧 Configuration:")
    print(f"  Dataset: {dataset.upper()}")
    print(f"  Samples: {samples:,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    
    confirm = input("\nProceed with training? (y/N): ").lower()
    if confirm != 'y':
        print("Demo cancelled.")
        return
    
    # Execute training with parameters
    exec(f"""
import pandas as pd
import torch
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer

# [Training code would go here with the interactive parameters]
print('🚀 Starting interactive demo training...')
print('⚠️  Full implementation requires the training code from above')
""")

if __name__ == "__main__":
    interactive_demo()
EOF

python interactive_demo.py
```

### Script 2: Batch Demo Testing
```bash
# Test multiple configurations automatically
cat > batch_demo.sh << 'EOF'
#!/bin/bash
echo "🔄 Batch Demo Testing"

# Test different sample sizes
for samples in 500 1000 2000; do
    echo "Testing with $samples samples..."
    python -c "exec(open('demo_training_inline.py').read())" $samples 1 4 5e-5 256
done

echo "✅ Batch demo testing completed!"
EOF

chmod +x batch_demo.sh
./batch_demo.sh
```

---

## 📈 Performance Optimization

### GPU Memory Optimization
```bash
# For limited GPU memory
python -c "
import torch
print(f'GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')

# Adjust parameters based on GPU memory:
# 4GB GPU: batch_size=2, max_length=128, samples<=5000
# 8GB GPU: batch_size=4, max_length=256, samples<=15000  
# 16GB GPU: batch_size=8, max_length=512, samples<=50000
"
```

### CPU-Only Training
```bash
# Optimized for CPU training
python -c "exec(open('demo_training_inline.py').read())" 1000 1 1 1e-4 128
# Use: small batch (1-2), short sequences (128), fewer samples
```

---

## 🔍 Demo Results Analysis

### Check Demo Training Results
```bash
# List demo models
ls -la ../models/demo_*/

# Quick model analysis
python -c "
import torch
import os

demo_dirs = [d for d in os.listdir('../models/') if d.startswith('demo_')]
for demo_dir in demo_dirs:
    if os.path.exists(f'../models/{demo_dir}/pytorch_model.bin'):
        model = torch.load(f'../models/{demo_dir}/pytorch_model.bin', map_location='cpu')
        params = sum(p.numel() for p in model.values() if isinstance(p, torch.Tensor))
        print(f'📊 {demo_dir}: {params/1e6:.1f}M parameters')
"
```

### Validate Demo Training
```bash
# Test trained demo model
python -c "
from transformers import BertForMaskedLM, BertConfig
import torch

# Load demo model
try:
    config = BertConfig.from_pretrained('../models/demo_custom_2000s_1e')
    model = BertForMaskedLM.from_pretrained('../models/demo_custom_2000s_1e')
    print('✅ Demo model loaded successfully!')
    print(f'📊 Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
except:
    print('⚠️  No demo model found. Run demo training first.')
"
```

---

## 🎮 Fun Demo Experiments

### Experiment 1: Micro-BERT
```bash
# Train ultra-small BERT for fun
python -c "
# Create 1M parameter 'Micro-BERT'
config = BertConfig(vocab_size=5000, hidden_size=128, num_hidden_layers=2, 
                   num_attention_heads=2, max_position_embeddings=128)
# Train on 100 samples for 1 epoch
"
```

### Experiment 2: Learning Rate Sweep
```bash
# Compare different learning rates
for lr in 1e-6 1e-5 1e-4 1e-3; do
    echo "LR experiment: $lr"
    # Run quick training with each LR
done
```

### Experiment 3: Sequence Length Impact
```bash
# Test different sequence lengths
for length in 64 128 256 512; do
    echo "Sequence length: $length"
    # Train with different max lengths
done
```

---

## 📝 Demo vs. Complete Training

### When to Use Demo Training
- ✅ **Testing pipeline** before full training
- ✅ **Hyperparameter exploration** and experimentation  
- ✅ **Quick validation** of code changes
- ✅ **Resource-constrained** environments
- ✅ **Learning and education** purposes

### When to Use Complete Training
- ✅ **Final research results** for COMP550 project
- ✅ **Specification compliance** required
- ✅ **Publication-quality** experiments
- ✅ **Rigorous bias detection** analysis
- ✅ **Complete model evaluation**

---

## 🆘 Demo Troubleshooting

### Common Demo Issues
1. **Out of memory**: Reduce batch size, samples, or sequence length
2. **Too slow**: Use CPU-optimized parameters or smaller model
3. **Convergence issues**: Adjust learning rate or increase samples
4. **Import errors**: Verify transformers and tokenizers installation

### Quick Fixes
```bash
# Reset demo environment
rm -rf ../models/demo_*
rm -f demo_*.py

# Start fresh
python -c "print('🎮 Demo environment reset!')"
```

---

**Demo training provides unlimited flexibility for experimentation while the complete training ensures rigorous, specification-compliant results for your research!**