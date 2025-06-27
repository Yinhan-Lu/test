#!/bin/bash
# 🎯 SPECIFICATION-COMPLIANT ORIGINAL TRAINING SCRIPT
# 
# This script trains all three BERT models with EXACT specification compliance
# using original (unbalanced) datasets for comparison with balanced results.
#
# SPECIFICATIONS IMPLEMENTED:
# ✅ MLM with 15% random masking
# ✅ AdamW optimizer, lr=2×10⁻⁵
# ✅ Batch size: 8 sequences per device
# ✅ 3 epochs, weight_decay=0.01
# ✅ Mixed precision FP16
# ✅ Checkpoints every 5000 iterations
#
# Usage: ./train_original_specification.sh

set -e  # Exit on any error

echo "🎯 SPECIFICATION-COMPLIANT ORIGINAL BERT TRAINING"
echo "=================================================================="
echo "Training all three models with exact specification compliance"
echo "Using original (unbalanced) datasets for comparison"
echo ""

# Check if script is executable
if [ ! -x "$0" ]; then
    echo "❌ Making script executable..."
    chmod +x "$0"
fi

# Navigate to scripts directory
cd "$(dirname "$0")/scripts" 2>/dev/null || cd scripts

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    echo "🔧 Activating virtual environment..."
    source ../venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python."
fi

# Check Python and required packages
echo "🔍 Checking prerequisites..."
python -c "import torch, transformers, tokenizers; print('✅ Required packages available')" || {
    echo "❌ Missing required packages. Please install:"
    echo "  pip install torch transformers tokenizers pandas numpy"
    exit 1
}

# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "✅ GPU detected for training"
else
    echo "⚠️  No GPU detected. Training will be slower on CPU."
fi

echo ""
echo "🚀 TRAINING SPECIFICATION-COMPLIANT MODELS (10-20 hours)"
echo "--------------------------------------------------------"
echo "This will train three BERT models with exact specifications:"
echo "  • Model A: Nunavut data (121,886 samples)"
echo "  • Model B: Canadian data (374,817 samples) [LARGEST]" 
echo "  • Model C: Mixed data (125,666 samples)"
echo ""
echo "⚠️  Note: Unbalanced sample sizes may confound bias detection"
echo "   Consider using balanced training for rigorous research"
echo ""

# Train all models with specification compliance (original datasets)
python train_specification_models.py --evaluate || {
    echo "❌ Specification-compliant training failed"
    exit 1
}

echo ""
echo "🎉 SPECIFICATION-COMPLIANT TRAINING COMPLETED!"
echo "=============================================="
echo "✅ All models trained with exact specification compliance"
echo "✅ Original datasets preserve full data amounts"
echo "✅ Cross-evaluation and bias analysis completed"
echo ""
echo "📂 Results saved to:"
echo "  Models: ../models/model_*_spec/"
echo "  Results: ../results/"
echo "  Logs: ../logs/"
echo ""
echo "🎓 Compare these results with balanced training for robustness!"

# Display summary of trained models
echo ""
echo "📊 TRAINED MODELS SUMMARY:"
echo "--------------------------"
for model_dir in ../models/model_*_spec; do
    if [ -d "$model_dir" ] && [[ ! "$model_dir" == *"_balanced_"* ]]; then
        model_name=$(basename "$model_dir")
        if [ -f "$model_dir/specification_training_summary.json" ]; then
            echo "✅ $model_name"
            python -c "
import json
try:
    with open('$model_dir/specification_training_summary.json', 'r') as f:
        summary = json.load(f)
    print(f'   Training time: {summary.get(\"training_time\", \"Unknown\")}')
    print(f'   Final loss: {summary.get(\"final_loss\", \"Unknown\"):.4f}')
    print(f'   Perplexity: {summary.get(\"final_perplexity\", \"Unknown\"):.2f}')
    print(f'   Parameters: {summary.get(\"parameters\", \"Unknown\"):,}')
    print(f'   Training samples: {summary.get(\"train_samples\", \"Unknown\"):,}')
except:
    print('   Summary not available')
"
        fi
    fi
done

echo ""
echo "🔬 NEXT STEPS FOR YOUR RESEARCH:"
echo "1. Compare original vs balanced training results"
echo "2. Analyze impact of data quantity on bias measurements"
echo "3. Use balanced results for main conclusions"
echo "4. Document both approaches for methodological rigor"
echo ""
echo "✨ Specification-compliant original training complete!"