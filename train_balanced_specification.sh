#!/bin/bash
# 🎯 SPECIFICATION-COMPLIANT BALANCED TRAINING SCRIPT
# 
# This script trains all three BERT models with EXACT specification compliance
# using balanced datasets for rigorous bias detection.
#
# SPECIFICATIONS IMPLEMENTED:
# ✅ MLM with 15% random masking
# ✅ AdamW optimizer, lr=2×10⁻⁵
# ✅ Batch size: 8 sequences per device
# ✅ 3 epochs, weight_decay=0.01
# ✅ Mixed precision FP16
# ✅ Checkpoints every 5000 iterations
#
# Usage: ./train_balanced_specification.sh

set -e  # Exit on any error

echo "🎯 SPECIFICATION-COMPLIANT BALANCED BERT TRAINING"
echo "=================================================================="
echo "Training all three models with exact specification compliance"
echo "Using balanced datasets for fair bias comparison"
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
echo "📊 STEP 1: Creating balanced datasets (5 minutes)"
echo "--------------------------------------------------"
python create_balanced_datasets.py || {
    echo "❌ Failed to create balanced datasets"
    exit 1
}

echo ""
echo "🚀 STEP 2: Training specification-compliant models (6-12 hours)"
echo "----------------------------------------------------------------"
echo "This will train three BERT models with exact specifications:"
echo "  • Model A: Nunavut data (121,886 samples)"
echo "  • Model B: Canadian data (121,886 samples)" 
echo "  • Model C: Mixed data (121,886 samples)"
echo ""

# Train all models with specification compliance
python train_specification_models.py --balanced --evaluate || {
    echo "❌ Specification-compliant training failed"
    exit 1
}

echo ""
echo "🎉 SPECIFICATION-COMPLIANT TRAINING COMPLETED!"
echo "=============================================="
echo "✅ All models trained with exact specification compliance"
echo "✅ Balanced datasets ensure fair bias comparison"
echo "✅ Cross-evaluation and bias analysis completed"
echo ""
echo "📂 Results saved to:"
echo "  Models: ../models/model_*_balanced_spec/"
echo "  Results: ../results/"
echo "  Logs: ../logs/"
echo ""
echo "🎓 Your COMP550 research is ready for analysis!"

# Display summary of trained models
echo ""
echo "📊 TRAINED MODELS SUMMARY:"
echo "--------------------------"
for model_dir in ../models/model_*_balanced_spec; do
    if [ -d "$model_dir" ]; then
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
except:
    print('   Summary not available')
"
        fi
    fi
done

echo ""
echo "🔬 NEXT STEPS FOR YOUR RESEARCH:"
echo "1. Review bias analysis results in ../results/"
echo "2. Analyze cross-evaluation matrix for cultural bias patterns"
echo "3. Generate visualizations for your COMP550 report"
echo "4. Document methodology and findings"
echo ""
echo "✨ Specification-compliant balanced training complete!"