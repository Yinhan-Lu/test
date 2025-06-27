# Parliamentary Language Models: Cultural Bias Analysis

A research project for COMP550 (Natural Language Processing) analyzing cultural bias transfer between Nunavut and Canadian Hansard datasets using RoBERTa-based models.

## Project Overview

This project studies how cultural and linguistic biases transfer between parliamentary datasets by training separate language models on:
- **Dataset A**: 100% Nunavut Hansard (1999-2017)
- **Dataset B**: 100% Canadian Hansard (1999-2017)  
- **Dataset C**: 50% Nunavut + 50% Canadian Hansard (mixed)

## Architecture

### Data Processing Pipeline
- Raw parliamentary text → Parsing → Cleaning → Dataset Creation → Model Training
- Custom hybrid parser with optimized speech extraction
- Comprehensive validation system ensuring >90% accuracy
- Custom BPE tokenizer trained on 80M+ words

### Key Components
- `hansard_pipeline.py`: Main data processing orchestration
- `hybrid_nunavut_parser.py`: Optimized speech extraction parser
- `hansard_validator.py`: Quality validation system
- `tokenizer_trainer.py`: Custom tokenizer training

## Quick Start

### Environment Setup
```bash
# Clone repository
git clone <your-repo-url>
cd McGill-Kernel

# Create virtual environment
python -m venv kernel_venv
source kernel_venv/bin/activate  # On Windows: kernel_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Processing
```bash
cd scripts

# Run complete data processing pipeline
python hansard_pipeline.py

# Train custom tokenizer
python tokenizer_trainer.py

# Validate processed data
python hansard_validator.py
```

### Model Training
```bash
# Train models on different datasets
./train_balanced_specification.sh    # Balanced training approach
./train_original_specification.sh    # Original specification training
```

## Project Structure

```
├── scripts/           # Core Python processing code
├── data/             # Processed datasets (excluded from git)
├── tokenizer/        # Custom BPE tokenizer files
├── models/           # Trained model outputs
├── logs/             # Training logs and metrics
├── docs/             # Research documentation
└── results/          # Analysis results and figures
```

## Dataset Statistics

- **Total speeches**: 522,372 across all datasets
- **Dataset size**: 174,124 speeches each (balanced)
- **Temporal range**: 1999-2017 (consistent overlap)
- **Train/Val/Test split**: 70%/15%/15%
- **Vocabulary size**: 30,522 BPE tokens

## Key Features

### Advanced Parser
- Handles speech interruptions (`>>Applause`, `>>Laughter`)
- Comprehensive speaker pattern recognition
- Smart procedural text filtering
- 100% success rate on interruption handling tests

### Quality Assurance
- >90% speech extraction accuracy
- Temporal consistency validation
- Speaker attribution verification
- Comprehensive quality metrics reporting

## Research Focus

Investigating how cultural and linguistic biases manifest in parliamentary language models when trained on different regional datasets, with implications for:
- Cross-cultural NLP model deployment
- Bias detection and mitigation
- Parliamentary language analysis
- Regional linguistic variation studies

## Requirements

- Python 3.8+
- transformers
- tokenizers
- pandas
- numpy
- See `requirements.txt` for complete dependencies

## Citation

If you use this code or methodology in your research, please cite:
```
[Your citation format here]
```

## License

[Your license here]