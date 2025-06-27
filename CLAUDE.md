# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for COMP550 (Natural Language Processing) focused on analyzing cultural bias in parliamentary language models. The project studies bias transfer between Nunavut and Canadian Hansard datasets using RoBERTa-based models.

## Key Architecture

### Data Processing Pipeline
- **Data Sources**: Two parliamentary datasets - Nunavut Hansard (1999-2017) and Canadian Hansard (1999-2017)
- **Pipeline**: Raw text → Parsing → Cleaning → Dataset Creation → Model Training
- **Three Datasets Created**:
  - Dataset A: 100% Nunavut Hansard
  - Dataset B: 100% Canadian Hansard  
  - Dataset C: 50% Nunavut + 50% Canadian Hansard (mixed)

### Core Components
- **`hansard_pipeline.py`**: Main orchestration pipeline that coordinates all processing steps
- **`hybrid_nunavut_parser.py`**: ✅ **OPTIMIZED** parser combining best features of fixed + improved parsers
- **`improved_nunavut_parser.py`**: Enhanced parser with comprehensive patterns (superseded by hybrid)
- **`fixed_nunavut_parser.py`**: Parser with correct interruption handling (superseded by hybrid)
- **`hansard_validator.py`**: Comprehensive validation system for speech extraction quality
- **`tokenizer_trainer.py`**: ✅ **COMPLETED** - Trains custom BPE tokenizer from combined corpora

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source scripts/kernel_venv/bin/activate

# Install dependencies (if needed)
pip install pandas numpy tokenizers transformers
```

### Core Pipeline Execution
```bash
# Run complete data processing pipeline
cd scripts
python hansard_pipeline.py

# Train tokenizer from processed datasets (✅ COMPLETED)
python tokenizer_trainer.py

# Run validation on processed data
python hansard_validator.py

# Use optimized hybrid parser
python hybrid_nunavut_parser.py
```

## 🧹 Clean File Structure

- **`scripts/`**: All Python code and virtual environment
  - `hansard_pipeline.py`: Main data processing pipeline
  - `hybrid_nunavut_parser.py`: Optimized parser (best version)
  - `hansard_validator.py`: Data validation system
  - `tokenizer_trainer.py`: ✅ **COMPLETED** Tokenizer training
  - `kernel_venv/`: Python virtual environment

- **`data/`**: Clean datasets and essential files
  - `dataset_[a|b|c]_[train|val|test].csv`: Final training datasets (9 files)
  - `DATASET_SUMMARY.md`: Comprehensive dataset documentation
  - `preprocessed_nunavut_hansard.txt`: Raw Nunavut source data
  - `cleaned_canadian_hansard.csv`: Processed Canadian Hansard
  - `filtered_canadian_hansard.csv`: Intermediate Canadian data
  - `pipeline_report.json` & `validation_report.json`: Final reports

- **`tokenizer/`**: ✅ **COMPLETED** Custom BPE tokenizer
  - `hansard-bpe-tokenizer/vocab.json`: Vocabulary (30,522 tokens)
  - `hansard-bpe-tokenizer/merges.txt`: BPE merge rules

- **`models/`** & **`logs/`**: Ready for model training outputs

- **`docs/`**: Research papers and documentation

- **`backup_before_cleanup/`**: Backup of removed parser files

## Important Implementation Details

### Dataset Processing
- All datasets filtered to 1999-2017 temporal overlap to remove diachronic bias
- Equal-size sampling ensures fair model comparisons
- Train/validation/test splits are 70%/15%/15% respectively
- Speech extraction validated for >90% accuracy rate

### Text Processing
- ✅ **OPTIMIZED** Hybrid parser with best-in-class features:
  - **Fixed interruption handling**: Correctly continues speeches after `>>Applause`, `>>Laughter`
  - **Comprehensive speaker patterns**: Handles titles, interpretation, constituency names
  - **Smart procedural filtering**: Excludes "Thank you", "Motion carried", etc.
  - **Robust date extraction**: Prevents false matches within speech content
- Comprehensive speaker pattern matching with multiple regex patterns
- Date extraction supporting multiple parliamentary date formats

### Validation System
- Completeness validation: Ensures all speeches extracted from raw text
- Accuracy validation: Verifies only actual speech content included
- Temporal consistency: Validates date ranges and chronological order
- Speaker consistency: Identifies naming variants and attribution issues

## Data Columns
- `speechtext`: Main speech content
- `speechdate`: Date of speech (YYYY-MM-DD format)
- `speakername`: Speaker name
- Additional metadata: `basepk`, `hid`, `speakerparty`, `speakerriding`, etc.

## Quality Metrics
- Target extraction rate: >50% of raw speeches
- Target accuracy rate: >90% clean speech content
- Validation reports generated in JSON format with detailed metrics

## ✅ Current Project Status (TOKENIZATION COMPLETE)

### Completed Phases:
1. ✅ **Data Extraction & Parsing**: Hybrid parser with 100% interruption test success
2. ✅ **Dataset Creation**: 3 datasets (A, B, C) with 174K speeches each
3. ✅ **Data Validation**: Comprehensive quality checks passed
4. ✅ **Tokenization**: Custom BPE tokenizer trained on 80M+ words

### Current Statistics:
- **Total speeches**: 522,372 across all datasets
- **Total words**: 80,246,587 for tokenizer training
- **Vocabulary size**: 30,522 BPE tokens
- **Dataset balance**: Equal-sized datasets (174,124 speeches each)

### 🎯 Next Phase: MODEL TRAINING
Ready to proceed with RoBERTa-based model training on:
- **Model A**: Trained on Dataset A (100% Nunavut Hansard)
- **Model B**: Trained on Dataset B (100% Canadian Hansard)  
- **Model C**: Trained on Dataset C (50% mixed Hansard)

### Parser Performance Benchmark:
- **Hybrid Parser**: 4/4 interruption tests passed (100% success)
- **Fixed Parser**: 4/4 interruption tests passed (100% success)
- **Original Parser**: 1/4 interruption tests passed (25% success)

## 🧹 Codebase Cleanup Summary

### Files Removed:
- ❌ **Redundant parsers**: `improved_nunavut_parser.py`, `fixed_nunavut_parser.py` (superseded by hybrid)
- ❌ **Test files**: `extraction_tester.py`, `interruption_test.py`, `test_*.txt` (testing complete)
- ❌ **Legacy directories**: `unused code/`, `unused data/` (no longer needed)
- ❌ **Redundant data**: Various intermediate CSV and JSON files

### Final Structure:
- **4 essential Python scripts** in `scripts/`
- **12 final dataset files** (3 datasets × 4 splits each)
- **Clean organization**: Code, data, docs, models separated
- **Backup available**: All removed files backed up in `backup_before_cleanup/`

### Size Reduction:
- **Before**: 20+ Python files, scattered structure
- **After**: 4 core Python files, organized structure
- **Space saved**: Removed ~15 redundant files and 2 large directories