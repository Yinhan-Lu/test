#!/usr/bin/env python3
"""
Create balanced datasets with equal sample sizes for rigorous bias detection.

This script ensures fair comparison between models by sampling equal amounts
of training data from each dataset, eliminating data quantity confounds in
bias measurements.

Research Justification:
- Equal sample sizes isolate cultural bias effects from data quantity effects
- Prevents confounding between model performance and training data amount
- Enables valid statistical comparison of bias measurements
- Follows best practices for controlled experimental design

Usage:
    python create_balanced_datasets.py [--target-size SIZE] [--seed SEED]
    
Example:
    python create_balanced_datasets.py --target-size 121886 --seed 42
"""

import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BalancedDatasetCreator:
    """
    Creates balanced datasets with equal sample sizes for fair bias comparison.
    """
    
    def __init__(self, target_size: int = None, random_seed: int = 42):
        self.target_size = target_size
        self.random_seed = random_seed
        self.dataset_info = {}
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        logger.info(f"Initializing balanced dataset creator")
        logger.info(f"Random seed: {random_seed}")
        
    def analyze_current_datasets(self) -> Dict[str, Dict]:
        """Analyze current dataset sizes and distributions."""
        logger.info("Analyzing current dataset sizes...")
        
        datasets = ['a', 'b', 'c']
        splits = ['train', 'val', 'test']
        
        for dataset in datasets:
            self.dataset_info[dataset] = {}
            
            for split in splits:
                file_path = f"../data/dataset_{dataset}_{split}.csv"
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    # Basic statistics
                    total_samples = len(df)
                    valid_samples = len(df[df['speechtext'].str.len() >= 20])
                    
                    # Date range
                    if 'speechdate' in df.columns:
                        df['speechdate'] = pd.to_datetime(df['speechdate'])
                        date_range = (df['speechdate'].min(), df['speechdate'].max())
                    else:
                        date_range = (None, None)
                    
                    # Speech length statistics
                    speech_lengths = df['speechtext'].str.len()
                    
                    self.dataset_info[dataset][split] = {
                        'file_path': file_path,
                        'total_samples': total_samples,
                        'valid_samples': valid_samples,
                        'date_range': date_range,
                        'mean_speech_length': speech_lengths.mean(),
                        'median_speech_length': speech_lengths.median(),
                        'std_speech_length': speech_lengths.std()
                    }
                    
                    logger.info(f"Dataset {dataset.upper()} {split}: {total_samples:,} samples")
                else:
                    logger.warning(f"Dataset file not found: {file_path}")
                    self.dataset_info[dataset][split] = None
        
        return self.dataset_info
    
    def determine_target_size(self) -> int:
        """Determine optimal target size for balanced datasets."""
        if self.target_size is not None:
            logger.info(f"Using specified target size: {self.target_size:,}")
            return self.target_size
        
        # Find the size of the smallest training dataset
        train_sizes = []
        for dataset in ['a', 'b', 'c']:
            if self.dataset_info[dataset]['train'] is not None:
                size = self.dataset_info[dataset]['train']['valid_samples']
                train_sizes.append(size)
                logger.info(f"Dataset {dataset.upper()} valid training samples: {size:,}")
        
        if not train_sizes:
            raise ValueError("No valid training datasets found!")
        
        target_size = min(train_sizes)
        logger.info(f"Determined target size (smallest dataset): {target_size:,}")
        
        return target_size
    
    def create_balanced_sample(self, 
                             df: pd.DataFrame, 
                             target_size: int, 
                             split_name: str,
                             dataset_name: str) -> pd.DataFrame:
        """Create a balanced sample maintaining representativeness."""
        
        # Filter valid speeches
        valid_df = df[df['speechtext'].str.len() >= 20].copy()
        
        if len(valid_df) < target_size:
            logger.warning(f"Dataset {dataset_name} {split_name} has only {len(valid_df)} valid samples, "
                          f"less than target {target_size}")
            return valid_df
        
        # For training sets, ensure temporal representativeness
        if split_name == 'train' and 'speechdate' in valid_df.columns:
            # Convert to datetime if not already
            valid_df['speechdate'] = pd.to_datetime(valid_df['speechdate'])
            
            # Sort by date and take stratified sample across time periods
            valid_df = valid_df.sort_values('speechdate')
            
            # Divide into time bins and sample proportionally
            n_bins = min(10, len(valid_df) // 100)  # 10 time bins or fewer
            
            if n_bins > 1:
                valid_df['time_bin'] = pd.cut(range(len(valid_df)), bins=n_bins, labels=False)
                
                # Sample proportionally from each time bin
                samples_per_bin = target_size // n_bins
                remainder = target_size % n_bins
                
                sampled_dfs = []
                for bin_id in range(n_bins):
                    bin_df = valid_df[valid_df['time_bin'] == bin_id]
                    
                    # Add one extra sample to first 'remainder' bins
                    bin_sample_size = samples_per_bin + (1 if bin_id < remainder else 0)
                    bin_sample_size = min(bin_sample_size, len(bin_df))
                    
                    if len(bin_df) > 0:
                        bin_sample = bin_df.sample(n=bin_sample_size, random_state=self.random_seed + bin_id)
                        sampled_dfs.append(bin_sample)
                
                balanced_df = pd.concat(sampled_dfs, ignore_index=True)
                balanced_df = balanced_df.drop('time_bin', axis=1)
                
                logger.info(f"Created stratified temporal sample for {dataset_name} {split_name}: "
                           f"{len(balanced_df):,} samples across {n_bins} time periods")
            else:
                # Simple random sample if too few samples for stratification
                balanced_df = valid_df.sample(n=target_size, random_state=self.random_seed)
                logger.info(f"Created random sample for {dataset_name} {split_name}: {len(balanced_df):,} samples")
        else:
            # Simple random sample for validation/test sets
            balanced_df = valid_df.sample(n=target_size, random_state=self.random_seed)
            logger.info(f"Created random sample for {dataset_name} {split_name}: {len(balanced_df):,} samples")
        
        return balanced_df.reset_index(drop=True)
    
    def create_all_balanced_datasets(self) -> Dict[str, Dict]:
        """Create balanced versions of all datasets."""
        logger.info("Creating balanced datasets...")
        
        # Analyze current datasets
        self.analyze_current_datasets()
        
        # Determine target size
        target_train_size = self.determine_target_size()
        
        # Calculate proportional sizes for val/test splits
        # Maintain 70/15/15 split ratio
        target_val_size = int(target_train_size * 0.15 / 0.70)
        target_test_size = target_val_size  # Same size as validation
        
        target_sizes = {
            'train': target_train_size,
            'val': target_val_size,
            'test': target_test_size
        }
        
        logger.info(f"Target sizes: train={target_train_size:,}, val={target_val_size:,}, test={target_test_size:,}")
        
        balanced_info = {}
        datasets = ['a', 'b', 'c']
        
        for dataset in datasets:
            logger.info(f"\nProcessing Dataset {dataset.upper()}...")
            balanced_info[dataset] = {}
            
            for split in ['train', 'val', 'test']:
                if self.dataset_info[dataset][split] is None:
                    logger.warning(f"Skipping {dataset} {split} - file not found")
                    continue
                
                # Load original dataset
                file_path = self.dataset_info[dataset][split]['file_path']
                df = pd.read_csv(file_path)
                
                # Create balanced sample
                target_size = target_sizes[split]
                balanced_df = self.create_balanced_sample(df, target_size, split, dataset)
                
                # Save balanced dataset
                output_path = f"../data/dataset_{dataset}_balanced_{split}.csv"
                balanced_df.to_csv(output_path, index=False)
                
                # Store information
                balanced_info[dataset][split] = {
                    'original_size': len(df),
                    'balanced_size': len(balanced_df),
                    'target_size': target_size,
                    'output_path': output_path,
                    'sampling_ratio': len(balanced_df) / len(df)
                }
                
                logger.info(f"  {split}: {len(df):,} → {len(balanced_df):,} samples "
                           f"(ratio: {balanced_info[dataset][split]['sampling_ratio']:.3f})")
        
        return balanced_info
    
    def validate_balanced_datasets(self, balanced_info: Dict) -> bool:
        """Validate that balanced datasets maintain quality and representativeness."""
        logger.info("Validating balanced datasets...")
        
        validation_passed = True
        
        for dataset in ['a', 'b', 'c']:
            for split in ['train', 'val', 'test']:
                if split not in balanced_info[dataset]:
                    continue
                    
                info = balanced_info[dataset][split]
                output_path = info['output_path']
                
                if not os.path.exists(output_path):
                    logger.error(f"Balanced dataset file not created: {output_path}")
                    validation_passed = False
                    continue
                
                # Load and validate
                df = pd.read_csv(output_path)
                
                # Check size
                expected_size = info['target_size']
                actual_size = len(df)
                
                if actual_size != expected_size:
                    # Allow for slight differences due to insufficient source data
                    size_diff = abs(actual_size - expected_size)
                    if size_diff > expected_size * 0.05:  # More than 5% difference
                        logger.error(f"Size mismatch for {dataset} {split}: "
                                   f"expected {expected_size}, got {actual_size}")
                        validation_passed = False
                    else:
                        logger.warning(f"Minor size difference for {dataset} {split}: "
                                     f"expected {expected_size}, got {actual_size}")
                
                # Check data quality
                invalid_speeches = df[df['speechtext'].str.len() < 20]
                if len(invalid_speeches) > 0:
                    logger.error(f"Found {len(invalid_speeches)} invalid speeches in {dataset} {split}")
                    validation_passed = False
                
                # Check for required columns
                required_columns = ['speechtext', 'speechdate', 'speakername']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.error(f"Missing columns in {dataset} {split}: {missing_columns}")
                    validation_passed = False
                
                logger.info(f"✅ {dataset} {split}: {actual_size:,} samples validated")
        
        return validation_passed
    
    def generate_balancing_report(self, balanced_info: Dict) -> Dict:
        """Generate comprehensive report on dataset balancing."""
        logger.info("Generating balancing report...")
        
        report = {
            'created_timestamp': datetime.now().isoformat(),
            'methodology': {
                'target_size_determination': 'Size of smallest training dataset',
                'sampling_method': 'Stratified temporal sampling for training, random for val/test',
                'random_seed': self.random_seed,
                'quality_filter': 'Speeches with length >= 20 characters'
            },
            'original_sizes': {},
            'balanced_sizes': {},
            'sampling_ratios': {},
            'validation_passed': False
        }
        
        # Collect statistics
        for dataset in ['a', 'b', 'c']:
            if dataset not in balanced_info:
                continue
                
            report['original_sizes'][dataset] = {}
            report['balanced_sizes'][dataset] = {}
            report['sampling_ratios'][dataset] = {}
            
            for split in ['train', 'val', 'test']:
                if split in balanced_info[dataset]:
                    info = balanced_info[dataset][split]
                    report['original_sizes'][dataset][split] = info['original_size']
                    report['balanced_sizes'][dataset][split] = info['balanced_size']
                    report['sampling_ratios'][dataset][split] = info['sampling_ratio']
        
        # Validate datasets
        report['validation_passed'] = self.validate_balanced_datasets(balanced_info)
        
        # Calculate summary statistics
        train_sizes = [report['balanced_sizes'][d]['train'] for d in ['a', 'b', 'c'] 
                      if 'train' in report['balanced_sizes'][d]]
        
        report['summary'] = {
            'total_datasets_created': len(balanced_info),
            'training_sample_sizes': train_sizes,
            'training_size_equality': len(set(train_sizes)) == 1,  # All same size?
            'total_balanced_samples': sum(
                sum(split_sizes.values()) for split_sizes in report['balanced_sizes'].values()
            )
        }
        
        # Save report
        os.makedirs("../results", exist_ok=True)
        report_path = f"../results/balanced_datasets_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Balancing report saved to: {report_path}")
        
        return report

def main():
    """Main function for creating balanced datasets."""
    parser = argparse.ArgumentParser(description="Create balanced datasets for fair bias comparison")
    parser.add_argument('--target-size', type=int, default=None,
                       help='Target sample size (default: size of smallest dataset)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    logger.info("🎯 Creating Balanced Datasets for Rigorous Bias Detection")
    logger.info("="*70)
    
    # Create balanced dataset creator
    creator = BalancedDatasetCreator(target_size=args.target_size, random_seed=args.seed)
    
    # Create balanced datasets
    balanced_info = creator.create_all_balanced_datasets()
    
    # Generate and validate report
    report = creator.generate_balancing_report(balanced_info)
    
    # Print summary
    print(f"\n{'='*70}")
    print("BALANCED DATASETS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nMethodological Justification:")
    print(f"  • Equal sample sizes eliminate data quantity confounds")
    print(f"  • Isolate cultural bias effects from training data amount")
    print(f"  • Enable valid statistical comparison of model performance")
    print(f"  • Follow best practices for controlled experimental design")
    
    print(f"\nDataset Sizes After Balancing:")
    for dataset in ['a', 'b', 'c']:
        if dataset in report['balanced_sizes']:
            sizes = report['balanced_sizes'][dataset]
            print(f"  Dataset {dataset.upper()}: train={sizes.get('train', 0):,}, "
                  f"val={sizes.get('val', 0):,}, test={sizes.get('test', 0):,}")
    
    print(f"\nSampling Ratios (balanced/original):")
    for dataset in ['a', 'b', 'c']:
        if dataset in report['sampling_ratios'] and 'train' in report['sampling_ratios'][dataset]:
            ratio = report['sampling_ratios'][dataset]['train']
            print(f"  Dataset {dataset.upper()}: {ratio:.3f} ({ratio*100:.1f}% of original)")
    
    print(f"\nValidation: {'✅ PASSED' if report['validation_passed'] else '❌ FAILED'}")
    print(f"Training Size Equality: {'✅ YES' if report['summary']['training_size_equality'] else '❌ NO'}")
    
    if report['validation_passed'] and report['summary']['training_size_equality']:
        print(f"\n🎉 Balanced datasets created successfully!")
        print(f"   Ready for rigorous bias detection training.")
        print(f"\nNext steps:")
        print(f"  1. Train models with balanced datasets:")
        print(f"     python bert_trainer_production.py a --balanced")
        print(f"  2. Compare results with unbalanced training")
        print(f"  3. Focus analysis on balanced results for main conclusions")
    else:
        print(f"\n⚠️  Issues detected. Check logs and fix before proceeding.")

if __name__ == "__main__":
    main()