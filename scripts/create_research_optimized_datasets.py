#!/usr/bin/env python3
"""
Research-optimized dataset creation for cultural bias detection.

This script creates datasets optimized for bias detection research rather than 
traditional ML model optimization:

RESEARCH-OPTIMIZED APPROACH:
- 85% training data (maximize cultural signal learning)
- 15% test data (sufficient for reliable bias evaluation)
- No validation set (not needed for bias detection research)

TRADITIONAL ML APPROACH:
- 70% training / 15% validation / 15% test
- Optimized for hyperparameter tuning and overfitting prevention

Research Justification:
- More training data captures richer cultural patterns
- Validation sets don't contribute to bias measurements
- Fixed model specifications eliminate need for hyperparameter tuning
- Test sets provide sufficient data for cross-evaluation analysis

Usage:
    python create_research_optimized_datasets.py [--approach research|traditional]
    
Example:
    python create_research_optimized_datasets.py --approach research --balanced
"""

import os
import pandas as pd
import numpy as np
import logging
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchOptimizedDatasetCreator:
    """
    Creates datasets optimized for cultural bias detection research.
    """
    
    def __init__(self, approach: str = "research", target_size: int = None, random_seed: int = 42):
        self.approach = approach
        self.target_size = target_size
        self.random_seed = random_seed
        self.dataset_info = {}
        
        # Define split ratios based on approach
        if approach == "research":
            self.splits = {
                'train': 0.85,  # More data for cultural learning
                'test': 0.15    # Sufficient for bias evaluation
            }
        elif approach == "traditional":
            self.splits = {
                'train': 0.70,
                'val': 0.15,
                'test': 0.15
            }
        else:
            raise ValueError("Approach must be 'research' or 'traditional'")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        logger.info(f"Initializing {approach} dataset creator")
        logger.info(f"Split ratios: {self.splits}")
        logger.info(f"Random seed: {random_seed}")
        
    def justify_approach(self):
        """Explain the methodological justification for the chosen approach."""
        logger.info("📋 METHODOLOGICAL JUSTIFICATION")
        logger.info("="*60)
        
        if self.approach == "research":
            logger.info("🔬 RESEARCH-OPTIMIZED APPROACH (85% train / 15% test)")
            logger.info("Optimized for cultural bias detection research:")
            logger.info("  ✅ 85% training data maximizes cultural signal learning")
            logger.info("  ✅ 15% test data sufficient for reliable cross-evaluation")
            logger.info("  ✅ No validation needed (fixed model specifications)")
            logger.info("  ✅ +21% more training data captures richer cultural patterns")
            logger.info("")
            logger.info("Research Benefits:")
            logger.info("  • Better cultural representation in learned models")
            logger.info("  • More robust bias detection through richer training")
            logger.info("  • Optimized for research goals, not model performance")
            
        else:  # traditional
            logger.info("🎓 TRADITIONAL ML APPROACH (70% train / 15% val / 15% test)")
            logger.info("Standard machine learning practice:")
            logger.info("  ✅ Training set for model learning")
            logger.info("  ✅ Validation set for hyperparameter tuning")
            logger.info("  ✅ Test set for final evaluation")
            logger.info("")
            logger.info("ML Benefits:")
            logger.info("  • Prevents overfitting through validation monitoring")
            logger.info("  • Enables hyperparameter optimization")
            logger.info("  • Follows established ML best practices")
        
        logger.info("="*60)
    
    def analyze_current_datasets(self) -> Dict[str, Dict]:
        """Analyze current dataset sizes and distributions."""
        logger.info("📊 Analyzing current dataset sizes...")
        
        datasets = ['a', 'b', 'c']
        splits = ['train', 'val', 'test']
        
        for dataset in datasets:
            self.dataset_info[dataset] = {}
            total_samples = 0
            
            for split in splits:
                file_path = f"../data/dataset_{dataset}_{split}.csv"
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    valid_samples = len(df[df['speechtext'].str.len() >= 20])
                    total_samples += valid_samples
                    
                    self.dataset_info[dataset][split] = {
                        'file_path': file_path,
                        'total_samples': len(df),
                        'valid_samples': valid_samples
                    }
                    
                    logger.info(f"Dataset {dataset.upper()} {split}: {valid_samples:,} valid samples")
                else:
                    logger.warning(f"Dataset file not found: {file_path}")
                    self.dataset_info[dataset][split] = None
            
            self.dataset_info[dataset]['total_valid'] = total_samples
            logger.info(f"Dataset {dataset.upper()} total: {total_samples:,} valid samples")
        
        return self.dataset_info
    
    def determine_target_size(self) -> int:
        """Determine optimal target size for balanced datasets."""
        if self.target_size is not None:
            logger.info(f"Using specified target size: {self.target_size:,}")
            return self.target_size
        
        # Find the size of the smallest total dataset
        total_sizes = []
        for dataset in ['a', 'b', 'c']:
            if 'total_valid' in self.dataset_info[dataset]:
                size = self.dataset_info[dataset]['total_valid']
                total_sizes.append(size)
                logger.info(f"Dataset {dataset.upper()} total valid samples: {size:,}")
        
        if not total_sizes:
            raise ValueError("No valid datasets found!")
        
        target_size = min(total_sizes)
        logger.info(f"Determined target size (smallest total dataset): {target_size:,}")
        
        return target_size
    
    def create_research_optimized_splits(self, all_texts: List[str], dataset_name: str) -> Dict[str, List[str]]:
        """Create research-optimized dataset splits."""
        logger.info(f"Creating {self.approach} splits for dataset {dataset_name.upper()}")
        
        # Shuffle all texts
        texts_array = np.array(all_texts)
        indices = np.random.permutation(len(texts_array))
        shuffled_texts = texts_array[indices].tolist()
        
        splits_data = {}
        current_idx = 0
        
        for split_name, ratio in self.splits.items():
            split_size = int(len(shuffled_texts) * ratio)
            
            # Handle remainder for last split
            if split_name == list(self.splits.keys())[-1]:
                split_texts = shuffled_texts[current_idx:]
            else:
                split_texts = shuffled_texts[current_idx:current_idx + split_size]
                current_idx += split_size
            
            splits_data[split_name] = split_texts
            logger.info(f"  {split_name}: {len(split_texts):,} samples ({ratio*100:.0f}%)")
        
        return splits_data
    
    def combine_original_datasets(self, dataset_name: str) -> List[str]:
        """Combine all original splits into one dataset for re-splitting."""
        logger.info(f"Combining original splits for dataset {dataset_name.upper()}")
        
        all_texts = []
        
        for split in ['train', 'val', 'test']:
            if self.dataset_info[dataset_name][split] is not None:
                file_path = self.dataset_info[dataset_name][split]['file_path']
                df = pd.read_csv(file_path)
                
                # Filter valid speeches
                valid_df = df[df['speechtext'].str.len() >= 20]
                texts = valid_df['speechtext'].astype(str).tolist()
                all_texts.extend(texts)
                
                logger.info(f"  Added {len(texts):,} texts from {split} split")
        
        logger.info(f"  Total combined: {len(all_texts):,} texts")
        return all_texts
    
    def create_balanced_research_datasets(self, balanced: bool = True) -> Dict[str, Dict]:
        """Create research-optimized balanced or original datasets."""
        logger.info("🔬 Creating research-optimized datasets...")
        
        # Justify approach
        self.justify_approach()
        
        # Analyze current datasets
        self.analyze_current_datasets()
        
        # Determine target size if balancing
        if balanced:
            target_size = self.determine_target_size()
            logger.info(f"📊 Creating balanced datasets with {target_size:,} samples each")
        else:
            logger.info("📊 Creating research-optimized splits from original datasets")
        
        result_info = {}
        datasets = ['a', 'b', 'c']
        
        for dataset in datasets:
            logger.info(f"\n🔄 Processing Dataset {dataset.upper()}...")
            result_info[dataset] = {}
            
            # Get all texts from original dataset
            all_texts = self.combine_original_datasets(dataset)
            
            # Sample if balancing
            if balanced and len(all_texts) > target_size:
                # Random sample
                indices = np.random.choice(len(all_texts), target_size, replace=False)
                sampled_texts = [all_texts[i] for i in indices]
                logger.info(f"  Sampled {len(sampled_texts):,} from {len(all_texts):,} total texts")
                all_texts = sampled_texts
            
            # Create splits
            splits_data = self.create_research_optimized_splits(all_texts, dataset)
            
            # Save splits
            for split_name, texts in splits_data.items():
                # Create DataFrame with required columns
                df_data = {
                    'speechtext': texts,
                    'speechdate': ['1999-01-01'] * len(texts),  # Placeholder
                    'speakername': ['Unknown'] * len(texts),    # Placeholder
                    'basepk': list(range(len(texts))),
                    'hid': list(range(len(texts))),
                    'pid': list(range(len(texts))),
                    'opid': list(range(len(texts))),
                    'speakeroldname': ['Unknown'] * len(texts),
                    'speakerposition': ['Unknown'] * len(texts),
                    'maintopic': ['Unknown'] * len(texts),
                    'subtopic': ['Unknown'] * len(texts),
                    'subsubtopic': ['Unknown'] * len(texts),
                    'speakerparty': ['Unknown'] * len(texts),
                    'speakerriding': ['Unknown'] * len(texts),
                    'speakerurl': ['Unknown'] * len(texts)
                }
                
                df = pd.DataFrame(df_data)
                
                # Determine output filename
                if balanced:
                    if self.approach == "research":
                        output_file = f"../data/dataset_{dataset}_balanced_research_{split_name}.csv"
                    else:
                        output_file = f"../data/dataset_{dataset}_balanced_{split_name}.csv"
                else:
                    if self.approach == "research":
                        output_file = f"../data/dataset_{dataset}_research_{split_name}.csv"
                    else:
                        output_file = f"../data/dataset_{dataset}_{split_name}.csv"
                
                df.to_csv(output_file, index=False)
                
                result_info[dataset][split_name] = {
                    'samples': len(texts),
                    'output_file': output_file
                }
                
                logger.info(f"  💾 Saved {split_name}: {len(texts):,} samples → {output_file}")
        
        return result_info
    
    def generate_research_report(self, result_info: Dict, balanced: bool) -> Dict:
        """Generate comprehensive report on research-optimized datasets."""
        logger.info("📄 Generating research optimization report...")
        
        report = {
            'created_timestamp': datetime.now().isoformat(),
            'approach': self.approach,
            'balanced': balanced,
            'random_seed': self.random_seed,
            'split_ratios': self.splits,
            'methodology': {
                'optimization_goal': 'Cultural bias detection research',
                'justification': 'Maximize cultural signal while maintaining evaluation validity',
                'benefits': self._get_approach_benefits()
            },
            'datasets_created': result_info
        }
        
        # Calculate summary statistics
        total_samples = {}
        for split in self.splits.keys():
            total_samples[split] = sum(
                result_info[dataset][split]['samples'] 
                for dataset in ['a', 'b', 'c']
                if split in result_info[dataset]
            )
        
        report['summary'] = {
            'total_samples_per_split': total_samples,
            'datasets_created': len(result_info),
            'approach_benefits': self._get_approach_benefits()
        }
        
        # Save report
        os.makedirs("../results", exist_ok=True)
        report_path = f"../results/research_optimized_datasets_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Research optimization report saved to: {report_path}")
        return report
    
    def _get_approach_benefits(self) -> List[str]:
        """Get benefits of the chosen approach."""
        if self.approach == "research":
            return [
                "21% more training data for richer cultural learning",
                "Optimized for bias detection rather than model performance",
                "Eliminates unnecessary validation overhead",
                "Maximizes cultural signal capture",
                "Test set sufficient for reliable cross-evaluation"
            ]
        else:
            return [
                "Standard ML practice for model optimization", 
                "Validation enables hyperparameter tuning",
                "Prevents overfitting through monitoring",
                "Widely accepted academic methodology",
                "Familiar to ML research community"
            ]

def main():
    """Main function for creating research-optimized datasets."""
    parser = argparse.ArgumentParser(description="Create research-optimized datasets for bias detection")
    parser.add_argument('--approach', choices=['research', 'traditional'], default='research',
                       help='Dataset optimization approach (default: research)')
    parser.add_argument('--balanced', action='store_true',
                       help='Create balanced datasets with equal sample sizes')
    parser.add_argument('--target-size', type=int, default=None,
                       help='Target sample size for balanced datasets')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    logger.info("🔬 Research-Optimized Dataset Creation")
    logger.info("="*60)
    
    # Create dataset creator
    creator = ResearchOptimizedDatasetCreator(
        approach=args.approach,
        target_size=args.target_size,
        random_seed=args.seed
    )
    
    # Create datasets
    result_info = creator.create_balanced_research_datasets(balanced=args.balanced)
    
    # Generate report
    report = creator.generate_research_report(result_info, args.balanced)
    
    # Print summary
    print(f"\n{'='*70}")
    print("RESEARCH-OPTIMIZED DATASET CREATION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nApproach: {args.approach.upper()}")
    print(f"Balanced: {'Yes' if args.balanced else 'No'}")
    print(f"Split ratios: {creator.splits}")
    
    if args.approach == "research":
        print(f"\n🔬 Research Optimization Benefits:")
        for benefit in report['methodology']['benefits']:
            print(f"  • {benefit}")
    
    print(f"\nDatasets Created:")
    for dataset in ['a', 'b', 'c']:
        print(f"  Dataset {dataset.upper()}:")
        for split, info in result_info[dataset].items():
            print(f"    {split}: {info['samples']:,} samples")
    
    print(f"\n✅ Research-optimized datasets created successfully!")
    
    if args.approach == "research":
        print(f"\n🎯 Next Steps for Bias Detection Research:")
        print(f"  1. Train models with research-optimized datasets")
        print(f"  2. Use cross-evaluation for bias detection")
        print(f"  3. Compare with traditional approach for robustness")
    
    print(f"\nReport saved to: ../results/research_optimized_datasets_report_*.json")

if __name__ == "__main__":
    main()