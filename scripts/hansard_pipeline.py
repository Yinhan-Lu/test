import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import os
from typing import Dict, Tuple, List
from improved_nunavut_parser import ImprovedNunavutParser
from hansard_validator import HansardValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HansardPipeline:
    """
    Comprehensive pipeline for processing Nunavut and Canadian Hansard data.
    
    This pipeline:
    1. Cleans Canadian Hansard data to proper format
    2. Parses Nunavut Hansard with improved extraction
    3. Ensures temporal alignment between corpora (1999-2017)
    4. Creates three datasets as specified:
       - Dataset A: 100% Nunavut Hansard (1999-2017)
       - Dataset B: 100% Canadian Hansard (1999-2017)  
       - Dataset C: 50% Nunavut + 50% Canadian Hansard (1999-2017)
    5. Validates extraction completeness and accuracy
    """
    
    def __init__(self):
        self.target_date_range = ('1999-01-01', '2017-12-31')
        self.parser = ImprovedNunavutParser()
        self.validator = HansardValidator()
        
        # Required columns for final datasets
        self.required_columns = [
            'basepk', 'hid', 'speechdate', 'pid', 'opid', 'speakeroldname',
            'speakerposition', 'maintopic', 'subtopic', 'subsubtopic', 
            'speechtext', 'speakerparty', 'speakerriding', 'speakername', 'speakerurl'
        ]
    
    def clean_canadian_hansard(self, input_path: str) -> pd.DataFrame:
        """
        Clean and standardize Canadian Hansard data.
        
        Args:
            input_path: Path to raw Canadian Hansard CSV
            
        Returns:
            Cleaned DataFrame with standardized format
        """
        logger.info("Cleaning Canadian Hansard data...")
        

        try:
            # Load the data in chunks to handle large file
            chunk_list = []
            chunk_size = 10000
            
            for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
                chunk_list.append(chunk)
            
            df = pd.concat(chunk_list, ignore_index=True)
            logger.info(f"Loaded {len(df)} Canadian Hansard records")
        except Exception as e:
            logger.error(f"Error loading Canadian Hansard: {e}")
            return pd.DataFrame()
        
        # Clean the data
        original_count = len(df)
        
        # Remove rows with missing essential information
        # .dropna() removes rows with missing values in the specified columns
        df = df.dropna(subset=['speechtext', 'speechdate'])
        logger.info(f"Removed {original_count - len(df)} rows with missing speech/date")
        
        # Convert speech date to datetime
        df['speechdate'] = pd.to_datetime(df['speechdate'], errors='coerce')
        df = df.dropna(subset=['speechdate'])
        
        # Apply temporal filtering (1999-2017)
        start_date = pd.to_datetime(self.target_date_range[0])
        end_date = pd.to_datetime(self.target_date_range[1])
        
        original_count = len(df)
        df = df[(df['speechdate'] >= start_date) & (df['speechdate'] <= end_date)]
        logger.info(f"Filtered to target date range: {original_count - len(df)} records removed")
        
        # Clean speech text
        # .astype(str) converts the column to a string type
        df['speechtext'] = df['speechtext'].astype(str)
        df = df[df['speechtext'].str.len() > 20]  # Remove very short speeches
        
        # Standardize speaker names
        # .fillna('Unknown Speaker') fills missing values with 'Unknown Speaker'
        # .get('speakeroldname', df['speakername']) gets the value of the column 'speakeroldname' if it exists,
        #  otherwise gets the value of the column 'speakername'
        df['speakername'] = df['speakername'].fillna('Unknown Speaker')
        df['speakeroldname'] = df.get('speakeroldname', df['speakername'])
        
        # Fill missing columns with defaults
        for col in self.required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Ensure proper data types
        # range(len(df)) creates a list of numbers from 0 to the length of the dataframe
        # the reason we do this is because the basepk is a unique identifier for each row
        # and we need to ensure that it is unique for each row
        df['basepk'] = range(len(df))
        df['speechdate'] = df['speechdate'].dt.strftime('%Y-%m-%d')
        
        logger.info(f"Canadian Hansard cleaned: {len(df)} records")
        logger.info(f"Date range: {df['speechdate'].min()} to {df['speechdate'].max()}")
        
        return df[self.required_columns]
    
    def process_nunavut_hansard(self, input_path: str) -> pd.DataFrame:
        """
        Process Nunavut Hansard using improved parser.
        
        Args:
            input_path: Path to preprocessed Nunavut Hansard text
            
        Returns:
            Parsed DataFrame with standardized format
        """
        logger.info("Processing Nunavut Hansard data...")
        
        # Parse with improved parser
        df = self.parser.parse_hansard(input_path)
        
        if df.empty:
            logger.error("Failed to parse Nunavut Hansard!")
            return df
        
        # Apply temporal filtering (1999-2017)
        start_date = pd.to_datetime(self.target_date_range[0])
        end_date = pd.to_datetime(self.target_date_range[1])
        
        df['speechdate'] = pd.to_datetime(df['speechdate'])
        original_count = len(df)
        df = df[(df['speechdate'] >= start_date) & (df['speechdate'] <= end_date)]
        logger.info(f"Filtered to target date range: {original_count - len(df)} records removed")
        
        # Standardize to required format
        standardized_df = pd.DataFrame()
        standardized_df['basepk'] = range(len(df))
        standardized_df['hid'] = df['hid']
        standardized_df['speechdate'] = df['speechdate'].dt.strftime('%Y-%m-%d')
        standardized_df['pid'] = ''
        standardized_df['opid'] = ''
        standardized_df['speakeroldname'] = df['speakername']
        standardized_df['speakerposition'] = ''
        standardized_df['maintopic'] = ''
        standardized_df['subtopic'] = ''
        standardized_df['subsubtopic'] = ''
        standardized_df['speechtext'] = df['speechtext']
        standardized_df['speakerparty'] = ''
        standardized_df['speakerriding'] = ''
        standardized_df['speakername'] = df['speakername']
        standardized_df['speakerurl'] = ''
        
        logger.info(f"Nunavut Hansard processed: {len(standardized_df)} records")
        
        # Get date range from the filtered df before converting to string
        if len(df) > 0:
            logger.info(f"Date range: {df['speechdate'].min().strftime('%Y-%m-%d')} to {df['speechdate'].max().strftime('%Y-%m-%d')}")
        
        return standardized_df
    
    def create_datasets(self, canadian_df: pd.DataFrame, nunavut_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create three datasets of EQUAL SIZE by downsampling the larger corpus.
        
        This ensures that models are trained on the same amount of data,
        isolating the effect of the data source.
        
        Args:
            canadian_df: Cleaned Canadian Hansard DataFrame
            nunavut_df: Processed Nunavut Hansard DataFrame
            
        Returns:
            Dictionary containing datasets A, B, and C of equal size.
        """
        logger.info("Creating datasets with equal sizing...")

        # Determine the limiting size N by finding the smaller corpus
        n_canadian = len(canadian_df)
        n_nunavut = len(nunavut_df)
        limiting_size = min(n_canadian, n_nunavut)
        
        logger.info(f"Canadian corpus size: {n_canadian}")
        logger.info(f"Nunavut corpus size: {n_nunavut}")
        logger.info(f"Setting limiting size for all datasets to N = {limiting_size}")

        datasets = {}

        # Dataset A: 100% Nunavut Hansard (Sampled to N)
        datasets['A'] = nunavut_df.sample(n=limiting_size, random_state=42).reset_index(drop=True)
        datasets['A']['basepk'] = range(len(datasets['A']))
        logger.info(f"Dataset A (Nunavut only): {len(datasets['A'])} records")

        # Dataset B: 100% Canadian Hansard (Sampled to N)
        datasets['B'] = canadian_df.sample(n=limiting_size, random_state=42).reset_index(drop=True)
        datasets['B']['basepk'] = range(len(datasets['B']))
        logger.info(f"Dataset B (Canadian only): {len(datasets['B'])} records")

        # Dataset C: 50% Nunavut + 50% Canadian Hansard (Total size N)
        # Calculate sample sizes for 50-50 split, which is N/2 from each.
        sample_size_half = limiting_size // 2
        
        canadian_sample = canadian_df.sample(n=sample_size_half, random_state=42)
        nunavut_sample = nunavut_df.sample(n=sample_size_half, random_state=42)
        
        # Combine and shuffle
        dataset_c = pd.concat([canadian_sample, nunavut_sample], ignore_index=True)
        dataset_c = dataset_c.sample(frac=1, random_state=42).reset_index(drop=True)
        dataset_c['basepk'] = range(len(dataset_c))
        
        # Ensure final size is exactly the limiting size due to potential odd numbers
        if len(dataset_c) < limiting_size:
             # Add one more from the larger pool to compensate for integer division
            if n_canadian > n_nunavut:
                 extra_sample = canadian_df.drop(canadian_sample.index).sample(n=1, random_state=42)
            else:
                 extra_sample = nunavut_df.drop(nunavut_sample.index).sample(n=1, random_state=42)
            dataset_c = pd.concat([dataset_c, extra_sample], ignore_index=True)

        datasets['C'] = dataset_c
        logger.info(f"Dataset C (50-50 mix): {len(datasets['C'])} records")
        logger.info(f"  - Canadian records: {len(canadian_sample)}")
        logger.info(f"  - Nunavut records: {len(nunavut_sample)}")
        
        return datasets
    
    def split_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Split each dataset into train/validation/test sets.
        
        Args:
            datasets: Dictionary of datasets A, B, C
            
        Returns:
            Dictionary with train/val/test splits for each dataset
        """
        logger.info("Creating train/validation/test splits...")
        
        splits = {}
        
        for name, df in datasets.items():
            # Sort by date for consistent splitting
            df_sorted = df.sort_values('speechdate').reset_index(drop=True)
            
            # 70% train, 15% validation, 15% test
            n = len(df_sorted)
            train_end = int(0.7 * n)
            val_end = int(0.85 * n)
            
            splits[name] = {
                'train': df_sorted[:train_end].copy(),
                'val': df_sorted[train_end:val_end].copy(),
                'test': df_sorted[val_end:].copy()
            }
            
            # Update basepk for each split
            for split_name, split_df in splits[name].items():
                split_df['basepk'] = range(len(split_df))
            
            logger.info(f"Dataset {name} split: "
                       f"train={len(splits[name]['train'])}, "
                       f"val={len(splits[name]['val'])}, "
                       f"test={len(splits[name]['test'])}")
        
        return splits
    
    def save_datasets(self, splits: Dict[str, Dict[str, pd.DataFrame]], output_dir: str):
        """
        Save all datasets to CSV files.
        
        Args:
            splits: Dictionary with train/val/test splits for each dataset
            output_dir: Directory to save files
        """
        logger.info(f"Saving datasets to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, dataset_splits in splits.items():
            for split_name, df in dataset_splits.items():
                filename = f"dataset_{dataset_name.lower()}_{split_name}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {filename}: {len(df)} records")
    
    def run_validation(self, nunavut_raw_path: str, nunavut_processed_path: str) -> Dict:
        """
        Run comprehensive validation on the processing pipeline.
        
        Args:
            nunavut_raw_path: Path to raw Nunavut text
            nunavut_processed_path: Path to processed Nunavut CSV
            
        Returns:
            Validation results dictionary
        """
        logger.info("Running validation...")
        
        results = self.validator.run_comprehensive_validation(
            nunavut_raw_path, nunavut_processed_path
        )
        
        return results
    
    def generate_pipeline_report(self, 
                                datasets: Dict[str, pd.DataFrame],
                                splits: Dict[str, Dict[str, pd.DataFrame]],
                                validation_results: Dict) -> Dict:
        """
        Generate comprehensive pipeline report.
        
        Args:
            datasets: Created datasets
            splits: Train/val/test splits
            validation_results: Validation results
            
        Returns:
            Complete pipeline report
        """
        report = {
            'pipeline_timestamp': pd.Timestamp.now().isoformat(),
            'temporal_range': self.target_date_range,
            'datasets': {},
            'splits': {},
            'validation': validation_results,
            'summary': {}
        }
        
        # Dataset statistics
        for name, df in datasets.items():
            dates = pd.to_datetime(df['speechdate'])
            speakers = df['speakername'].value_counts()
            
            report['datasets'][name] = {
                'total_records': len(df),
                'unique_speakers': df['speakername'].nunique(),
                'unique_dates': df['speechdate'].nunique(),
                'date_range': [dates.min().isoformat(), dates.max().isoformat()],
                'avg_speech_length': df['speechtext'].str.len().mean(),
                'median_speech_length': df['speechtext'].str.len().median(),
                'total_words': df['speechtext'].str.split().str.len().sum(),
                'top_speakers': dict(speakers.head(10))
            }
        
        # Split statistics
        for dataset_name, dataset_splits in splits.items():
            report['splits'][dataset_name] = {}
            for split_name, split_df in dataset_splits.items():
                report['splits'][dataset_name][split_name] = {
                    'records': len(split_df),
                    'percentage': len(split_df) / len(datasets[dataset_name]) * 100
                }
        
        # Summary
        total_records = sum(len(df) for df in datasets.values())
        extraction_rate = validation_results.get('completeness', {}).get('extraction_rate', 0)
        accuracy_rate = validation_results.get('accuracy', {}).get('accuracy_rate', 0)
        
        report['summary'] = {
            'total_records_processed': total_records,
            'extraction_rate': extraction_rate,
            'accuracy_rate': accuracy_rate,
            'overall_score': validation_results.get('overall_score', 0),
            'pipeline_success': extraction_rate > 50 and accuracy_rate > 90
        }
        
        return report

def main():
    """Main pipeline execution."""
    logger.info("Starting Hansard processing pipeline...")
    
    pipeline = HansardPipeline()
    
    # --- Robust Path Definition ---
    # Construct paths relative to this script's location to ensure it can be
    # run from anywhere.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # File paths
    canadian_input = os.path.join(project_root, 'data', 'cleaned_canadian_hansard.csv')
    nunavut_input = os.path.join(project_root, 'data', 'preprocessed_nunavut_hansard.txt')
    output_dir = os.path.join(project_root, 'data')
    
    try:
        # Step 1: Clean Canadian Hansard
        logger.info("=" * 50)
        logger.info("STEP 1: Cleaning Canadian Hansard")
        logger.info("=" * 50)
        canadian_df = pipeline.clean_canadian_hansard(canadian_input)
        
        if canadian_df.empty:
            logger.error("Failed to clean Canadian Hansard!")
            return
        
        # Step 2: Process Nunavut Hansard
        logger.info("=" * 50)
        logger.info("STEP 2: Processing Nunavut Hansard")
        logger.info("=" * 50)
        nunavut_df = pipeline.process_nunavut_hansard(nunavut_input)
        
        if nunavut_df.empty:
            logger.error("Failed to process Nunavut Hansard!")
            return
        
        # Save processed Nunavut data for validation
        nunavut_processed_path = os.path.join(output_dir, 'processed_nunavut_hansard_pipeline.csv')
        nunavut_df.to_csv(nunavut_processed_path, index=False)
        
        # Step 3: Create datasets
        logger.info("=" * 50)
        logger.info("STEP 3: Creating datasets")
        logger.info("=" * 50)
        datasets = pipeline.create_datasets(canadian_df, nunavut_df)
        
        # Step 4: Create splits
        logger.info("=" * 50)
        logger.info("STEP 4: Creating train/val/test splits")
        logger.info("=" * 50)
        splits = pipeline.split_datasets(datasets)
        
        # Step 5: Save datasets
        logger.info("=" * 50)
        logger.info("STEP 5: Saving datasets")
        logger.info("=" * 50)
        pipeline.save_datasets(splits, output_dir)
        
        # Step 6: Run validation
        logger.info("=" * 50)
        logger.info("STEP 6: Running validation")
        logger.info("=" * 50)
        validation_results = pipeline.run_validation(nunavut_input, nunavut_processed_path)
        
        # Step 7: Generate report
        logger.info("=" * 50)
        logger.info("STEP 7: Generating pipeline report")
        logger.info("=" * 50)
        report = pipeline.generate_pipeline_report(datasets, splits, validation_results)
        
        # Save report
        report_path = os.path.join(output_dir, 'pipeline_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display summary
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total records processed: {report['summary']['total_records_processed']}")
        print(f"Extraction rate: {report['summary']['extraction_rate']:.1f}%")
        print(f"Accuracy rate: {report['summary']['accuracy_rate']:.1f}%")
        print(f"Overall score: {report['summary']['overall_score']:.1f}%")
        print(f"Pipeline success: {report['summary']['pipeline_success']}")
        print("\nDatasets created:")
        for name, info in report['datasets'].items():
            print(f"  Dataset {name}: {info['total_records']} records, {info['unique_speakers']} speakers")
        print(f"\nDetailed report saved to: {report_path}")
        
        logger.info("Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main() 