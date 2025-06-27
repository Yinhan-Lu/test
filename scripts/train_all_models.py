#!/usr/bin/env python3
"""
Complete BERT model training pipeline for cultural bias analysis.

This script orchestrates the training of all three models sequentially:
- Model A: Trained on 100% Nunavut Hansard data
- Model B: Trained on 100% Canadian Hansard data  
- Model C: Trained on 50% Nunavut + 50% Canadian Hansard data

Usage:
    python train_all_models.py [--demo] [--resume-from CHECKPOINT]
    
Options:
    --demo: Train demo models (faster, smaller)
    --resume-from: Resume training from checkpoint
    --evaluate: Run evaluation after training
    
Example:
    python train_all_models.py                    # Train all production models
    python train_all_models.py --demo             # Train demo models only
    python train_all_models.py --evaluate         # Train and evaluate
"""

import os
import sys
import argparse
import logging
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainingOrchestrator:
    """Orchestrates the complete model training pipeline."""
    
    def __init__(self, demo_mode: bool = False, use_balanced: bool = False):
        self.demo_mode = demo_mode
        self.use_balanced = use_balanced
        self.trainer_script = "bert_trainer_demo.py" if demo_mode else "bert_trainer_production.py"
        self.results = {}
        
        logger.info(f"Initializing training orchestrator (demo_mode={demo_mode}, balanced={use_balanced})")
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are available."""
        logger.info("Checking prerequisites...")
        
        # Check if tokenizer exists
        tokenizer_path = "../tokenizer/hansard-bpe-tokenizer"
        if not os.path.exists(f"{tokenizer_path}/vocab.json"):
            logger.error(f"Tokenizer not found at {tokenizer_path}")
            logger.error("Please train the BPE tokenizer first")
            return False
        
        # Check if datasets exist
        datasets = ['a', 'b', 'c']
        splits = ['train', 'val', 'test']
        
        for dataset in datasets:
            for split in splits:
                if self.use_balanced:
                    dataset_file = f"../data/dataset_{dataset}_balanced_{split}.csv"
                else:
                    dataset_file = f"../data/dataset_{dataset}_{split}.csv"
                    
                if not os.path.exists(dataset_file):
                    if self.use_balanced:
                        logger.error(f"Balanced dataset file not found: {dataset_file}")
                        logger.error("Please create balanced datasets first:")
                        logger.error("  python create_balanced_datasets.py")
                    else:
                        logger.error(f"Dataset file not found: {dataset_file}")
                    return False
        
        # Check if trainer script exists
        if not os.path.exists(self.trainer_script):
            logger.error(f"Trainer script not found: {self.trainer_script}")
            return False
        
        logger.info("✅ All prerequisites satisfied")
        return True
    
    def estimate_training_time(self) -> Dict[str, str]:
        """Estimate training times for each model."""
        if self.demo_mode:
            return {
                'model_a': "~30 seconds",
                'model_b': "~30 seconds", 
                'model_c': "~30 seconds",
                'total': "~2 minutes"
            }
        else:
            # Load dataset sizes to estimate time
            dataset_sizes = {}
            
            try:
                with open("../data/DATASET_SUMMARY.md", 'r') as f:
                    content = f.read()
                    
                # Extract training sample counts
                lines = content.split('\n')
                for line in lines:
                    if "TRAIN:" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            size = int(parts[1].replace(',', ''))
                            if 'DATASET A' in content[content.find(line)-200:content.find(line)]:
                                dataset_sizes['a'] = size
                            elif 'DATASET B' in content[content.find(line)-200:content.find(line)]:
                                dataset_sizes['b'] = size
                            elif 'DATASET C' in content[content.find(line)-200:content.find(line)]:
                                dataset_sizes['c'] = size
            except:
                # Fallback estimates
                dataset_sizes = {'a': 122000, 'b': 375000, 'c': 126000}
            
            # Estimate based on ~1000 samples per minute on modern GPU
            estimates = {}
            total_hours = 0
            
            for model, size in dataset_sizes.items():
                hours = size / 60000  # Very rough estimate
                estimates[f'model_{model}'] = f"~{hours:.1f}-{hours*2:.1f} hours"
                total_hours += hours
                
            estimates['total'] = f"~{total_hours:.1f}-{total_hours*2:.1f} hours"
            
            return estimates
    
    def train_model(self, dataset_name: str, resume_from: Optional[str] = None) -> bool:
        """Train a single model."""
        logger.info(f"Starting training for Model {dataset_name.upper()}")
        
        # Prepare command
        cmd = ["python", self.trainer_script, dataset_name]
        
        # Add balanced flag if using balanced datasets
        if self.use_balanced and not self.demo_mode:
            cmd.append("--balanced")
        
        try:
            # Run training
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Model {dataset_name.upper()} training completed successfully")
                
                # Parse training results if available
                model_suffix = ""
                if self.demo_mode:
                    model_suffix = "demo_"
                elif self.use_balanced:
                    model_suffix = ""  # Production trainer handles the balanced suffix
                
                model_name = f"{'demo_' if self.demo_mode else ''}model_{dataset_name}{'_balanced' if self.use_balanced and not self.demo_mode else ''}"
                model_dir = f"../models/{model_name}"
                summary_file = f"{model_dir}/training_summary.json"
                
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                        self.results[f"model_{dataset_name}"] = summary
                        logger.info(f"  Training time: {summary.get('training_time', 'Unknown')}")
                        logger.info(f"  Final loss: {summary.get('final_loss', 'Unknown')}")
                        logger.info(f"  Final perplexity: {summary.get('final_perplexity', 'Unknown')}")
                
                return True
            else:
                logger.error(f"❌ Model {dataset_name.upper()} training failed")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error training Model {dataset_name.upper()}: {e}")
            return False
    
    def train_all_models(self, resume_from: Optional[str] = None) -> bool:
        """Train all three models sequentially."""
        logger.info("Starting complete model training pipeline")
        
        # Print training estimates
        estimates = self.estimate_training_time()
        logger.info("Training time estimates:")
        for model, time in estimates.items():
            logger.info(f"  {model}: {time}")
        
        models = ['a', 'b', 'c']
        successful_models = []
        
        for model_name in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING MODEL {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            success = self.train_model(model_name, resume_from)
            
            if success:
                successful_models.append(model_name)
                logger.info(f"✅ Model {model_name.upper()} completed")
            else:
                logger.error(f"❌ Model {model_name.upper()} failed")
                
                # Ask user if they want to continue
                response = input(f"Model {model_name.upper()} training failed. Continue with remaining models? (y/n): ")
                if response.lower() != 'y':
                    break
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING PIPELINE SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully trained: {len(successful_models)}/3 models")
        logger.info(f"Successful models: {', '.join([m.upper() for m in successful_models])}")
        
        if len(successful_models) == 3:
            logger.info("🎉 All models trained successfully!")
            return True
        else:
            logger.warning(f"⚠️  Only {len(successful_models)}/3 models completed successfully")
            return False
    
    def run_evaluation(self) -> bool:
        """Run evaluation on trained models."""
        logger.info("Running evaluation on trained models...")
        
        evaluator_script = "model_evaluator_production.py"
        
        if not os.path.exists(evaluator_script):
            logger.error(f"Evaluator script not found: {evaluator_script}")
            return False
        
        try:
            cmd = ["python", evaluator_script]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode == 0:
                logger.info("✅ Evaluation completed successfully")
                print(result.stdout)  # Show evaluation results
                return True
            else:
                logger.error("❌ Evaluation failed")
                logger.error(f"stderr: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running evaluation: {e}")
            return False
    
    def generate_final_report(self):
        """Generate a final training report."""
        logger.info("Generating final training report...")
        
        report = {
            'training_completed': datetime.now().isoformat(),
            'demo_mode': self.demo_mode,
            'trainer_script': self.trainer_script,
            'models_trained': list(self.results.keys()),
            'training_results': self.results
        }
        
        # Save report
        os.makedirs("../results", exist_ok=True)
        report_file = f"../results/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to: {report_file}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("FINAL TRAINING REPORT")
        print(f"{'='*80}")
        print(f"Training mode: {'Demo' if self.demo_mode else 'Production'}")
        print(f"Models trained: {len(self.results)}/3")
        print(f"Report saved to: {report_file}")
        
        if self.results:
            print(f"\nTraining Summary:")
            for model_name, summary in self.results.items():
                print(f"  {model_name}:")
                print(f"    Training time: {summary.get('training_time', 'Unknown')}")
                print(f"    Final loss: {summary.get('final_loss', 'Unknown')}")
                print(f"    Parameters: {summary.get('parameters', 'Unknown'):,}")

def main():
    """Main training orchestration function."""
    parser = argparse.ArgumentParser(description="Complete BERT model training pipeline")
    parser.add_argument('--demo', action='store_true', help='Train demo models (faster)')
    parser.add_argument('--balanced', action='store_true', help='Use balanced datasets for fair bias comparison')
    parser.add_argument('--resume-from', type=str, help='Resume training from checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = ModelTrainingOrchestrator(demo_mode=args.demo, use_balanced=args.balanced)
    
    # Check prerequisites
    if not orchestrator.check_prerequisites():
        logger.error("Prerequisites not satisfied. Exiting.")
        sys.exit(1)
    
    print(f"\n🚀 BERT Cultural Bias Analysis Training Pipeline")
    print(f"{'='*60}")
    print(f"Mode: {'Demo' if args.demo else 'Production'}")
    print(f"Dataset type: {'Balanced' if args.balanced else 'Original'}")
    print(f"Trainer: {orchestrator.trainer_script}")
    
    if args.balanced:
        print(f"\n📊 Using balanced datasets for rigorous bias detection:")
        print(f"  • Equal sample sizes eliminate data quantity confounds")
        print(f"  • Fair comparison isolates cultural bias effects")
        print(f"  • Methodologically sound experimental design")
    
    if not args.demo:
        time_estimate = "1-4 hours" if args.balanced else "2-12 hours"
        print(f"\n⚠️  Production training will take {time_estimate}!")
        print(f"Make sure you have sufficient GPU resources.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            sys.exit(0)
    
    # Train all models
    success = orchestrator.train_all_models(resume_from=args.resume_from)
    
    # Run evaluation if requested
    if args.evaluate and success:
        orchestrator.run_evaluation()
    
    # Generate final report
    orchestrator.generate_final_report()
    
    if success:
        print("\n🎉 Training pipeline completed successfully!")
        if not args.evaluate:
            print("\nTo run evaluation:")
            print("  python model_evaluator_production.py")
    else:
        print("\n⚠️  Training pipeline completed with some failures.")
        print("Check logs for details.")

if __name__ == "__main__":
    main()