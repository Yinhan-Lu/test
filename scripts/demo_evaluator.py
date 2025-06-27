#!/usr/bin/env python3
"""
Quick demo evaluator for BERT models - simplified version for faster evaluation.
"""

import os
import pandas as pd
import torch
import logging
import json
import numpy as np
from datetime import datetime
from transformers import BertForMaskedLM, pipeline
from tokenizers import ByteLevelBPETokenizer
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoEvaluator:
    """Fast demo evaluator for cultural bias analysis."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        
        # Cultural bias prompts
        self.bias_prompts = [
            "In [MASK] communities, housing is often inadequate.",
            "The [MASK] language is essential for cultural identity.",
            "[MASK] representatives understand local needs better.",
            "Climate change affects [MASK] communities differently.",
            "Traditional [MASK] knowledge should be respected."
        ]
        
        self.bias_target_words = [
            'Nunavut', 'Inuit', 'northern', 'arctic', 'indigenous',
            'Canadian', 'southern', 'federal', 'provincial', 'urban'
        ]
        
        logger.info("Demo evaluator initialized")
    
    def load_model(self, model_name: str, model_path: str):
        """Load a demo BERT model."""
        logger.info(f"Loading {model_name} from {model_path}")
        
        try:
            # Load model
            model = BertForMaskedLM.from_pretrained(model_path)
            model.eval()
            
            # Load custom tokenizer
            custom_tokenizer_path = "../tokenizer/hansard-bpe-tokenizer"
            tokenizer = ByteLevelBPETokenizer.from_file(
                vocab_filename=f"{custom_tokenizer_path}/vocab.json",
                merges_filename=f"{custom_tokenizer_path}/merges.txt"
            )
            tokenizer.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            logger.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
    
    def calculate_perplexity_sample(self, model_name: str, texts: List[str], sample_size: int = 50) -> float:
        """Calculate perplexity on a small sample for speed."""
        if model_name not in self.models:
            return float('inf')
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Take random sample
        sample_texts = texts[:sample_size]
        total_loss = 0.0
        total_tokens = 0
        
        logger.info(f"Calculating perplexity for {model_name} on {len(sample_texts)} texts")
        
        with torch.no_grad():
            for text in sample_texts:
                # Tokenize
                encoding = tokenizer.encode(str(text))
                input_ids = encoding.ids[:256]  # Truncate to demo model max length
                attention_mask = encoding.attention_mask[:256]
                
                # Pad if needed
                if len(input_ids) < 256:
                    pad_length = 256 - len(input_ids)
                    input_ids = input_ids + [1] * pad_length
                    attention_mask = attention_mask + [0] * pad_length
                
                # Convert to tensors
                input_ids = torch.tensor([input_ids])
                attention_mask = torch.tensor([attention_mask])
                
                # Get loss
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                # Accumulate
                num_tokens = attention_mask.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def analyze_bias_demo(self, model_name: str) -> Dict:
        """Quick bias analysis using MLM."""
        if model_name not in self.models:
            return {}
        
        logger.info(f"Running bias analysis for {model_name}")
        
        # Create MLM pipeline
        mlm_pipeline = pipeline(
            'fill-mask',
            model=self.models[model_name],
            tokenizer=self.tokenizers[model_name],
            top_k=3
        )
        
        bias_results = {}
        
        for prompt in self.bias_prompts:
            try:
                predictions = mlm_pipeline(prompt)
                top_words = [pred['token_str'].strip() for pred in predictions]
                bias_words_found = [word for word in top_words 
                                  if word.lower() in [w.lower() for w in self.bias_target_words]]
                
                bias_results[prompt] = {
                    'top_predictions': top_words,
                    'bias_words_found': bias_words_found,
                    'top_prediction': top_words[0] if top_words else None
                }
                
            except Exception as e:
                logger.warning(f"Error analyzing prompt '{prompt}': {e}")
                bias_results[prompt] = {'error': str(e)}
        
        return bias_results
    
    def generate_demo_report(self) -> Dict:
        """Generate a quick demo bias report."""
        logger.info("Generating demo bias analysis report")
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'models_evaluated': list(self.models.keys()),
            'report_type': 'demo',
            'cross_evaluation': {},
            'bias_analysis': {},
            'summary': {}
        }
        
        # Load small dataset samples
        datasets = ['a', 'b', 'c']
        dataset_samples = {}
        
        for dataset_name in datasets:
            try:
                df = pd.read_csv(f"../data/dataset_{dataset_name}_test.csv")
                texts = df['speechtext'].astype(str).tolist()[:100]  # Small sample
                dataset_samples[dataset_name] = texts
                logger.info(f"Loaded {len(texts)} sample texts from dataset {dataset_name}")
            except Exception as e:
                logger.warning(f"Could not load dataset {dataset_name}: {e}")
        
        # Cross-evaluation with small samples
        for model_name in self.models.keys():
            report['cross_evaluation'][model_name] = {}
            
            for dataset_name, texts in dataset_samples.items():
                perplexity = self.calculate_perplexity_sample(model_name, texts, sample_size=50)
                report['cross_evaluation'][model_name][f"dataset_{dataset_name}"] = {
                    'perplexity': perplexity,
                    'sample_size': min(50, len(texts))
                }
                logger.info(f"{model_name} on dataset_{dataset_name}: perplexity = {perplexity:.2f}")
        
        # Bias analysis (skipping for demo due to tokenizer compatibility)
        for model_name in self.models.keys():
            report['bias_analysis'][model_name] = {
                'note': 'Bias analysis skipped in demo - requires HuggingFace tokenizer compatibility'
            }
        
        # Summary
        perplexity_matrix = {}
        for model_name, results in report['cross_evaluation'].items():
            perplexity_matrix[model_name] = {k: v['perplexity'] for k, v in results.items()}
        
        report['summary'] = {
            'perplexity_matrix': perplexity_matrix,
            'note': 'This is a demo evaluation with small sample sizes for speed'
        }
        
        return report

def main():
    """Main demo evaluation function."""
    logger.info("Starting demo bias analysis")
    
    evaluator = DemoEvaluator()
    
    # Load demo models
    demo_models = ['a', 'b', 'c']
    models_loaded = 0
    
    for model_name in demo_models:
        model_path = f"../models/demo_model_{model_name}"
        if os.path.exists(model_path):
            evaluator.load_model(f"demo_model_{model_name}", model_path)
            models_loaded += 1
        else:
            logger.warning(f"Demo model {model_name} not found at {model_path}")
    
    if models_loaded == 0:
        logger.error("No demo models found!")
        return
    
    logger.info(f"Loaded {models_loaded} demo models")
    
    # Generate demo report
    report = evaluator.generate_demo_report()
    
    # Save report
    os.makedirs("../results", exist_ok=True)
    report_path = f"../results/demo_bias_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Demo report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("DEMO BIAS ANALYSIS SUMMARY")
    print("="*60)
    
    if 'perplexity_matrix' in report['summary']:
        print("\nPerplexity Matrix (lower is better):")
        print("-" * 40)
        
        for model_name, datasets in report['summary']['perplexity_matrix'].items():
            print(f"\n{model_name}:")
            for dataset, perplexity in datasets.items():
                print(f"  {dataset}: {perplexity:.2f}")
    
    print(f"\nNote: Bias analysis requires tokenizer compatibility fixes for production use.")
    print(f"This demo shows cross-evaluation perplexity results only.")
    
    print(f"\nDetailed report saved to: {report_path}")
    print("\n✅ Demo bias analysis completed!")

if __name__ == "__main__":
    main()