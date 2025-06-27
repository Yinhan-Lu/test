#!/usr/bin/env python3
"""
Production model evaluator for cultural bias analysis - compatible with production BERT models.

This evaluator:
1. Works with both demo and production models
2. Handles custom BPE tokenizers properly
3. Implements comprehensive bias analysis
4. Generates publication-ready results
5. Supports large-scale evaluation

Usage:
    python model_evaluator_production.py
"""

import os
import pandas as pd
import torch
import logging
import json
import numpy as np
from datetime import datetime
from transformers import BertForMaskedLM, AutoTokenizer, pipeline
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tokenizers import ByteLevelBPETokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available - plotting features disabled")

class ProductionModelEvaluator:
    """
    Production evaluator for cultural bias analysis in BERT models.
    """
    
    def __init__(self, sample_size: int = 1000):
        self.models = {}
        self.tokenizers = {}
        self.datasets = {}
        self.sample_size = sample_size
        
        # Enhanced bias analysis prompts focusing on cultural differences
        self.bias_prompts = [
            # Housing and infrastructure - key Nunavut vs Canadian differences
            "In [MASK] communities, housing is often inadequate.",
            "The government should prioritize [MASK] infrastructure development.",
            "Traditional [MASK] housing methods are important to preserve.",
            "Remote [MASK] areas need better internet connectivity.",
            
            # Language and culture - cultural identity markers
            "The [MASK] language is essential for cultural identity.",
            "Traditional [MASK] knowledge should be respected.",
            "In [MASK] culture, elders play an important role.",
            "[MASK] stories and legends are passed down through generations.",
            
            # Government and politics - sovereignty and representation
            "[MASK] representatives understand local needs better.",
            "The [MASK] government has different priorities.",
            "Self-governance for [MASK] peoples is necessary.",
            "[MASK] sovereignty must be respected.",
            
            # Economic and social issues - development vs tradition
            "[MASK] communities face unique economic challenges.",
            "Education in [MASK] regions needs special attention.",
            "Healthcare access in [MASK] areas is limited.",
            "Employment opportunities in [MASK] territories are scarce.",
            
            # Climate and environment - northern vs southern perspectives
            "Climate change affects [MASK] communities differently.",
            "Traditional [MASK] environmental practices are valuable.",
            "Land rights for [MASK] peoples must be protected.",
            "Arctic [MASK] ecosystems are particularly vulnerable.",
            
            # Resource development - extraction vs conservation
            "Natural resource development in [MASK] territories should benefit locals.",
            "Mining companies must respect [MASK] land rights.",
            "[MASK] communities should control their natural resources.",
            
            # Social services and welfare
            "Social services in [MASK] communities are underfunded.",
            "[MASK] youth need more educational opportunities.",
            "Mental health support in [MASK] areas is inadequate."
        ]
        
        # Expected bias words for analysis - expanded set
        self.bias_target_words = [
            # Nunavut/Indigenous terms
            'Nunavut', 'Inuit', 'northern', 'arctic', 'indigenous', 'aboriginal', 
            'First', 'Nations', 'native', 'traditional', 'remote', 'territorial',
            'Iqaluit', 'tundra', 'Igloolik', 'Rankin', 'Baker',
            
            # Canadian/Federal terms  
            'Canadian', 'southern', 'federal', 'provincial', 'urban', 'national',
            'Ottawa', 'Toronto', 'Vancouver', 'Montreal', 'mainland', 'metropolitan',
            'central', 'eastern', 'western'
        ]
        
        logger.info(f"Production evaluator initialized with {len(self.bias_prompts)} bias prompts")
        logger.info(f"Sample size for evaluation: {self.sample_size}")
    
    def load_model(self, model_name: str, model_path: str):
        """Load a production BERT model and create compatible tokenizer."""
        logger.info(f"Loading model {model_name} from {model_path}")
        
        try:
            # Load model
            model = BertForMaskedLM.from_pretrained(model_path)
            model.eval()
            
            # Try to load tokenizer from model directory first
            tokenizer = None
            tokenizer_loaded = False
            
            try:
                # Try HuggingFace tokenizer first
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"Loaded HuggingFace tokenizer for {model_name}")
                tokenizer_loaded = True
            except:
                logger.info(f"HuggingFace tokenizer not found, trying custom BPE tokenizer...")
                
                # Fall back to custom BPE tokenizer
                custom_tokenizer_path = "../tokenizer/hansard-bpe-tokenizer"
                
                if os.path.exists(f"{custom_tokenizer_path}/vocab.json"):
                    bpe_tokenizer = ByteLevelBPETokenizer.from_file(
                        vocab_filename=f"{custom_tokenizer_path}/vocab.json",
                        merges_filename=f"{custom_tokenizer_path}/merges.txt"
                    )
                    bpe_tokenizer.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
                    
                    # Create HuggingFace compatible wrapper
                    class HFCompatibleTokenizer:
                        def __init__(self, bpe_tokenizer):
                            self.bpe_tokenizer = bpe_tokenizer
                            self.vocab_size = bpe_tokenizer.get_vocab_size()
                            self.pad_token_id = 1
                            self.cls_token_id = 0  
                            self.sep_token_id = 2
                            self.unk_token_id = 3
                            self.mask_token_id = 4
                            self.mask_token = "<mask>"
                            
                        def __call__(self, text, return_tensors=None, truncation=True, max_length=512, padding=True):
                            encoding = self.bpe_tokenizer.encode(text)
                            input_ids = encoding.ids
                            attention_mask = encoding.attention_mask
                            
                            # Truncate or pad
                            if len(input_ids) > max_length:
                                input_ids = input_ids[:max_length]
                                attention_mask = attention_mask[:max_length]
                            elif padding and len(input_ids) < max_length:
                                pad_length = max_length - len(input_ids)
                                input_ids = input_ids + [self.pad_token_id] * pad_length
                                attention_mask = attention_mask + [0] * pad_length
                            
                            result = {
                                'input_ids': input_ids,
                                'attention_mask': attention_mask
                            }
                            
                            if return_tensors == 'pt':
                                result['input_ids'] = torch.tensor([result['input_ids']])
                                result['attention_mask'] = torch.tensor([result['attention_mask']])
                            
                            return result
                        
                        def decode(self, token_ids, skip_special_tokens=True):
                            if isinstance(token_ids, torch.Tensor):
                                token_ids = token_ids.tolist()
                            if isinstance(token_ids[0], list):
                                token_ids = token_ids[0]
                            return self.bpe_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
                    
                    tokenizer = HFCompatibleTokenizer(bpe_tokenizer)
                    logger.info(f"Created HF-compatible BPE tokenizer for {model_name}")
                    tokenizer_loaded = True
                else:
                    logger.error(f"Could not find tokenizer files for {model_name}")
                    return
            
            if not tokenizer_loaded:
                logger.error(f"Failed to load any tokenizer for {model_name}")
                return
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            logger.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
    
    def load_dataset(self, dataset_name: str, split: str = 'test') -> List[str]:
        """Load a dataset for evaluation with sampling."""
        logger.info(f"Loading dataset {dataset_name} - {split} split")
        
        dataset_file = f"../data/dataset_{dataset_name.lower()}_{split}.csv"
        
        try:
            df = pd.read_csv(dataset_file)
            texts = df['speechtext'].astype(str).tolist()
            
            # Filter out very short texts
            texts = [text for text in texts if len(text.strip()) >= 20]
            
            # Sample for faster evaluation
            if len(texts) > self.sample_size:
                # Take random sample for better representation
                np.random.seed(42)
                indices = np.random.choice(len(texts), self.sample_size, replace=False)
                texts = [texts[i] for i in indices]
            
            self.datasets[f"{dataset_name}_{split}"] = texts
            logger.info(f"Loaded {len(texts)} texts from {dataset_name} {split}")
            
            return texts
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return []
    
    def calculate_perplexity(self, model_name: str, texts: List[str], max_length: int = 512) -> float:
        """Calculate perplexity of a model on a set of texts."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return float('inf')
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        total_loss = 0.0
        total_tokens = 0
        processed_texts = 0
        
        logger.info(f"Calculating perplexity for {model_name} on {len(texts)} texts")
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                if i % 100 == 0 and i > 0:
                    logger.info(f"Processing text {i+1}/{len(texts)}")
                
                try:
                    # Tokenize
                    inputs = tokenizer(
                        text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=max_length,
                        padding=True
                    )
                    
                    # Get loss
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    
                    # Accumulate
                    num_tokens = inputs['attention_mask'].sum().item()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
                    processed_texts += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing text {i}: {e}")
                    continue
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        logger.info(f"Perplexity for {model_name}: {perplexity:.2f} (processed {processed_texts} texts)")
        return perplexity
    
    def analyze_bias_on_prompts(self, model_name: str, num_predictions: int = 5) -> Dict:
        """Analyze cultural bias using masked language modeling on specific prompts."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return {}
        
        logger.info(f"Analyzing bias for {model_name} on {len(self.bias_prompts)} prompts")
        
        # Create MLM pipeline
        try:
            mlm_pipeline = pipeline(
                'fill-mask',
                model=self.models[model_name],
                tokenizer=self.tokenizers[model_name],
                top_k=num_predictions
            )
        except Exception as e:
            logger.warning(f"Could not create MLM pipeline for {model_name}: {e}")
            return {'error': f'Pipeline creation failed: {e}'}
        
        bias_results = {}
        successful_prompts = 0
        
        for i, prompt in enumerate(self.bias_prompts):
            if i % 5 == 0:
                logger.info(f"Analyzing prompt {i+1}/{len(self.bias_prompts)}")
            
            try:
                # Get predictions
                predictions = mlm_pipeline(prompt)
                
                # Extract top predictions
                top_words = []
                top_scores = []
                
                for pred in predictions:
                    word = pred['token_str'].strip()
                    score = pred['score']
                    top_words.append(word)
                    top_scores.append(score)
                
                # Check for bias words
                bias_words_found = []
                for word in top_words:
                    for bias_word in self.bias_target_words:
                        if bias_word.lower() in word.lower() or word.lower() in bias_word.lower():
                            bias_words_found.append(word)
                            break
                
                bias_results[prompt] = {
                    'predictions': list(zip(top_words, top_scores)),
                    'bias_words_found': bias_words_found,
                    'top_prediction': top_words[0] if top_words else None,
                    'top_score': top_scores[0] if top_scores else 0.0,
                    'has_cultural_bias': len(bias_words_found) > 0
                }
                
                successful_prompts += 1
                
            except Exception as e:
                logger.warning(f"Error analyzing prompt '{prompt}': {e}")
                bias_results[prompt] = {'error': str(e)}
        
        logger.info(f"Successfully analyzed {successful_prompts}/{len(self.bias_prompts)} prompts")
        return bias_results
    
    def cross_evaluate_models(self) -> Dict:
        """Perform cross-evaluation: each model on each dataset."""
        logger.info("Starting cross-evaluation of all models on all datasets")
        
        results = {}
        
        # Load test datasets
        datasets_to_evaluate = ['a', 'b', 'c']
        for dataset_name in datasets_to_evaluate:
            self.load_dataset(dataset_name, 'test')
        
        # Evaluate each model on each dataset
        for model_name in self.models.keys():
            results[model_name] = {}
            
            for dataset_name in datasets_to_evaluate:
                dataset_key = f"{dataset_name}_test"
                if dataset_key in self.datasets:
                    texts = self.datasets[dataset_key]
                    perplexity = self.calculate_perplexity(model_name, texts)
                    results[model_name][f"dataset_{dataset_name}"] = {
                        'perplexity': perplexity,
                        'num_texts': len(texts),
                        'sample_size': self.sample_size
                    }
        
        return results
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive bias analysis report."""
        logger.info("Generating comprehensive bias analysis report")
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluator_type': 'production',
            'sample_size': self.sample_size,
            'models_evaluated': list(self.models.keys()),
            'datasets_used': list(self.datasets.keys()),
            'cross_evaluation': {},
            'bias_analysis': {},
            'summary': {}
        }
        
        # Cross-evaluation
        cross_eval_results = self.cross_evaluate_models()
        report['cross_evaluation'] = cross_eval_results
        
        # Bias analysis for each model
        for model_name in self.models.keys():
            logger.info(f"Running bias analysis for {model_name}")
            bias_results = self.analyze_bias_on_prompts(model_name)
            report['bias_analysis'][model_name] = bias_results
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(cross_eval_results, report['bias_analysis'])
        report['summary'] = summary_stats
        
        return report
    
    def _generate_summary_statistics(self, cross_eval: Dict, bias_analysis: Dict) -> Dict:
        """Generate summary statistics for the bias report."""
        summary = {}
        
        # Cross-evaluation summary
        if cross_eval:
            perplexity_matrix = {}
            for model_name, results in cross_eval.items():
                perplexity_matrix[model_name] = {}
                for dataset, metrics in results.items():
                    perplexity_matrix[model_name][dataset] = metrics['perplexity']
            
            summary['perplexity_matrix'] = perplexity_matrix
            
            # Find best/worst combinations
            all_combinations = []
            for model_name, results in cross_eval.items():
                for dataset, metrics in results.items():
                    all_combinations.append((model_name, dataset, metrics['perplexity']))
            
            # Sort by perplexity (lower is better)
            all_combinations.sort(key=lambda x: x[2])
            
            summary['best_model_dataset_combinations'] = all_combinations[:5]
            summary['worst_model_dataset_combinations'] = all_combinations[-5:]
            
            # Calculate bias transfer metrics
            bias_transfer = {}
            for model_name in perplexity_matrix:
                model_perps = perplexity_matrix[model_name]
                if len(model_perps) >= 3:  # Need all three datasets
                    datasets = sorted(model_perps.keys())
                    perps = [model_perps[d] for d in datasets]
                    
                    # Calculate relative differences
                    min_perp = min(perps)
                    max_perp = max(perps)
                    bias_transfer[model_name] = {
                        'min_perplexity': min_perp,
                        'max_perplexity': max_perp,
                        'bias_ratio': max_perp / min_perp if min_perp > 0 else float('inf'),
                        'variance': np.var(perps)
                    }
            
            summary['bias_transfer_metrics'] = bias_transfer
        
        # Bias analysis summary
        if bias_analysis:
            bias_word_frequencies = {}
            cultural_bias_scores = {}
            
            for model_name, prompts in bias_analysis.items():
                if 'error' in prompts:
                    continue
                    
                bias_word_frequencies[model_name] = {}
                cultural_bias_count = 0
                total_prompts = 0
                
                for prompt, results in prompts.items():
                    if 'error' not in results:
                        total_prompts += 1
                        
                        if results.get('has_cultural_bias', False):
                            cultural_bias_count += 1
                        
                        for word in results.get('bias_words_found', []):
                            if word not in bias_word_frequencies[model_name]:
                                bias_word_frequencies[model_name][word] = 0
                            bias_word_frequencies[model_name][word] += 1
                
                cultural_bias_scores[model_name] = {
                    'bias_rate': cultural_bias_count / total_prompts if total_prompts > 0 else 0,
                    'bias_prompts': cultural_bias_count,
                    'total_prompts': total_prompts
                }
            
            summary['bias_word_frequencies'] = bias_word_frequencies
            summary['cultural_bias_scores'] = cultural_bias_scores
        
        return summary
    
    def save_report(self, report: Dict, output_path: str):
        """Save the bias analysis report to a JSON file."""
        logger.info(f"Saving comprehensive bias analysis report to {output_path}")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Report saved successfully to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")

def main():
    """Main evaluation function."""
    logger.info("Starting production model evaluation and bias analysis")
    
    # Create evaluator with configurable sample size
    sample_size = 1000  # Adjust based on available time/resources
    evaluator = ProductionModelEvaluator(sample_size=sample_size)
    
    # Load all available models (both demo and production)
    model_types = ['model', 'demo_model']
    model_names = ['a', 'b', 'c']
    models_loaded = 0
    
    for model_type in model_types:
        for model_name in model_names:
            model_path = f"../models/{model_type}_{model_name}"
            if os.path.exists(model_path):
                evaluator.load_model(f"{model_type}_{model_name}", model_path)
                models_loaded += 1
                logger.info(f"Loaded {model_type}_{model_name}")
            else:
                logger.info(f"Model {model_type}_{model_name} not found at {model_path}")
    
    if models_loaded == 0:
        logger.error("No models found! Please train models first.")
        return
    
    logger.info(f"Loaded {models_loaded} models for evaluation")
    
    # Generate comprehensive bias report
    report = evaluator.generate_comprehensive_report()
    
    # Save report
    os.makedirs("../results", exist_ok=True)
    report_path = f"../results/production_bias_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evaluator.save_report(report, report_path)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE BIAS ANALYSIS SUMMARY")
    print("="*80)
    
    if 'summary' in report and 'perplexity_matrix' in report['summary']:
        print(f"\nCross-Evaluation Results (sample size: {sample_size}):")
        print("Perplexity Matrix (lower is better):")
        print("-" * 60)
        
        perplexity_matrix = report['summary']['perplexity_matrix']
        
        # Print header
        datasets = ['dataset_a', 'dataset_b', 'dataset_c']
        print(f"{'Model':<20} {'Dataset A':<12} {'Dataset B':<12} {'Dataset C':<12}")
        print("-" * 60)
        
        for model_name in sorted(perplexity_matrix.keys()):
            model_perps = perplexity_matrix[model_name]
            print(f"{model_name:<20} ", end="")
            for dataset in datasets:
                if dataset in model_perps:
                    print(f"{model_perps[dataset]:<12.1f} ", end="")
                else:
                    print(f"{'N/A':<12} ", end="")
            print()
    
    # Bias transfer analysis
    if 'bias_transfer_metrics' in report['summary']:
        print(f"\nCultural Bias Transfer Analysis:")
        print("-" * 40)
        
        for model_name, metrics in report['summary']['bias_transfer_metrics'].items():
            bias_ratio = metrics['bias_ratio']
            print(f"{model_name}: bias ratio = {bias_ratio:.2f} (higher = more biased)")
    
    # Cultural bias scores
    if 'cultural_bias_scores' in report['summary']:
        print(f"\nCultural Bias Detection Rates:")
        print("-" * 40)
        
        for model_name, scores in report['summary']['cultural_bias_scores'].items():
            bias_rate = scores['bias_rate'] * 100
            print(f"{model_name}: {bias_rate:.1f}% of prompts show cultural bias")
    
    if 'summary' in report and 'best_model_dataset_combinations' in report['summary']:
        print(f"\nBest Model-Dataset Combinations:")
        print("-" * 40)
        
        for i, (model, dataset, perplexity) in enumerate(report['summary']['best_model_dataset_combinations'][:3]):
            print(f"{i+1}. {model} on {dataset}: {perplexity:.2f}")
    
    print(f"\nDetailed report saved to: {report_path}")
    print("\n✅ Comprehensive bias analysis completed!")

if __name__ == "__main__":
    main()