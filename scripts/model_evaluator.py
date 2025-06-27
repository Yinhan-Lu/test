import os
import pandas as pd
import torch
import logging
import json
import numpy as np
from datetime import datetime
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

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

class ModelEvaluator:
    """
    Comprehensive evaluator for cultural bias analysis in BERT models.
    
    This evaluator:
    1. Calculates perplexity on different datasets
    2. Performs cross-evaluation (each model on each dataset)
    3. Analyzes cultural bias through masked language modeling
    4. Generates comparative reports and visualizations
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.datasets = {}
        
        # Bias analysis prompts focusing on cultural differences
        self.bias_prompts = [
            # Housing and infrastructure
            "In [MASK] communities, housing is often inadequate.",
            "The government should prioritize [MASK] infrastructure development.",
            "Traditional [MASK] housing methods are important to preserve.",
            
            # Language and culture
            "The [MASK] language is essential for cultural identity.",
            "Traditional [MASK] knowledge should be respected.",
            "In [MASK] culture, elders play an important role.",
            
            # Government and politics
            "[MASK] representatives understand local needs better.",
            "The [MASK] government has different priorities.",
            "Self-governance for [MASK] peoples is necessary.",
            
            # Economic and social issues
            "[MASK] communities face unique economic challenges.",
            "Education in [MASK] regions needs special attention.",
            "Healthcare access in [MASK] areas is limited.",
            
            # Climate and environment
            "Climate change affects [MASK] communities differently.",
            "Traditional [MASK] environmental practices are valuable.",
            "Land rights for [MASK] peoples must be protected."
        ]
        
        # Expected bias words for analysis
        self.bias_target_words = [
            'Nunavut', 'Inuit', 'northern', 'arctic', 'indigenous',
            'Canadian', 'southern', 'federal', 'provincial', 'urban'
        ]
        
        logger.info("Model evaluator initialized")
    
    def load_model(self, model_name: str, model_path: str):
        """Load a trained BERT model and its tokenizer."""
        logger.info(f"Loading model {model_name} from {model_path}")
        
        try:
            # Load model
            model = BertForMaskedLM.from_pretrained(model_path)
            
            # Try to load tokenizer - first try AutoTokenizer for demo models
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"Loaded tokenizer using AutoTokenizer for {model_name}")
            except:
                try:
                    tokenizer = BertTokenizer.from_pretrained(model_path)
                    logger.info(f"Loaded tokenizer using BertTokenizer for {model_name}")
                except Exception as e:
                    logger.warning(f"Could not load tokenizer for {model_name}: {e}")
                    # Create a dummy tokenizer for demo purposes
                    from tokenizers import ByteLevelBPETokenizer
                    custom_tokenizer_path = "../tokenizer/hansard-bpe-tokenizer"
                    
                    if os.path.exists(f"{custom_tokenizer_path}/vocab.json"):
                        bpe_tokenizer = ByteLevelBPETokenizer.from_file(
                            vocab_filename=f"{custom_tokenizer_path}/vocab.json",
                            merges_filename=f"{custom_tokenizer_path}/merges.txt"
                        )
                        bpe_tokenizer.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
                        
                        # Create a wrapper to make it compatible
                        class BPETokenizerWrapper:
                            def __init__(self, bpe_tokenizer):
                                self.bpe_tokenizer = bpe_tokenizer
                                self.vocab_size = bpe_tokenizer.get_vocab_size()
                                
                            def __call__(self, text, return_tensors=None, truncation=True, max_length=256, padding=True):
                                encoding = self.bpe_tokenizer.encode(text)
                                input_ids = encoding.ids
                                attention_mask = encoding.attention_mask
                                
                                # Truncate or pad
                                if len(input_ids) > max_length:
                                    input_ids = input_ids[:max_length]
                                    attention_mask = attention_mask[:max_length]
                                elif padding and len(input_ids) < max_length:
                                    pad_length = max_length - len(input_ids)
                                    input_ids = input_ids + [1] * pad_length
                                    attention_mask = attention_mask + [0] * pad_length
                                
                                result = {
                                    'input_ids': torch.tensor([input_ids]) if return_tensors == 'pt' else input_ids,
                                    'attention_mask': torch.tensor([attention_mask]) if return_tensors == 'pt' else attention_mask
                                }
                                return result
                        
                        tokenizer = BPETokenizerWrapper(bpe_tokenizer)
                        logger.info(f"Created BPE tokenizer wrapper for {model_name}")
                    else:
                        logger.error(f"Could not find custom tokenizer files for {model_name}")
                        return
            
            # Set to evaluation mode
            model.eval()
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            logger.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
    
    def load_dataset(self, dataset_name: str, split: str = 'test') -> List[str]:
        """Load a dataset for evaluation."""
        logger.info(f"Loading dataset {dataset_name} - {split} split")
        
        dataset_file = f"../data/dataset_{dataset_name.lower()}_{split}.csv"
        
        try:
            df = pd.read_csv(dataset_file)
            texts = df['speechtext'].astype(str).tolist()
            
            # Filter out very short texts
            texts = [text for text in texts if len(text.strip()) >= 20]
            
            # Take a sample for evaluation (to speed up)
            if len(texts) > 1000:
                texts = texts[:1000]
            
            self.datasets[f"{dataset_name}_{split}"] = texts
            logger.info(f"Loaded {len(texts)} texts from {dataset_name} {split}")
            
            return texts
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return []
    
    def calculate_perplexity(self, model_name: str, texts: List[str], max_length: int = 256) -> float:
        """Calculate perplexity of a model on a set of texts."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return float('inf')
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        total_loss = 0.0
        total_tokens = 0
        
        logger.info(f"Calculating perplexity for {model_name} on {len(texts)} texts")
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                if i % 100 == 0:
                    logger.info(f"Processing text {i+1}/{len(texts)}")
                
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
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        logger.info(f"Perplexity for {model_name}: {perplexity:.2f}")
        return perplexity
    
    def analyze_bias_on_prompts(self, model_name: str, num_predictions: int = 5) -> Dict:
        """Analyze cultural bias using masked language modeling on specific prompts."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return {}
        
        logger.info(f"Analyzing bias for {model_name} on {len(self.bias_prompts)} prompts")
        
        # Create MLM pipeline
        mlm_pipeline = pipeline(
            'fill-mask',
            model=self.models[model_name],
            tokenizer=self.tokenizers[model_name],
            top_k=num_predictions
        )
        
        bias_results = {}
        
        for i, prompt in enumerate(self.bias_prompts):
            logger.info(f"Analyzing prompt {i+1}: {prompt}")
            
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
                
                bias_results[prompt] = {
                    'predictions': list(zip(top_words, top_scores)),
                    'bias_words_found': [word for word in top_words if word.lower() in [w.lower() for w in self.bias_target_words]],
                    'top_prediction': top_words[0] if top_words else None,
                    'top_score': top_scores[0] if top_scores else 0.0
                }
                
            except Exception as e:
                logger.error(f"Error analyzing prompt '{prompt}': {e}")
                bias_results[prompt] = {'error': str(e)}
        
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
                        'num_texts': len(texts)
                    }
        
        return results
    
    def generate_bias_report(self) -> Dict:
        """Generate comprehensive bias analysis report."""
        logger.info("Generating comprehensive bias analysis report")
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
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
            best_combinations = []
            worst_combinations = []
            
            for model_name, results in cross_eval.items():
                for dataset, metrics in results.items():
                    combo = (model_name, dataset, metrics['perplexity'])
                    best_combinations.append(combo)
                    worst_combinations.append(combo)
            
            best_combinations.sort(key=lambda x: x[2])
            worst_combinations.sort(key=lambda x: x[2], reverse=True)
            
            summary['best_model_dataset_combinations'] = best_combinations[:5]
            summary['worst_model_dataset_combinations'] = worst_combinations[:5]
        
        # Bias analysis summary
        if bias_analysis:
            bias_word_frequencies = {}
            
            for model_name, prompts in bias_analysis.items():
                bias_word_frequencies[model_name] = {}
                
                for prompt, results in prompts.items():
                    if 'bias_words_found' in results:
                        for word in results['bias_words_found']:
                            if word not in bias_word_frequencies[model_name]:
                                bias_word_frequencies[model_name][word] = 0
                            bias_word_frequencies[model_name][word] += 1
            
            summary['bias_word_frequencies'] = bias_word_frequencies
        
        return summary
    
    def save_report(self, report: Dict, output_path: str):
        """Save the bias analysis report to a JSON file."""
        logger.info(f"Saving bias analysis report to {output_path}")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Report saved successfully to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")

def main():
    """Main evaluation function."""
    logger.info("Starting model evaluation and bias analysis")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Load all trained models (checking both full and demo models)
    models_to_load = ['a', 'b', 'c']
    models_loaded = 0
    
    for model_name in models_to_load:
        # Try demo models first
        demo_model_path = f"../models/demo_model_{model_name}"
        full_model_path = f"../models/model_{model_name}"
        
        if os.path.exists(demo_model_path):
            evaluator.load_model(f"demo_model_{model_name}", demo_model_path)
            models_loaded += 1
            logger.info(f"Loaded demo model {model_name} from {demo_model_path}")
        elif os.path.exists(full_model_path):
            evaluator.load_model(f"model_{model_name}", full_model_path)
            models_loaded += 1
            logger.info(f"Loaded full model {model_name} from {full_model_path}")
        else:
            logger.warning(f"Model {model_name} not found at {demo_model_path} or {full_model_path}")
    
    if models_loaded == 0:
        logger.error("No trained models found! Please train models first.")
        return
    
    logger.info(f"Loaded {models_loaded} models for evaluation")
    
    # Generate comprehensive bias report
    report = evaluator.generate_bias_report()
    
    # Save report
    os.makedirs("../results", exist_ok=True)
    report_path = f"../results/bias_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evaluator.save_report(report, report_path)
    
    # Print summary
    print("\n" + "="*60)
    print("BIAS ANALYSIS SUMMARY")
    print("="*60)
    
    if 'summary' in report and 'perplexity_matrix' in report['summary']:
        print("\nPerplexity Matrix (lower is better):")
        print("-" * 40)
        
        perplexity_matrix = report['summary']['perplexity_matrix']
        for model_name, datasets in perplexity_matrix.items():
            print(f"\n{model_name}:")
            for dataset, perplexity in datasets.items():
                print(f"  {dataset}: {perplexity:.2f}")
    
    if 'summary' in report and 'best_model_dataset_combinations' in report['summary']:
        print(f"\nBest Model-Dataset Combinations:")
        print("-" * 40)
        
        for i, (model, dataset, perplexity) in enumerate(report['summary']['best_model_dataset_combinations'][:3]):
            print(f"{i+1}. {model} on {dataset}: {perplexity:.2f}")
    
    print(f"\nDetailed report saved to: {report_path}")
    print("\n✅ Bias analysis completed!")

if __name__ == "__main__":
    main()