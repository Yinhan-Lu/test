#!/usr/bin/env python3
"""
Production BERT trainer for parliamentary language models - optimized for GPU training.

This trainer implements:
1. Automatic GPU detection and optimization
2. Mixed precision training (FP16) for memory efficiency
3. Gradient accumulation for effective large batch sizes
4. Learning rate scheduling with warmup
5. Automatic checkpointing and resuming
6. Memory-efficient data loading
7. Production logging and monitoring

Usage:
    python bert_trainer_production.py <dataset_name>
    
Example:
    python bert_trainer_production.py a  # Train on Dataset A (Nunavut)
    python bert_trainer_production.py b  # Train on Dataset B (Canadian)
    python bert_trainer_production.py c  # Train on Dataset C (Mixed)
"""

import os
import gc
import psutil
import pandas as pd
import torch
import logging
import json
from datetime import datetime
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    Trainer,
    TrainingArguments,
    set_seed,
    get_linear_schedule_with_warmup
)
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/production_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedHansardDataset(Dataset):
    """
    Memory-optimized dataset class for Hansard speeches.
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512, cache_tokenization: bool = True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
        self._tokenized_cache = {}
        
        logger.info(f"Created dataset with {len(texts)} texts, max_length={max_length}")
        
        # Pre-tokenize a sample to estimate memory usage
        if len(texts) > 0:
            sample_encoding = self.tokenizer.encode(str(texts[0]))
            logger.info(f"Sample tokenization: {len(sample_encoding.ids)} tokens")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if self.cache_tokenization and idx in self._tokenized_cache:
            return self._tokenized_cache[idx]
        
        text = str(self.texts[idx])
        
        # Tokenize with our BPE tokenizer
        encoding = self.tokenizer.encode(text)
        
        # Truncate or pad to max_length
        input_ids = encoding.ids
        attention_mask = encoding.attention_mask
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        else:
            # Pad with pad_token_id (which is 1 for our tokenizer)
            pad_length = self.max_length - len(input_ids)
            input_ids = input_ids + [1] * pad_length  # 1 is <pad> token
            attention_mask = attention_mask + [0] * pad_length
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
        # Cache if enabled and reasonable size
        if self.cache_tokenization and len(self._tokenized_cache) < 10000:
            self._tokenized_cache[idx] = result
            
        return result

class ProductionBERTTrainer:
    """
    Production BERT trainer with GPU optimization and advanced features.
    """
    
    def __init__(self, dataset_name: str, custom_tokenizer_path: str, use_balanced: bool = False):
        self.dataset_name = dataset_name.upper()
        self.use_balanced = use_balanced
        self.model_name = f"model_{dataset_name.lower()}{'_balanced' if use_balanced else ''}"
        self.custom_tokenizer_path = custom_tokenizer_path
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Initialize paths
        self.model_dir = f"../models/{self.model_name}"
        self.log_dir = f"../logs/{self.model_name}"
        self.checkpoint_dir = f"../checkpoints/{self.model_name}"
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Detect and configure device
        self.device = self._setup_device()
        
        logger.info(f"Initializing Production BERT trainer for {self.dataset_name}")
        logger.info(f"Dataset type: {'Balanced' if self.use_balanced else 'Original'}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model directory: {self.model_dir}")
        
    def _setup_device(self) -> torch.device:
        """Setup optimal device configuration."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"GPU detected: {gpu_name}")
            logger.info(f"GPU memory: {gpu_memory:.1f}GB")
            logger.info(f"GPU count: {gpu_count}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU detected, using CPU (will be slow)")
            
        return device
    
    def _get_optimal_batch_size(self, model_size_mb: float) -> Tuple[int, int]:
        """Calculate optimal batch size based on available memory."""
        if self.device.type == "cuda":
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Conservative estimates for BERT-base
            if gpu_memory_gb >= 24:
                return 32, 4  # train_batch_size, gradient_accumulation_steps
            elif gpu_memory_gb >= 16:
                return 16, 8
            elif gpu_memory_gb >= 12:
                return 8, 16
            elif gpu_memory_gb >= 8:
                return 4, 32
            else:
                return 2, 64
        elif self.device.type == "mps":
            return 8, 8  # Conservative for MPS
        else:
            return 4, 16  # CPU fallback
    
    def _log_system_info(self):
        """Log system resource information."""
        # CPU info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1e9
        
        logger.info(f"System info:")
        logger.info(f"  CPU cores: {cpu_count}")
        logger.info(f"  RAM: {memory_gb:.1f}GB")
        
        # GPU info
        if self.device.type == "cuda":
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  GPU memory: {gpu_memory_gb:.1f}GB")
    
    def load_custom_tokenizer(self):
        """Load our trained BPE tokenizer."""
        logger.info("Loading custom BPE tokenizer...")
        
        tokenizer = ByteLevelBPETokenizer.from_file(
            vocab_filename=f"{self.custom_tokenizer_path}/vocab.json",
            merges_filename=f"{self.custom_tokenizer_path}/merges.txt"
        )
        
        # Set special tokens
        tokenizer.add_special_tokens([
            "<s>", "<pad>", "</s>", "<unk>", "<mask>"
        ])
        
        logger.info(f"Custom tokenizer loaded with vocab size: {tokenizer.get_vocab_size()}")
        return tokenizer
    
    def load_dataset(self, split: str = 'train') -> List[str]:
        """Load dataset with memory optimization."""
        dataset_type = "balanced" if self.use_balanced else "original"
        logger.info(f"Loading {dataset_type} dataset {self.dataset_name} - {split} split...")
        
        if self.use_balanced:
            dataset_file = f"../data/dataset_{self.dataset_name.lower()}_balanced_{split}.csv"
        else:
            dataset_file = f"../data/dataset_{self.dataset_name.lower()}_{split}.csv"
        
        try:
            # Use chunking for large files
            chunk_size = 10000
            texts = []
            
            for chunk in pd.read_csv(dataset_file, chunksize=chunk_size):
                chunk_texts = chunk['speechtext'].astype(str).tolist()
                # Filter out very short texts
                chunk_texts = [text for text in chunk_texts if len(text.strip()) >= 20]
                texts.extend(chunk_texts)
                
                # Log progress for large datasets
                if len(texts) % 50000 == 0:
                    logger.info(f"Loaded {len(texts)} speeches so far...")
            
            logger.info(f"Loaded {len(texts)} speeches from {split} split")
            return texts
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    
    def create_production_model_config(self, tokenizer):
        """Create production BERT model configuration."""
        vocab_size = tokenizer.get_vocab_size()
        
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=768,           # Full BERT-base size
            num_hidden_layers=12,      # Full BERT-base layers
            num_attention_heads=12,    # Full BERT-base heads
            intermediate_size=3072,    # Full BERT-base FFN size
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,  # <pad> token ID
            position_embedding_type="absolute"
        )
        
        logger.info(f"Created production BERT config:")
        logger.info(f"  Vocab size: {config.vocab_size:,}")
        logger.info(f"  Hidden size: {config.hidden_size}")
        logger.info(f"  Layers: {config.num_hidden_layers}")
        logger.info(f"  Attention heads: {config.num_attention_heads}")
        
        return config
    
    def train_production_model(self, 
                             num_epochs: int = 3, 
                             max_steps: Optional[int] = None,
                             resume_from_checkpoint: Optional[str] = None):
        """
        Train production BERT model with all optimizations.
        """
        logger.info(f"Starting production BERT training for {self.dataset_name}")
        self._log_system_info()
        
        # Load tokenizer and datasets
        tokenizer = self.load_custom_tokenizer()
        
        train_texts = self.load_dataset('train')
        val_texts = self.load_dataset('val')
        
        if not train_texts or not val_texts:
            logger.error("Failed to load datasets!")
            return None
        
        # Create datasets with optimizations
        train_dataset = OptimizedHansardDataset(
            train_texts, tokenizer, max_length=512, cache_tokenization=False  # Don't cache for large datasets
        )
        val_dataset = OptimizedHansardDataset(
            val_texts, tokenizer, max_length=512, cache_tokenization=True  # Cache validation set
        )
        
        logger.info(f"Created datasets: train={len(train_dataset):,}, val={len(val_dataset):,}")
        
        # Create model
        config = self.create_production_model_config(tokenizer)
        model = BertForMaskedLM(config)
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / 1e6  # Assuming float32
        
        logger.info(f"Created BERT model with {param_count:,} parameters ({model_size_mb:.1f}MB)")
        
        # Move model to device
        model = model.to(self.device)
        
        # Get optimal batch sizes
        train_batch_size, gradient_accumulation_steps = self._get_optimal_batch_size(model_size_mb)
        effective_batch_size = train_batch_size * gradient_accumulation_steps
        
        logger.info(f"Batch configuration:")
        logger.info(f"  Per-device batch size: {train_batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        
        # Calculate training steps
        total_train_samples = len(train_dataset)
        steps_per_epoch = total_train_samples // effective_batch_size
        total_steps = steps_per_epoch * num_epochs if max_steps is None else max_steps
        warmup_steps = min(1000, total_steps // 10)  # 10% warmup
        
        logger.info(f"Training configuration:")
        logger.info(f"  Total samples: {total_train_samples:,}")
        logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  Warmup steps: {warmup_steps:,}")
        
        # Custom MLM data collator
        def data_collator(examples):
            batch = {}
            batch['input_ids'] = torch.stack([example['input_ids'] for example in examples])
            batch['attention_mask'] = torch.stack([example['attention_mask'] for example in examples])
            
            # Create labels for MLM (clone input_ids)
            batch['labels'] = batch['input_ids'].clone()
            
            # Mask 15% of tokens randomly for MLM
            probability_matrix = torch.full(batch['labels'].shape, 0.15)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            # Don't mask special tokens (pad, cls, sep)
            special_tokens_mask = (batch['input_ids'] == 1) | (batch['input_ids'] == 0) | (batch['input_ids'] == 2)
            masked_indices = masked_indices & ~special_tokens_mask
            
            # Replace masked tokens with [MASK] token (ID 4)
            batch['input_ids'][masked_indices] = 4  # <mask> token
            
            # Only compute loss on masked tokens
            batch['labels'][~masked_indices] = -100
            
            return batch
        
        # Production training arguments
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            # Learning rate and optimization
            learning_rate=2e-5,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            warmup_steps=warmup_steps,
            
            # Mixed precision training (FP16) for GPU
            fp16=self.device.type == "cuda",
            fp16_opt_level="O1",
            
            # Logging and evaluation
            logging_dir=self.log_dir,
            logging_steps=100,
            logging_first_step=True,
            eval_steps=1000,
            eval_strategy="steps",
            
            # Checkpointing
            save_steps=1000,
            save_strategy="steps",
            save_total_limit=3,  # Keep only 3 latest checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Performance optimizations
            dataloader_num_workers=4 if self.device.type != "cpu" else 0,
            dataloader_pin_memory=self.device.type == "cuda",
            remove_unused_columns=False,
            
            # Disable external reporting for now
            report_to=None,
            
            # Reproducibility
            seed=42,
            data_seed=42,
            
            # Early stopping patience
            early_stopping_patience=3,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Add custom callbacks for monitoring
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Start training
        logger.info("Starting production training...")
        start_time = datetime.now()
        
        try:
            if resume_from_checkpoint:
                logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
                train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                train_result = trainer.train()
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            return None
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            return None
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        logger.info(f"Training completed in {training_time}")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Save final model and tokenizer
        logger.info("Saving production model...")
        trainer.save_model()
        
        # Save our custom tokenizer
        tokenizer.save_model(self.model_dir)
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_result = trainer.evaluate()
        final_perplexity = np.exp(eval_result['eval_loss'])
        
        logger.info(f"Final evaluation loss: {eval_result['eval_loss']:.4f}")
        logger.info(f"Final perplexity: {final_perplexity:.2f}")
        
        # Save comprehensive training summary
        training_summary = {
            'model_name': self.model_name,
            'dataset': self.dataset_name,
            'model_type': 'production',
            'dataset_type': 'balanced' if self.use_balanced else 'original',
            'balanced_training': self.use_balanced,
            'training_time': str(training_time),
            'final_loss': train_result.training_loss,
            'final_eval_loss': eval_result['eval_loss'],
            'final_perplexity': final_perplexity,
            'num_epochs': num_epochs,
            'total_steps': train_result.global_step,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'vocab_size': tokenizer.get_vocab_size(),
            'max_length': 512,
            'parameters': param_count,
            'model_size_mb': model_size_mb,
            'effective_batch_size': effective_batch_size,
            'device': str(self.device),
            'fp16_enabled': training_args.fp16,
            'training_completed': end_time.isoformat(),
            'methodology_note': 'Balanced datasets ensure equal sample sizes for fair bias comparison' if self.use_balanced else 'Original unbalanced datasets',
            'config': {
                'hidden_size': config.hidden_size,
                'num_layers': config.num_hidden_layers,
                'num_attention_heads': config.num_attention_heads,
                'vocab_size': config.vocab_size
            }
        }
        
        summary_path = os.path.join(self.model_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_path}")
        
        # Clear GPU memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        return training_summary

def main():
    """Main production training function."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Train production BERT model")
    parser.add_argument('dataset_name', choices=['a', 'b', 'c'], 
                       help='Dataset to train on: a (Nunavut), b (Canadian), c (Mixed)')
    parser.add_argument('--balanced', action='store_true',
                       help='Use balanced datasets with equal sample sizes for fair bias comparison')
    parser.add_argument('--resume-from', type=str,
                       help='Resume training from checkpoint')
    
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nThis will train a full production BERT model with:")
        print("  - 110M+ parameters (BERT-base architecture)")
        print("  - GPU optimization with mixed precision training")
        print("  - Automatic batch size optimization")
        print("  - Advanced checkpointing and resuming")
        print("\nFor rigorous bias detection, use --balanced flag:")
        print("  python bert_trainer_production.py a --balanced")
        return
    
    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    
    # Path to our custom tokenizer
    custom_tokenizer_path = "../tokenizer/hansard-bpe-tokenizer"
    
    if not os.path.exists(f"{custom_tokenizer_path}/vocab.json"):
        print(f"Error: Custom tokenizer not found at {custom_tokenizer_path}")
        print("Please ensure the BPE tokenizer is trained and saved.")
        return
    
    # Check for balanced datasets if requested
    if args.balanced:
        balanced_file = f"../data/dataset_{dataset_name}_balanced_train.csv"
        if not os.path.exists(balanced_file):
            print(f"Error: Balanced dataset not found at {balanced_file}")
            print("Please create balanced datasets first:")
            print("  python create_balanced_datasets.py")
            return
    
    # Create production trainer and start training
    trainer = ProductionBERTTrainer(dataset_name, custom_tokenizer_path, use_balanced=args.balanced)
    
    dataset_type = "balanced" if args.balanced else "original"
    model_save_path = f"../models/model_{dataset_name}{'_balanced' if args.balanced else ''}/"
    
    print(f"\n🚀 Starting production BERT training for Dataset {dataset_name.upper()}")
    print(f"Dataset type: {dataset_type}")
    print(f"Expected training time: {'1-4 hours' if args.balanced else '2-12 hours'} depending on GPU")
    print(f"Model will be saved to: {model_save_path}")
    
    if args.balanced:
        print("\n📊 Using balanced datasets for rigorous bias detection:")
        print("  • Equal sample sizes eliminate data quantity confounds")
        print("  • Fair comparison isolates cultural bias effects")
        print("  • Methodologically sound experimental design")
    
    print("\nPress Ctrl+C to safely interrupt training...")
    
    summary = trainer.train_production_model(
        num_epochs=3,  # Can be adjusted based on convergence
        resume_from_checkpoint=args.resume_from
    )
    
    if summary:
        print(f"\n✅ Production training completed successfully!")
        print(f"Model: {summary['model_name']}")
        print(f"Dataset: {summary['dataset']}")
        print(f"Training time: {summary['training_time']}")
        print(f"Final loss: {summary['final_loss']:.4f}")
        print(f"Final perplexity: {summary['final_perplexity']:.2f}")
        print(f"Model size: {summary['parameters']:,} parameters ({summary['model_size_mb']:.1f}MB)")
        print(f"Device used: {summary['device']}")
        print(f"FP16 enabled: {summary['fp16_enabled']}")
        print(f"Model saved to: ../models/{summary['model_name']}/")
    else:
        print("❌ Production training failed!")

if __name__ == "__main__":
    main()