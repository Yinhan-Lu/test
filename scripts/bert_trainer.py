import os
import pandas as pd
import torch
import logging
import json
from datetime import datetime
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HansardDataset(Dataset):
    """
    Dataset class for Hansard speeches optimized for BERT MLM training.
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
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
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

class BERTTrainer:
    """
    Custom BERT trainer for parliamentary language models.
    
    This trainer:
    1. Uses our custom BPE tokenizer (30,522 vocab)
    2. Trains BERT-base with Masked Language Modeling (MLM)
    3. Optimized for CPU training with memory management
    4. Tracks training metrics and saves checkpoints
    """
    
    def __init__(self, dataset_name: str, custom_tokenizer_path: str):
        self.dataset_name = dataset_name.upper()
        self.model_name = f"model_{dataset_name.lower()}"
        self.custom_tokenizer_path = custom_tokenizer_path
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Initialize paths
        self.model_dir = f"../models/{self.model_name}"
        self.log_dir = f"../logs/{self.model_name}"
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(f"Initializing BERT trainer for {self.dataset_name}")
        
    def load_custom_tokenizer(self):
        """Load our trained BPE tokenizer directly."""
        logger.info("Loading custom BPE tokenizer...")
        
        # Load the trained BPE tokenizer
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
        """Load dataset and return list of speech texts."""
        logger.info(f"Loading dataset {self.dataset_name} - {split} split...")
        
        dataset_file = f"../data/dataset_{self.dataset_name.lower()}_{split}.csv"
        
        try:
            df = pd.read_csv(dataset_file)
            texts = df['speechtext'].astype(str).tolist()
            
            # Filter out very short texts (less than 20 characters)
            texts = [text for text in texts if len(text.strip()) >= 20]
            
            logger.info(f"Loaded {len(texts)} speeches from {split} split")
            return texts
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    
    def create_model_config(self, tokenizer):
        """Create BERT model configuration with our custom vocabulary."""
        vocab_size = tokenizer.get_vocab_size()
        
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=768,           # BERT-base size
            num_hidden_layers=12,      # BERT-base layers
            num_attention_heads=12,    # BERT-base heads
            intermediate_size=3072,    # BERT-base FFN size
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
        
        logger.info(f"Created BERT config with vocab_size={config.vocab_size}")
        return config
    
    def train_model(self, num_epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """
        Train BERT model with Masked Language Modeling.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size (reduced for CPU)
            learning_rate: Learning rate with warmup
        """
        logger.info(f"Starting BERT training for {self.dataset_name}")
        
        # Load tokenizer and datasets
        tokenizer = self.load_custom_tokenizer()
        
        train_texts = self.load_dataset('train')
        val_texts = self.load_dataset('val')
        
        if not train_texts or not val_texts:
            logger.error("Failed to load datasets!")
            return
        
        # Create datasets
        train_dataset = HansardDataset(train_texts, tokenizer, max_length=512)
        val_dataset = HansardDataset(val_texts, tokenizer, max_length=512)
        
        logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Create model
        config = self.create_model_config(tokenizer)
        model = BertForMaskedLM(config)
        
        logger.info(f"Created BERT model with {model.num_parameters():,} parameters")
        
        # Custom data collator for MLM since we're using BPE tokenizer directly
        def data_collator(examples):
            batch = {}
            batch['input_ids'] = torch.stack([example['input_ids'] for example in examples])
            batch['attention_mask'] = torch.stack([example['attention_mask'] for example in examples])
            
            # Create labels for MLM (clone input_ids)
            batch['labels'] = batch['input_ids'].clone()
            
            # Mask 15% of tokens randomly for MLM
            probability_matrix = torch.full(batch['labels'].shape, 0.15)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            # Don't mask special tokens
            special_tokens_mask = (batch['input_ids'] == 1) | (batch['input_ids'] == 0) | (batch['input_ids'] == 2)  # pad, cls, sep
            masked_indices = masked_indices & ~special_tokens_mask
            
            # Replace masked tokens with [MASK] token (ID 4)
            batch['input_ids'][masked_indices] = 4  # <mask> token
            
            # Only compute loss on masked tokens
            batch['labels'][~masked_indices] = -100
            
            return batch
        
        # Training arguments optimized for CPU
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=1000,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=self.log_dir,
            logging_steps=500,
            eval_steps=2000,
            save_steps=2000,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=0,  # CPU optimization
            fp16=False,  # Disable for CPU
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
            seed=42
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )
        
        # Start training
        logger.info("Starting training...")
        start_time = datetime.now()
        
        train_result = trainer.train()
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        logger.info(f"Training completed in {training_time}")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Save final model and tokenizer
        logger.info("Saving model...")
        trainer.save_model()
        
        # Save our custom tokenizer
        tokenizer.save(os.path.join(self.model_dir, "vocab.json"), os.path.join(self.model_dir, "merges.txt"))
        
        # Save training summary
        training_summary = {
            'model_name': self.model_name,
            'dataset': self.dataset_name,
            'training_time': str(training_time),
            'final_loss': train_result.training_loss,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'vocab_size': tokenizer.get_vocab_size(),
            'max_length': 512,
            'total_steps': train_result.global_step,
            'training_completed': end_time.isoformat()
        }
        
        summary_path = os.path.join(self.model_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_path}")
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_result = trainer.evaluate()
        logger.info(f"Final evaluation loss: {eval_result['eval_loss']:.4f}")
        logger.info(f"Final perplexity: {np.exp(eval_result['eval_loss']):.2f}")
        
        return training_summary

def main():
    """Main training function - allows training specific models."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python bert_trainer.py <dataset_name>")
        print("Example: python bert_trainer.py a")
        print("Options: a (Nunavut), b (Canadian), c (Mixed)")
        return
    
    dataset_name = sys.argv[1].lower()
    
    if dataset_name not in ['a', 'b', 'c']:
        print("Error: dataset_name must be 'a', 'b', or 'c'")
        return
    
    # Path to our custom tokenizer
    custom_tokenizer_path = "../tokenizer/hansard-bpe-tokenizer"
    
    # Create trainer and start training
    trainer = BERTTrainer(dataset_name, custom_tokenizer_path)
    
    # Train with optimized parameters for CPU
    summary = trainer.train_model(
        num_epochs=3,
        batch_size=8,  # Reduced for CPU training
        learning_rate=2e-5
    )
    
    if summary:
        print(f"\n✅ Training completed successfully!")
        print(f"Model: {summary['model_name']}")
        print(f"Dataset: {summary['dataset']}")
        print(f"Training time: {summary['training_time']}")
        print(f"Final loss: {summary['final_loss']:.4f}")
        print(f"Model saved to: ../models/{summary['model_name']}/")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()