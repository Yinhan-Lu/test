import os
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
    """Fast dataset for demo training."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):  # Reduced max length
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

class BERTDemoTrainer:
    """
    Demo BERT trainer with smaller model for faster training.
    This demonstrates the full pipeline but with reduced complexity.
    """
    
    def __init__(self, dataset_name: str, custom_tokenizer_path: str):
        self.dataset_name = dataset_name.upper()
        self.model_name = f"demo_model_{dataset_name.lower()}"
        self.custom_tokenizer_path = custom_tokenizer_path
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Initialize paths
        self.model_dir = f"../models/{self.model_name}"
        self.log_dir = f"../logs/{self.model_name}"
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(f"Initializing demo BERT trainer for {self.dataset_name}")
        
    def load_custom_tokenizer(self):
        """Load our trained BPE tokenizer."""
        logger.info("Loading custom BPE tokenizer...")
        
        tokenizer = ByteLevelBPETokenizer.from_file(
            vocab_filename=f"{self.custom_tokenizer_path}/vocab.json",
            merges_filename=f"{self.custom_tokenizer_path}/merges.txt"
        )
        
        tokenizer.add_special_tokens([
            "<s>", "<pad>", "</s>", "<unk>", "<mask>"
        ])
        
        logger.info(f"Custom tokenizer loaded with vocab size: {tokenizer.get_vocab_size()}")
        return tokenizer
    
    def load_dataset(self, split: str = 'train', sample_size: int = 1000) -> List[str]:
        """Load a sample of the dataset for demo purposes."""
        logger.info(f"Loading sample dataset {self.dataset_name} - {split} split...")
        
        dataset_file = f"../data/dataset_{self.dataset_name.lower()}_{split}.csv"
        
        try:
            df = pd.read_csv(dataset_file)
            texts = df['speechtext'].astype(str).tolist()
            
            # Filter out very short texts
            texts = [text for text in texts if len(text.strip()) >= 20]
            
            # Take only a sample for demo
            if len(texts) > sample_size:
                texts = texts[:sample_size]
            
            logger.info(f"Loaded {len(texts)} speeches from {split} split (demo sample)")
            return texts
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    
    def create_demo_model_config(self, tokenizer):
        """Create a smaller BERT model configuration for demo."""
        vocab_size = tokenizer.get_vocab_size()
        
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=384,           # Smaller than BERT-base (768)
            num_hidden_layers=6,       # Fewer layers than BERT-base (12)
            num_attention_heads=6,     # Fewer heads than BERT-base (12)
            intermediate_size=1536,    # Smaller FFN than BERT-base (3072)
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=256,  # Shorter sequences
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,  # <pad> token ID
            position_embedding_type="absolute"
        )
        
        logger.info(f"Created demo BERT config with vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
        return config
    
    def train_demo_model(self):
        """Train a demo model to show the complete pipeline."""
        logger.info(f"Starting demo BERT training for {self.dataset_name}")
        
        # Load tokenizer and datasets (smaller samples)
        tokenizer = self.load_custom_tokenizer()
        
        train_texts = self.load_dataset('train', sample_size=500)  # Small sample
        val_texts = self.load_dataset('val', sample_size=100)     # Small sample
        
        if not train_texts or not val_texts:
            logger.error("Failed to load datasets!")
            return None
        
        # Create datasets with shorter max length
        train_dataset = HansardDataset(train_texts, tokenizer, max_length=256)
        val_dataset = HansardDataset(val_texts, tokenizer, max_length=256)
        
        logger.info(f"Created demo datasets: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Create smaller model
        config = self.create_demo_model_config(tokenizer)
        model = BertForMaskedLM(config)
        
        logger.info(f"Created demo BERT model with {model.num_parameters():,} parameters")
        
        # Custom data collator for MLM
        def data_collator(examples):
            batch = {}
            batch['input_ids'] = torch.stack([example['input_ids'] for example in examples])
            batch['attention_mask'] = torch.stack([example['attention_mask'] for example in examples])
            
            # Create labels for MLM
            batch['labels'] = batch['input_ids'].clone()
            
            # Mask 15% of tokens randomly for MLM
            probability_matrix = torch.full(batch['labels'].shape, 0.15)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            # Don't mask special tokens
            special_tokens_mask = (batch['input_ids'] == 1) | (batch['input_ids'] == 0) | (batch['input_ids'] == 2)
            masked_indices = masked_indices & ~special_tokens_mask
            
            # Replace masked tokens with [MASK] token (ID 4)
            batch['input_ids'][masked_indices] = 4  # <mask> token
            
            # Only compute loss on masked tokens
            batch['labels'][~masked_indices] = -100
            
            return batch
        
        # Demo training arguments (very fast)
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,  # Just 1 epoch for demo
            per_device_train_batch_size=4,  # Small batch
            per_device_eval_batch_size=4,
            warmup_steps=50,
            learning_rate=5e-5,  # Higher learning rate for faster convergence
            weight_decay=0.01,
            logging_dir=self.log_dir,
            logging_steps=20,    # Frequent logging
            eval_steps=100,
            save_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=0,
            fp16=False,
            remove_unused_columns=False,
            report_to=None,
            seed=42
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Start training
        logger.info("Starting demo training...")
        start_time = datetime.now()
        
        train_result = trainer.train()
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        logger.info(f"Demo training completed in {training_time}")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Save model
        logger.info("Saving demo model...")
        trainer.save_model()
        
        # Save tokenizer files
        tokenizer.save_model(self.model_dir)
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_result = trainer.evaluate()
        logger.info(f"Final evaluation loss: {eval_result['eval_loss']:.4f}")
        logger.info(f"Final perplexity: {np.exp(eval_result['eval_loss']):.2f}")
        
        # Save training summary
        training_summary = {
            'model_name': self.model_name,
            'dataset': self.dataset_name,
            'model_type': 'demo',
            'training_time': str(training_time),
            'final_loss': train_result.training_loss,
            'final_eval_loss': eval_result['eval_loss'],
            'final_perplexity': np.exp(eval_result['eval_loss']),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'vocab_size': tokenizer.get_vocab_size(),
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'parameters': model.num_parameters(),
            'training_completed': end_time.isoformat()
        }
        
        summary_path = os.path.join(self.model_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_path}")
        
        return training_summary

def main():
    """Main demo training function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python bert_trainer_demo.py <dataset_name>")
        print("Example: python bert_trainer_demo.py a")
        print("Options: a (Nunavut), b (Canadian), c (Mixed)")
        return
    
    dataset_name = sys.argv[1].lower()
    
    if dataset_name not in ['a', 'b', 'c']:
        print("Error: dataset_name must be 'a', 'b', or 'c'")
        return
    
    # Path to our custom tokenizer
    custom_tokenizer_path = "../tokenizer/hansard-bpe-tokenizer"
    
    # Create demo trainer and start training
    trainer = BERTDemoTrainer(dataset_name, custom_tokenizer_path)
    
    summary = trainer.train_demo_model()
    
    if summary:
        print(f"\n✅ Demo training completed successfully!")
        print(f"Model: {summary['model_name']}")
        print(f"Dataset: {summary['dataset']}")
        print(f"Training time: {summary['training_time']}")
        print(f"Final loss: {summary['final_loss']:.4f}")
        print(f"Final perplexity: {summary['final_perplexity']:.2f}")
        print(f"Model size: {summary['parameters']:,} parameters")
        print(f"Model saved to: ../models/{summary['model_name']}/")
    else:
        print("❌ Demo training failed!")

if __name__ == "__main__":
    main()