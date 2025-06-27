import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import glob
import os
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TokenizerTrainer:
    """
    Trains a custom Byte-Pair Encoding (BPE) tokenizer from scratch
    on the combined Hansard corpora.
    """

    def __init__(self, data_dir: str, output_dir: str):
        """
        Args:
            data_dir: Directory containing the final dataset CSVs.
            output_dir: Directory to save the trained tokenizer files.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.training_files = []

    def _get_training_files(self) -> List[str]:
        """Finds all training dataset files."""
        search_path = os.path.join(self.data_dir, 'dataset_*_train.csv')
        files = glob.glob(search_path)
        logger.info(f"Found {len(files)} training files: {files}")
        return files

    def _prepare_text_corpus(self, files: List[str], output_path: str):
        """
        Combines the 'speechtext' from multiple CSVs into a single text file.
        This single file is used to train the tokenizer efficiently.
        """
        logger.info(f"Combining text from {len(files)} files into a single corpus at {output_path}...")
        
        # Using a generator to handle potentially large data without high memory usage
        def text_generator():
            for file in files:
                try:
                    df = pd.read_csv(file)
                    if 'speechtext' in df.columns:
                        for text in df['speechtext'].dropna():
                            yield text
                    else:
                        logger.warning(f"'speechtext' column not found in {file}")
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for text_line in text_generator():
                f.write(text_line + "\n")
                count += 1
        
        logger.info(f"Wrote {count} lines of text to {output_path}")

    def train_tokenizer(self):
        """
        Orchestrates the tokenizer training process.
        """
        logger.info("Starting tokenizer training process...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Get all training data files
        training_files = self._get_training_files()
        if not training_files:
            logger.error("No training files found. Aborting tokenizer training.")
            logger.error(f"Please ensure files matching 'dataset_*_train.csv' exist in '{self.data_dir}'")
            return

        # 2. Combine into a single text file for training
        corpus_path = os.path.join(self.output_dir, "combined_training_corpus.txt")
        self._prepare_text_corpus(training_files, corpus_path)

        # 3. Initialize a BPE tokenizer
        # We use ByteLevelBPE for robustness with any text encoding.
        tokenizer = ByteLevelBPETokenizer()

        # 4. Train the tokenizer
        logger.info("Training BPE tokenizer from scratch...")
        tokenizer.train(
            files=[corpus_path],
            vocab_size=30_522,  # Standard vocab size, can be tuned
            min_frequency=2,    # A word must appear at least twice
            special_tokens=[   # Define special tokens the model will use
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ],
        )
        
        logger.info("Tokenizer training complete.")

        # 5. Save the tokenizer
        # This saves vocab.json and merges.txt
        tokenizer_path = os.path.join(self.output_dir, 'hansard-bpe-tokenizer')
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_model(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        # Optional: Clean up the large temporary corpus file
        os.remove(corpus_path)
        logger.info(f"Cleaned up temporary corpus file: {corpus_path}")

def main():
    """Main execution function."""
    # --- Robust Path Definition ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_directory = os.path.join(project_root, 'data')
    tokenizer_output_directory = os.path.join(project_root, 'tokenizer')
    
    trainer = TokenizerTrainer(data_dir=data_directory, output_dir=tokenizer_output_directory)
    trainer.train_tokenizer()

if __name__ == "__main__":
    main() 