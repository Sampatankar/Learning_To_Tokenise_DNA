"""
Main Training Script for Genomic Language Models.

This script orchestrates the entire training pipeline:
1. Loads and processes data.
2. Creates and trains a tokenizer.
3. Instantiates a model.
4. Sets up and runs the Hugging Face Trainer.
"""
import os
import math
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Hugging Face imports
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast
)
# Tokenizers imports
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.trainers import WordLevelTrainer

# Local project imports
from data_processing.data_loader import load_and_process_data
from models.factory import create_model

# --- TEMPORARY: Hardcoded Configuration ---
# This section will be replaced by loading a YAML file in Step 4.
# This configuration is set for the non-overlapping k-mer experiment.
CONFIG = {
    'data': {
        'name': 'InstaDeepAI/human_reference_genome',
        'max_buffer_size': 10000, # Using a smaller buffer for faster testing
        'processing_strategy': 'kmer_non_overlapping',
        'kmer_size': 8
    },
    'tokenizer': {
        'type': 'kmer',
        'special_tokens': ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    },
    'model': {
        'type': 'bert',
        'name': 'bert-base-uncased'
    },
    'training': {
        'output_dir': './outputs/bert_non_overlapping_kmer',
        'logging_dir': './logs/bert_non_overlapping_kmer',
        'num_train_epochs': 1, # Reduced for a quick test run
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 4,
        'save_steps': 5000,
        'save_total_limit': 2,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'logging_steps': 100,
        'evaluation_strategy': "steps",
        'eval_steps': 500,
        'mlm_probability': 0.15
    }
}

# --- TEMPORARY: Tokenizer Creation (to be moved to src/tokenizers/factory.py) ---
def create_kmer_tokenizer(special_tokens):
    """Creates a shell for a k-mer (WordLevel) tokenizer."""
    base_tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    base_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    base_tokenizer.decoder = decoders.WordPiece()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer,
        unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]", model_max_length=512,
    )
    return hf_tokenizer

# --- Dataset Preparation Helper ---
def prepare_tokenized_dataset(texts, tokenizer):
    """Tokenizes a list of texts and prepares it for the Trainer."""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            max_length=512,
            padding='max_length',
            truncation=True
        )
    raw_dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = raw_dataset.map(
        tokenize_function, batched=True, num_proc=os.cpu_count(),
        remove_columns=["text"], desc="Tokenising dataset"
    )
    # Add indices for BPB calculation later
    tokenized_dataset = tokenized_dataset.add_column("indices", range(len(tokenized_dataset)))
    tokenized_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'indices'])
    return tokenized_dataset

# --- Custom Trainer for Evaluation ---
# Extracted from NO_perplexity.py. This can be moved to src/training/trainer.py later.
class CustomTrainer(Trainer):
    def __init__(self, *args, raw_eval_texts=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the raw DNA sequences for correct base counting during evaluation
        self.raw_eval_texts = raw_eval_texts

    def compute_loss(self, model, inputs, return_outputs=False):
        """ Overridden to prevent TypeError from the 'indices' column. """
        model_inputs = dict(inputs)
        model_inputs.pop("indices", None)
        return super().compute_loss(model, model_inputs, return_outputs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Calculates perplexity and bits per base (BPB)."""
        eval_ds = eval_dataset or self.eval_dataset
        
        # --- Part 1: Perplexity Calculation ---
        dl = self.get_eval_dataloader(eval_ds)
        self.model.eval()
        total_mlm_loss = 0
        total_masked_tokens = 0

        for batch in tqdm(dl, desc="Perplexity Calculation"):
            with torch.no_grad():
                batch = self._prepare_inputs(batch)
                labels = batch.get("labels")
                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits
                masked_indices = (labels != -100)
                if not torch.any(masked_indices): continue
                
                num_masked = masked_indices.sum().item()
                total_masked_tokens += num_masked
                masked_logits = logits[masked_indices]
                masked_labels = labels[masked_indices]
                mlm_loss = F.cross_entropy(masked_logits, masked_labels, reduction='sum')
                total_mlm_loss += mlm_loss.item()

        mean_mlm_loss = total_mlm_loss / max(1, total_masked_tokens)
        perplexity = math.exp(mean_mlm_loss)

        # --- Part 2: BPB Calculation ---
        k = CONFIG['data']['kmer_size']
        base_limit = (self.tokenizer.model_max_length - 2) * k
        base_count = sum(len(self.raw_eval_texts[i][:base_limit]) for i in eval_ds["indices"])
        
        # Count non-special tokens
        pad_collator = DataCollatorWithPadding(self.tokenizer)
        bpb_dl = DataLoader(eval_ds, batch_size=self.args.per_device_eval_batch_size, collate_fn=pad_collator)
        ids_to_exclude = {self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id}
        token_count = 0
        for batch in bpb_dl:
            is_not_special = torch.ones_like(batch["input_ids"], dtype=torch.bool)
            for token_id in ids_to_exclude:
                if token_id is not None: is_not_special &= (batch["input_ids"] != token_id)
            token_count += int((batch["attention_mask"].bool() & is_not_special).sum().item())

        bits_per_token = mean_mlm_loss / math.log(2)
        tokens_per_base = token_count / max(1, base_count)
        bits_per_base = bits_per_token * tokens_per_base

        metrics = {
            f"{metric_key_prefix}_loss": mean_mlm_loss,
            f"{metric_key_prefix}_perplexity": perplexity,
            f"{metric_key_prefix}_bits_per_base": bits_per_base,
        }
        self.log(metrics)
        print(f"\nEvaluation Metrics: {metrics}")
        return metrics

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Load and Process Data
    data = load_and_process_data(CONFIG)
    raw_texts = data['raw_texts']
    processed_texts = data['processed_texts']

    # 2. Create and Train Tokenizer
    print("\n--- Creating and Training Tokenizer ---")
    tokenizer = create_kmer_tokenizer(CONFIG['tokenizer']['special_tokens'])
    trainer = WordLevelTrainer(special_tokens=CONFIG['tokenizer']['special_tokens'], show_progress=True)
    tokenizer.backend_tokenizer.train_from_iterator(processed_texts, trainer=trainer)
    print(f"Tokenizer trained with a vocabulary size of {tokenizer.vocab_size:,}")

    # 3. Prepare Dataset for Model
    print("\n--- Preparing Tokenized Dataset ---")
    tokenized_dataset = prepare_tokenized_dataset(processed_texts, tokenizer)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    print(f"Training set size: {len(train_dataset):,}, Evaluation set size: {len(eval_dataset):,}")

    # 4. Create Model
    print("\n--- Creating Model ---")
    model = create_model(CONFIG, tokenizer)

    # 5. Set up Trainer
    print("\n--- Setting up Trainer ---")
    training_args = TrainingArguments(**CONFIG['training'])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=CONFIG['training']['mlm_probability']
    )
    
    # Get the raw text for the evaluation split for correct BPB calculation
    raw_eval_texts = [raw_texts[i] for i in eval_dataset["indices"]]

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        raw_eval_texts=raw_eval_texts
    )

    # 6. Start Training
    print("\n--- Starting Training ---")
    trainer.train()

    print("\n--- Training Complete ---")
    # Save the final model
    trainer.save_model(CONFIG['training']['output_dir'])
    tokenizer.save_pretrained(CONFIG['training']['output_dir'])
    print(f"Final model and tokenizer saved to {CONFIG['training']['output_dir']}")
