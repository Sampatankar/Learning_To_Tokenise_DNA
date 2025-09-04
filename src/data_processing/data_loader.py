"""
Module to handle a single one pass data loader to be used for each model trained.

This module loads sequences from a specified Hugging Face dataset and processes
them according to a specified strategy (e.g., raw text, k-mers).
"""

import os
from typing import Dict, List
from datasets import load_dataset
from tqdm import tqdm

def load_and_process_data(config: dict) -> Dict[str, List[str]]:
    """
    Loads and preprocesses genomic data based on a configuration dictionary.

    """
    # Load configuration
    data_config = config.get('data', {})
    dataset_name = data_config.get('name', 'InstaDeepAI/human_reference_genome')
    max_buffer_size = data_config.get('max_buffer_size', 20_000)
    strategy = data_config.get('processing_strategy', 'raw')
    k = data_config.get('kmer_size', 6)  # Default k-mer size

    # Stream and collect raw sequences from Hugging Face
    print(f"Loading dataset: {dataset_name} (streaming)")
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    raw_texts: List[str] = []
    iterator = iter(dataset)
    for record in tqdm(iterator, desc=f"Streaming up to {max_buffer_size:,} sequences"):
        if len(raw_texts) >= max_buffer_size:
            print(f"\nReached RAM buffer limit of {max_buffer_size:,} sequences.")
            break
        seq = record.get("seq") or record.get("sequence")
        if seq:
            raw_texts.append(seq)
    print(f"Total sequences collected in buffer: {len(raw_texts):,}")

    # Process sequences based on the selected strategy
    processed_texts: List[str] = []
    print(f"Processing texts with strategy: '{strategy}'")

    if strategy == 'raw':
        # Used for BPE and Byte-Level (BLT) tokenisers
        processed_texts = raw_texts
    
    elif strategy == 'char_spaced':
        # Used for the single-character WordLevel tokeniser
        processed_texts = [" ".join(list(seq)) for seq in raw_texts]

    elif strategy == 'kmer_overlapping':
        # Used for overlapping k-mer model
        for seq in raw_texts:
            if len(seq) >= k:
                kmers = " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])
                processed_texts.append(kmers)

    elif strategy == 'kmer_non_overlapping':
        # Used for non-overlapping k-mer model
        for seq in raw_texts:
            if len(seq) >= k:
                kmers = " ".join([seq[i:i+k] for i in range(0, len(seq) - (len(seq) % k), k)])
                processed_texts.append(kmers)

    elif strategy == 'hybrid':
        # Special case for the hybrid tokeniser
        # 'processed_texts' will be k-mers for training the k-mer part of the vocab
        # 'raw_texts' will have a leading space for the Metaspace BPE part
        print(f"Generating {k}-mers for hybrid tokeniser training...")
        for seq in raw_texts:
            if len(seq) >= k:
                kmers = " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])
                processed_texts.append(kmers)
        # Modify raw_texts in-place for the BPE component
        raw_texts = [" " + seq for seq in raw_texts]

    else:
        raise ValueError(f"Unknown processing strategy: '{strategy}'")

    print("Processing complete.")
    return {'raw_texts': raw_texts, 'processed_texts': processed_texts}


if __name__ == '__main__':
    # --- Example of how to use this data loader ---
    print("--- Testing Data Loader ---")

    # 1. BPE / BLT Config (raw text)
    bpe_config = {
        'data': {
            'max_buffer_size': 100,
            'processing_strategy': 'raw'
        }
    }
    print("\n1. Testing 'raw' strategy for BPE/BLT...")
    bpe_data = load_and_process_data(bpe_config)
    print(f"   Loaded {len(bpe_data['raw_texts'])} raw sequences.")
    print(f"   First sequence (first 50 chars): '{bpe_data['raw_texts'][0][:50]}...'")

    # 2. Overlapping K-mer Config
    kmer_config = {
        'data': {
            'max_buffer_size': 100,
            'processing_strategy': 'kmer_overlapping',
            'kmer_size': 6
        }
    }
    print("\n2. Testing 'kmer_overlapping' strategy...")
    kmer_data = load_and_process_data(kmer_config)
    print(f"   Loaded {len(kmer_data['processed_texts'])} k-mer sequences.")
    print(f"   First k-mer sequence (first 50 chars): '{kmer_data['processed_texts'][0][:50]}...'")

    # 3. Hybrid Config
    hybrid_config = {
        'data': {
            'max_buffer_size': 100,
            'processing_strategy': 'hybrid',
            'kmer_size': 6
        }
    }
    print("\n3. Testing 'hybrid' strategy...")
    hybrid_data = load_and_process_data(hybrid_config)
    print(f"   Processed {len(hybrid_data['processed_texts'])} k-mer sequences for hybrid vocab.")
    print(f"   Processed {len(hybrid_data['raw_texts'])} raw sequences for hybrid vocab.")
    print(f"   First raw sequence for BPE part (first 50 chars): '{hybrid_data['raw_texts'][0][:50]}...'")
    print(f"   First k-mer sequence for k-mer part (first 50 chars): '{hybrid_data['processed_texts'][0][:50]}...'")