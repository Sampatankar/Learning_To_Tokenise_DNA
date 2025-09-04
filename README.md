# Introduction

This codebase is Github repo of my tokeniser adapted BERT (HuggingFace bert-base-uncased) architectures used to build model weights and input embeddings to be fed into the downstream Github sub-repo containing the BEND fork from the BaillieLab (https://github.com/baillielab/BEND), which is itself a fork from the original work by Marin et al. (https://bend.readthedocs.io/en/latest/?badge=latest).


## What's in the Repo:
    
### Python Packages
- In order to train and run the models adapted for tokenisers that outputs embeddings that can be evaluated by the BEND evaluation dataset, we have a requirements.txt that uses older versions of the following python modules:
    - datasets==2.14.6
    - pyarrow==12.0.1
    - transformers==4.33.3
    - tokenizers==0.13.3
    - accelerate==0.20.3

### Docker images
- We use docker images to specify the correct Python versions for building and training the BERT models (Python 3.12.2) and the BEND evaluation tests (Python 3.10.8)

### Handling Genomic Data Download:

#### src/data_processing/data_loader.py:
- Loads and preprocesses genomic data based on a configuration dictionary.
This function streams the InstaDeepAI Human Reference Genome dataset from HuggingFace, collects sequences into a buffer, and then processes them based on the specified strategy.
- InstaDeepAI/human_reference_genome

- Args:
    config (dict): A configuration dictionary expecting the following keys:
    - data:
        - name: The name of the Hugging Face dataset.
        - max_buffer_size: Max number of sequences to load into RAM.
        - processing_strategy: The method to use for processing.
            Options: 'raw', 'char_spaced', 'kmer_overlapping',
                    'kmer_non_overlapping', 'hybrid'.
        - kmer_size (optional): The size of k for k-mer strategies.

    - Returns:
        - A dictionary containing two keys:
        - 'raw_texts': A list of the original, unprocessed DNA sequences.
        - 'processed_texts': A list of sequences processed according to the specified strategy. 
        - For 'raw' and 'char_spaced',this is derived directly from the raw text. For k-mer strategies, this contains the space-separated k-mers.



## How do you run this code:
1) Create a venv environment to change the module versions to the required ones:
 - python -m venv name_of_environment
 - source name_of_environment/bin/activate
 These commands are for the MacOS (search for the adapted commands to initialise your Python environments for your )

2) 



