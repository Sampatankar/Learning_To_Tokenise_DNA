"""
Model Factory for Genomic Language Models.

This module contains model definitions and a factory function to instantiate
the correct model architecture based on a configuration dictionary.
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, BertConfig, BertForMaskedLM

# --- BLT (Byte Latent Transformer) Model Architecture ---
# Extracted from se_perplexity.py

class BLTConfig(PretrainedConfig):
    """Configuration class for the BLT model."""
    model_type = "blt"

    def __init__(
        self,
        vocab_size=256,
        local_model_dim=256,
        local_encoder_layers=2,
        local_decoder_layers=4,
        latent_model_dim=768,
        latent_model_layers=12,
        num_attention_heads=8,
        intermediate_size=3072,
        entropy_threshold=0.35,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.local_model_dim = local_model_dim
        self.local_encoder_layers = local_encoder_layers
        self.local_decoder_layers = local_decoder_layers
        self.latent_model_dim = latent_model_dim
        self.latent_model_layers = latent_model_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.entropy_threshold = entropy_threshold

class TransformerBlock(nn.Module):
    """A standard Transformer block used within the BLT model."""
    def __init__(self, d_model, n_heads, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + attn_output)
        ff_output = self.feed_forward(src)
        src = self.norm2(src + ff_output)
        return src

class BLTModel(PreTrainedModel):
    """
    Implementation of the Byte Latent Transformer (BLT) model.
    This is a custom architecture and not a standard Hugging Face model.
    """
    config_class = BLTConfig
    def __init__(self, config: BLTConfig):
        super().__init__(config)
        self.config = config
        self.byte_embedder = nn.Embedding(config.vocab_size, config.local_model_dim)
        self.local_encoder = nn.ModuleList([TransformerBlock(config.local_model_dim, config.num_attention_heads, config.intermediate_size) for _ in range(config.local_encoder_layers)])
        self.to_latent = nn.Linear(config.local_model_dim, config.latent_model_dim)
        self.from_latent = nn.Linear(config.latent_model_dim, config.local_model_dim)
        self.latent_transformer = nn.ModuleList([TransformerBlock(config.latent_model_dim, config.num_attention_heads, config.intermediate_size) for _ in range(config.latent_model_layers)])
        self.local_decoder = nn.ModuleList([TransformerBlock(config.local_model_dim, config.num_attention_heads, config.intermediate_size) for _ in range(config.local_decoder_layers)])
        self.to_logits = nn.Linear(config.local_model_dim, config.vocab_size)

    def forward(self, input_ids, patch_indices, attention_mask=None, labels=None, output_hidden_states=False):
        # This forward pass is a simplified placeholder. The full implementation
        # from se_perplexity.py should be used for actual BLT training.
        byte_embeddings = self.byte_embedder(input_ids)
        encoded_bytes = byte_embeddings
        for layer in self.local_encoder: encoded_bytes = layer(encoded_bytes)
        
        # Placeholder for complex patching logic
        final_decoded_bytes = encoded_bytes 
        for layer in self.local_decoder: final_decoded_bytes = layer(final_decoded_bytes)
        
        logits = self.to_logits(final_decoded_bytes)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return {"logits": logits, "loss": loss}


# --- Model Factory Function ---

def create_model(config: dict, tokenizer):
    """
    Instantiates a model based on the provided configuration.

    Args:
        config (dict): The main configuration dictionary for the experiment.
                       Expects a 'model' key with 'type' and 'name' subkeys.
        tokenizer: The trained tokenizer instance.

    Returns:
        A PyTorch model instance (PreTrainedModel).
    """
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'bert').lower()
    
    print(f"Creating model of type: '{model_type}'")

    if model_type == 'bert':
        # For standard BERT models (k-mer, BPE, etc.)
        bert_base_model = model_config.get('name', 'bert-base-uncased')
        model_conf = BertConfig.from_pretrained(bert_base_model, vocab_size=tokenizer.vocab_size)
        model = BertForMaskedLM(config=model_conf)
    
    elif model_type == 'blt':
        # For the custom Byte Latent Transformer model
        blt_conf = BLTConfig(**model_config.get('params', {}))
        model = BLTModel(config=blt_conf)
    
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")
        
    print(f"Model initialised with {model.num_parameters():,} parameters.")
    return model