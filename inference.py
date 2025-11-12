"""
Inference Pipeline for Multimodal Art Generator
Generate stylized images + captions from random seeds
"""

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# TOKENIZER (Same as training)
# ================================================================================

class ProductionTokenizer:
    """Production-ready tokenizer for encoding/decoding captions"""
    
    def __init__(self, vocab_size: int = 8000):
        """Initialize tokenizer"""
        self.vocab_size = vocab_size
        self.word2idx = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.next_idx = 4
    
    def load(self, path: str):
        """Load tokenizer from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        self.vocab_size = data['vocab_size']
        
        logger.info(f"✓ Tokenizer loaded: {len(self.word2idx)} tokens")
    
    def decode(self, tokens: torch.Tensor, remove_special: bool = True) -> str:
        """
        Decode token tensor to text
        
        Args:
            tokens: (seq_len,) tensor of token IDs
            remove_special: Remove <START>, <END>, <PAD> tokens
        
        Returns:
            Decoded text string
        """
        words = []
        
        for token_id in tokens:
            token_id = int(token_id.item()) if torch.is_tensor(token_id) else int(token_id)
            
            if token_id == 0:  # <PAD>
                # Stop at padding
                break
            elif token_id == 1:  # <START>
                # Skip START token
                continue
            elif token_id == 2:  # <END>
                # Stop at END token
                break
            elif token_id == 3:  # <UNK>
                # Skip unknown tokens
                continue
            else:
                word = self.idx2word.get(token_id, '[UNK]')
                # Filter out special token names and unknown
                if not word.startswith('<') and word != '[UNK]':
                    words.append(word)
        
        # Join words and clean up
        text = ' '.join(words)
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # If empty after filtering, return a default caption
        if not text or text.strip() == '':
            text = "an engraving"
        
        return text


# ================================================================================
# MODEL LOADING (From training)
# ================================================================================

def load_model(
    checkpoint_path: str,
    latent_dim: int = 1024,
    vocab_size: int = 1266,
    image_channels: int = 3,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load trained multimodal model
    
    Args:
        checkpoint_path: Path to best.pt checkpoint
        latent_dim: Latent dimension (1024)
        vocab_size: Vocabulary size (1266)
        image_channels: Number of image channels (3 for RGB)
        device: Device to load on (cuda/cpu)
    
    Returns:
        Model in eval mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import model architecture
    from model_architecture_large import MultimodalModel
    
    # Create model
    model = MultimodalModel(
        latent_dim=latent_dim,
        image_channels=image_channels,
        vocab_size=vocab_size,
        max_caption_length=100
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Move to device and eval mode
    model = model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded from {checkpoint_path}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Latent dim: {latent_dim}")
    logger.info(f"  Vocab size: {vocab_size}")
    
    return model


# ================================================================================
# GENERATION FUNCTIONS
# ================================================================================

def decode_caption_with_sampling(
    caption_logits: torch.Tensor,
    tokenizer: ProductionTokenizer,
    temperature: float = 0.7,
    max_length: int = 30,
    repetition_penalty: float = 1.2
) -> str:
    """
    Decode caption with temperature sampling and repetition filtering
    
    Args:
        caption_logits: (1, 100, 1266) logits from model
        tokenizer: Caption tokenizer
        temperature: Sampling temperature (higher = more random)
        max_length: Maximum caption length
        repetition_penalty: Penalty for repeating tokens
    
    Returns:
        Decoded caption text
    """
    caption_logits = caption_logits[0]  # (100, 1266)
    
    words = []
    last_token_id = None
    
    for step in range(min(len(caption_logits), max_length)):
        logits = caption_logits[step].clone()  # (1266,)
        
        # Apply repetition penalty
        if last_token_id is not None:
            logits[last_token_id] = logits[last_token_id] / repetition_penalty
        
        # Apply temperature
        logits = logits / temperature
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sample from distribution
        token_id = torch.multinomial(probs, 1).item()
        
        # Handle special tokens
        if token_id == 0:  # <PAD> - stop
            break
        elif token_id == 2:  # <END> - stop
            break
        elif token_id == 1:  # <START> - skip
            continue
        elif token_id == 3:  # <UNK> - skip
            continue
        else:
            word = tokenizer.idx2word.get(token_id, '[UNK]')
            # Add word if it's not a special token
            if not word.startswith('<') and word != '[UNK]':
                words.append(word)
                last_token_id = token_id
    
    # Join and clean
    text = ' '.join(words)
    text = ' '.join(text.split())
    
    # Default caption if empty
    if not text or len(words) < 2:
        text = "an engraving"
    
    return text


def tensor_to_image(
    image_tensor: torch.Tensor,
    normalize: bool = True
) -> Image.Image:
    """
    Convert model output tensor to PIL Image
    
    Args:
        image_tensor: (3, H, W) tensor in [-1, 1] range
        normalize: Whether to normalize from [-1, 1] to [0, 1]
    
    Returns:
        PIL Image
    """
    # Detach and move to CPU
    if torch.is_tensor(image_tensor):
        image_tensor = image_tensor.detach().cpu()
    
    # Handle batch dimension if present
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    
    # Normalize if needed
    if normalize:
        image_tensor = (image_tensor + 1) / 2  # [-1, 1] → [0, 1]
    
    # Clamp to valid range
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    # Convert to uint8
    image_array = (image_tensor * 255).numpy().astype(np.uint8)
    
    # Permute to (H, W, C) if needed
    if image_array.shape[0] == 3:
        image_array = np.transpose(image_array, (1, 2, 0))
    
    # Create PIL Image
    image_pil = Image.fromarray(image_array, mode='RGB')
    
    return image_pil


def generate_multimodal(
    model: nn.Module,
    tokenizer: ProductionTokenizer,
    seed: int = 42,
    device: Optional[torch.device] = None,
    latent_dim: int = 1024,
    temperature: float = 0.7
) -> Tuple[Image.Image, str, Dict]:
    """
    Generate image + caption from a single seed
    
    Args:
        model: Trained multimodal model
        tokenizer: Caption tokenizer
        seed: Random seed for reproducibility
        device: Device to generate on
        latent_dim: Latent dimension (1024)
        temperature: Sampling temperature for captions (higher = more diverse)
    
    Returns:
        Tuple of (PIL_Image, caption_text, metadata_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Generating from seed: {seed}")
    logger.info(f"{'='*70}")
    
    with torch.no_grad():
        # Generate latent vector from seed
        z = torch.randn(1, latent_dim, device=device)
        logger.info(f"✓ Generated latent vector: {z.shape}")
        
        # Generate image from latent
        image_tensor = model.decode_image(z)
        logger.info(f"✓ Generated image: {image_tensor.shape}")
        
        # Convert to PIL Image
        image_pil = tensor_to_image(image_tensor)
        logger.info(f"✓ Converted to PIL Image: {image_pil.size}")
        
        # Generate caption logits
        caption_logits = model.decode_text(z)
        logger.info(f"✓ Generated caption logits: {caption_logits.shape}")
        
        # Decode caption with sampling (better than greedy argmax)
        caption_text = decode_caption_with_sampling(
            caption_logits,
            tokenizer,
            temperature=temperature,
            max_length=30,
            repetition_penalty=1.2
        )
        logger.info(f"✓ Decoded caption:\n  '{caption_text}'")
    
    # Metadata
    metadata = {
        'seed': seed,
        'caption': caption_text,
        'latent_dim': latent_dim,
        'temperature': temperature,
        'image_size': image_pil.size,
        'device': str(device)
    }
    
    return image_pil, caption_text, metadata


def generate_batch(
    model: nn.Module,
    tokenizer: ProductionTokenizer,
    seeds: list,
    device: Optional[torch.device] = None,
    latent_dim: int = 1024
) -> list:
    """
    Generate multiple image+caption pairs
    
    Args:
        model: Trained multimodal model
        tokenizer: Caption tokenizer
        seeds: List of seed values
        device: Device to generate on
        latent_dim: Latent dimension
    
    Returns:
        List of (image, caption, metadata) tuples
    """
    results = []
    
    for i, seed in enumerate(seeds, 1):
        logger.info(f"\n[{i}/{len(seeds)}] Processing seed {seed}...")
        image, caption, metadata = generate_multimodal(
            model, tokenizer, seed, device, latent_dim
        )
        results.append((image, caption, metadata))
    
    return results


# ================================================================================
# INFERENCE INTERFACE
# ================================================================================

class MultimodalGenerator:
    """
    Easy-to-use interface for generating images + captions
    """
    
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best.pt",
        tokenizer_path: str = "checkpoints/tokenizer.json",
        latent_dim: int = 1024,
        vocab_size: int = 1266,
        device: Optional[str] = None
    ):
        """
        Initialize generator
        
        Args:
            checkpoint_path: Path to model weights
            tokenizer_path: Path to tokenizer
            latent_dim: Latent dimension
            vocab_size: Vocabulary size
            device: 'cuda', 'cpu', or None (auto)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = load_model(
            checkpoint_path,
            latent_dim=latent_dim,
            vocab_size=vocab_size,
            device=self.device
        )
        
        # Load tokenizer
        self.tokenizer = ProductionTokenizer(vocab_size)
        self.tokenizer.load(tokenizer_path)
        
        self.latent_dim = latent_dim
        
        logger.info("✓ Generator initialized successfully!\n")
    
    def generate(self, seed: int = 42) -> Tuple[Image.Image, str]:
        """
        Generate image + caption from seed
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (PIL_Image, caption_text)
        """
        image, caption, _ = generate_multimodal(
            self.model,
            self.tokenizer,
            seed,
            self.device,
            self.latent_dim
        )
        return image, caption
    
    def generate_with_metadata(self, seed: int = 42) -> Tuple[Image.Image, str, Dict]:
        """
        Generate image + caption + metadata from seed
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (PIL_Image, caption_text, metadata_dict)
        """
        return generate_multimodal(
            self.model,
            self.tokenizer,
            seed,
            self.device,
            self.latent_dim
        )
    
    def generate_batch(self, seeds: list) -> list:
        """
        Generate multiple images + captions
        
        Args:
            seeds: List of seed values
        
        Returns:
            List of (image, caption, metadata) tuples
        """
        return generate_batch(
            self.model,
            self.tokenizer,
            seeds,
            self.device,
            self.latent_dim
        )


# ================================================================================
# MAIN / DEMO
# ================================================================================

if __name__ == "__main__":
    
    # Initialize generator
    logger.info("Initializing Multimodal Generator...\n")
    
    generator = MultimodalGenerator(
        checkpoint_path="checkpoints/best.pt",
        tokenizer_path="checkpoints/tokenizer.json",
        latent_dim=1024,
        vocab_size=1266
    )
    
    # Test single generation
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Single Image + Caption Generation")
    logger.info("="*70)
    
    image, caption = generator.generate(seed=42)
    logger.info(f"\nResult:")
    logger.info(f"  Image size: {image.size}")
    logger.info(f"  Caption: {caption}")
    
    # Save result
    output_dir = Path("inference_outputs")
    output_dir.mkdir(exist_ok=True)
    image.save(output_dir / "sample_seed_42.png")
    logger.info(f"\n✓ Saved to: {output_dir / 'sample_seed_42.png'}")
    
    # Test batch generation
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Batch Generation (5 seeds)")
    logger.info("="*70)
    
    results = generator.generate_batch(seeds=[0, 1, 2, 3, 4])
    
    logger.info(f"\n✓ Generated {len(results)} samples")
    
    # Save batch results
    batch_metadata = []
    for i, (image, caption, metadata) in enumerate(results):
        image.save(output_dir / f"sample_seed_{metadata['seed']}.png")
        batch_metadata.append(metadata)
    
    # Save metadata
    with open(output_dir / "batch_metadata.json", "w") as f:
        json.dump(batch_metadata, f, indent=2)
    
    logger.info(f"✓ Saved {len(results)} samples and metadata")
    logger.info(f"\nAll outputs saved to: {output_dir}/")
    
    # Print sample captions
    logger.info("\n" + "="*70)
    logger.info("Generated Captions:")
    logger.info("="*70)
    for i, (image, caption, metadata) in enumerate(results):
        logger.info(f"\nSeed {metadata['seed']}: {caption}")
