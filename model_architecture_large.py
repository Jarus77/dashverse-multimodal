"""
Data Loader & Model Architecture (LARGE MODEL)
Multimodal AI Project: Latent Diffusion + Text Decoder

OPTIMIZED PARAMETERS:
- Latent dimension: 1024 (2x original)
- Text embedding: 512 (2x original)  
- Vocabulary: 8000 (optimized - analysis driven)
- Transformer heads: 8
- Encoder: 5 conv layers with BatchNorm

Components:
1. EngravingDataset - PyTorch dataset from metadata
2. MultimodalModel - Unified architecture
   - Image encoder: Enhanced CNN with BatchNorm
   - Image decoder: Upsampling decoder
   - Text decoder: Transformer
   - Shared latent space (1024-dim)
3. Training utilities
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# DATASET & DATA LOADING
# ==============================================================================

class EngravingDataset(Dataset):
    """PyTorch Dataset for engravings with captions"""
    
    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset
        
        Args:
            metadata_path: Path to metadata JSON file (with captions)
            images_dir: Directory containing resized images
            split: "train" or "val"
            train_ratio: Fraction for training
            seed: Random seed
            transform: Optional image transforms
            max_samples: Limit dataset size (for testing)
        """
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir)
        self.split = split
        self.transform = transform
        self.seed = seed
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        # Filter entries with captions
        valid_entries = [e for e in all_metadata if e.get("caption")]
        logger.info(f"Loaded {len(valid_entries)} entries with captions")
        
        # Split data
        np.random.seed(seed)
        indices = np.random.permutation(len(valid_entries))
        split_point = int(len(valid_entries) * train_ratio)
        
        if split == "train":
            self.entries = [valid_entries[i] for i in indices[:split_point]]
        else:  # val
            self.entries = [valid_entries[i] for i in indices[split_point:]]
        
        # Limit samples if specified
        if max_samples and len(self.entries) > max_samples:
            self.entries = self.entries[:max_samples]
        
        logger.info(f"Dataset ({split}): {len(self.entries)} samples")
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        """
        Return sample with image and caption
        
        Returns:
            Dict with keys: 'image', 'caption', 'filename'
        """
        entry = self.entries[idx]
        
        # Load image
        img_path = self.images_dir / entry['filename']
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor [0, 1]
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)  # HWC -> CHW
        
        # Get caption
        caption = entry['caption']
        
        return {
            'image': image,
            'caption': caption,
            'filename': entry['filename'],
            'id': entry['id']
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batch dict with stacked tensors
    """
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    filenames = [item['filename'] for item in batch]
    ids = torch.tensor([item['id'] for item in batch])
    
    return {
        'images': images,  # (B, 3, 512, 512)
        'captions': captions,  # List of strings
        'filenames': filenames,
        'ids': ids
    }


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class SimpleTokenizer:
    """Simple tokenizer for captions"""
    
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.next_idx = 4
    
    def build_vocab(self, captions: List[str], min_freq=1):
        """Build vocabulary from captions"""
        from collections import Counter
        
        word_freq = Counter()
        for caption in captions:
            words = caption.lower().split()
            word_freq.update(words)
        
        for word, freq in word_freq.most_common(self.vocab_size - 4):
            if freq >= min_freq:
                self.word2idx[word] = self.next_idx
                self.idx2word[self.next_idx] = word
                self.next_idx += 1
        
        logger.info(f"Built vocabulary with {len(self.word2idx)} tokens")
    
    def encode(self, caption: str, max_length=100) -> torch.Tensor:
        """Encode caption to tensor"""
        tokens = [1]  # <START>
        for word in caption.lower().split():
            word = word.strip('.,!?;:')
            idx = self.word2idx.get(word, 3)  # <UNK> if not found
            tokens.append(idx)
        tokens.append(2)  # <END>
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode tensor back to caption"""
        words = []
        for idx in tokens:
            idx = idx.item()
            if idx in [0, 1, 2]:  # Skip padding, start, end
                continue
            word = self.idx2word.get(idx, '<UNK>')
            words.append(word)
        return ' '.join(words)


class ImageEncoder(nn.Module):
    """Encode images to latent space (LARGE MODEL)"""
    
    def __init__(self, latent_dim=1024, image_channels=3):
        """
        Enhanced CNN encoder with BatchNorm for 1024-dim latent
        
        Args:
            latent_dim: Dimension of latent representation (1024)
            image_channels: Number of input channels (3 for RGB)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # 5-layer encoder with BatchNorm for stability
        self.encoder = nn.Sequential(
            # Layer 1: 512x512 -> 256x256
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 2: 256x256 -> 128x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 3: 128x128 -> 64x64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4: 64x64 -> 32x32
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Layer 5: 32x32 -> 16x16
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Project to latent space
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, x):
        """
        Encode image to latent vector
        
        Args:
            x: Image tensor (B, 3, 512, 512)
            
        Returns:
            Latent vector (B, 1024)
        """
        features = self.encoder(x)  # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        latent = self.fc(features)  # (B, 1024)
        return latent


class ImageDecoder(nn.Module):
    """Decode latent space to images (LARGE MODEL)"""
    
    def __init__(self, latent_dim=1024, image_channels=3):
        """
        Enhanced upsampling decoder for 1024-dim latent
        
        Args:
            latent_dim: Dimension of latent representation (1024)
            image_channels: Number of output channels (3 for RGB)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Project latent to spatial features
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512 * 16 * 16)  # Start from 16x16
        )
        
        # Upsampling layers: 16x16 -> 512x512
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 256x256 -> 512x512
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        """
        Decode latent to image
        
        Args:
            z: Latent vector (B, 1024)
            
        Returns:
            Reconstructed image (B, 3, 512, 512) in [-1, 1]
        """
        features = self.fc(z)  # (B, 512*16*16)
        features = features.view(-1, 512, 16, 16)  # (B, 512, 16, 16)
        image = self.decoder(features)  # (B, 3, 512, 512)
        return image


class TextDecoder(nn.Module):
    """Decode latent space to caption tokens (LARGE MODEL)"""
    
    def __init__(self, latent_dim=1024, vocab_size=8000, max_length=100, embedding_dim=512):
        """
        Transformer-based caption decoder (large)
        
        Args:
            latent_dim: Dimension of latent representation (1024)
            vocab_size: Size of vocabulary (8000)
            max_length: Maximum caption length (100)
            embedding_dim: Embedding dimension (512)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
        # Latent to initial hidden state
        self.fc_latent = nn.Sequential(
            nn.Linear(latent_dim, 768),
            nn.ReLU(),
            nn.Linear(768, embedding_dim)
        )
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Embedding(max_length, embedding_dim)
        
        # Transformer decoder (larger)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, z, caption_tokens=None):
        """
        Decode latent to caption
        
        Args:
            z: Latent vector (B, 1024)
            caption_tokens: Token indices for teacher forcing (B, max_length)
            
        Returns:
            Logits (B, max_length, vocab_size)
        """
        batch_size = z.size(0)
        
        # Convert latent to initial embedding
        hidden = self.fc_latent(z)  # (B, 512)
        hidden = hidden.unsqueeze(1)  # (B, 1, 512)
        
        # If training with teacher forcing
        if caption_tokens is not None:
            embeddings = self.token_embedding(caption_tokens)  # (B, max_length, 512)
            
            # Add positional encoding
            positions = torch.arange(embeddings.size(1), device=embeddings.device).unsqueeze(0)
            pos_embed = self.positional_encoding(positions)
            embeddings = embeddings + pos_embed
            
            # Concatenate with latent
            embeddings = torch.cat([hidden, embeddings[:, :-1, :]], dim=1)  # (B, max_length, 512)
        else:
            # Generate autoregressively (inference)
            embeddings = hidden
            for _ in range(self.max_length - 1):
                # Dummy encoder output (identity)
                memory = hidden
                out = self.transformer_decoder(embeddings, memory)
                next_token_logits = self.output_projection(out[:, -1:, :])  # (B, 1, 8000)
                next_token_idx = next_token_logits.argmax(dim=-1)  # (B, 1)
                next_embedding = self.token_embedding(next_token_idx)  # (B, 1, 512)
                embeddings = torch.cat([embeddings, next_embedding], dim=1)  # (B, t+1, 512)
        
        logits = self.output_projection(embeddings)  # (B, max_length, 8000)
        return logits


class MultimodalModel(nn.Module):
    """Unified multimodal model: image + caption from shared latent (LARGE)"""
    
    def __init__(
        self,
        latent_dim=1024,
        image_channels=3,
        vocab_size=8000,
        max_caption_length=100
    ):
        """
        Initialize large multimodal model
        
        Args:
            latent_dim: Dimension of shared latent space (1024)
            image_channels: Number of image channels (3)
            vocab_size: Size of vocabulary (8000)
            max_caption_length: Maximum caption length (100)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_caption_length = max_caption_length
        
        logger.info(f"Building MultimodalModel:")
        logger.info(f"  Latent dim: {latent_dim}")
        logger.info(f"  Vocab size: {vocab_size}")
        logger.info(f"  Max caption length: {max_caption_length}")
        
        # Encoders
        self.image_encoder = ImageEncoder(latent_dim, image_channels)
        
        # Decoders
        self.image_decoder = ImageDecoder(latent_dim, image_channels)
        self.text_decoder = TextDecoder(
            latent_dim,
            vocab_size,
            max_caption_length,
            embedding_dim=512
        )
    
    def encode(self, images):
        """Encode images to latent space"""
        return self.image_encoder(images)
    
    def decode_image(self, z):
        """Decode latent to image"""
        return self.image_decoder(z)
    
    def decode_text(self, z, caption_tokens=None):
        """Decode latent to caption"""
        return self.text_decoder(z, caption_tokens)
    
    def forward(self, images, caption_tokens=None):
        """
        Full forward pass
        
        Args:
            images: Image tensor (B, 3, 512, 512)
            caption_tokens: Optional caption tokens for training
            
        Returns:
            Dict with 'image_recon', 'caption_logits', 'z'
        """
        # Encode to latent
        z = self.encode(images)
        
        # Decode image
        image_recon = self.decode_image(z)
        
        # Decode caption
        caption_logits = self.decode_text(z, caption_tokens)
        
        return {
            'z': z,
            'image_recon': image_recon,
            'caption_logits': caption_logits
        }


# ==============================================================================
# UTILITIES
# ==============================================================================

def create_data_loaders(
    metadata_path: str,
    images_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        metadata_path: Path to metadata JSON
        images_dir: Path to images directory
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        max_samples: Optional limit on dataset size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = EngravingDataset(
        metadata_path,
        images_dir,
        split="train",
        max_samples=max_samples
    )
    
    val_dataset = EngravingDataset(
        metadata_path,
        images_dir,
        split="val",
        max_samples=max_samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

def main():
    """Test data loading and model"""
    
    logger.info("="*70)
    logger.info("TESTING LARGE MULTIMODAL MODEL")
    logger.info("="*70 + "\n")
    
    # Setup paths (adjust to your system)
    metadata_path = Path("data/metadata/engraving_metadata.json")
    images_dir = Path("data/processed/engraving/resized")
    
    # Create loaders (small batch for testing)
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        metadata_path,
        images_dir,
        batch_size=4,
        max_samples=100  # Small sample for testing
    )
    
    # Initialize model
    logger.info("Initializing multimodal model...")
    device = get_device()
    model = MultimodalModel(
        latent_dim=1024,
        image_channels=3,
        vocab_size=8000,
        max_caption_length=100
    ).to(device)
    
    # Count parameters
    total_params = count_parameters(model)
    logger.info(f"✓ Total trainable parameters: {total_params:,}")
    logger.info(f"  Image Encoder: {count_parameters(model.image_encoder):,}")
    logger.info(f"  Image Decoder: {count_parameters(model.image_decoder):,}")
    logger.info(f"  Text Decoder: {count_parameters(model.text_decoder):,}")
    
    # Test forward pass
    logger.info("\nRunning test forward pass...")
    for batch in train_loader:
        images = batch['images'].to(device)
        captions = batch['captions']
        
        logger.info(f"\nBatch shape: {images.shape}")
        logger.info(f"Sample captions: {captions[:2]}")
        
        with torch.no_grad():
            outputs = model(images)
        
        logger.info(f"\nOutput shapes:")
        logger.info(f"  Latent (z): {outputs['z'].shape}")
        logger.info(f"  Image reconstruction: {outputs['image_recon'].shape}")
        logger.info(f"  Caption logits: {outputs['caption_logits'].shape}")
        
        # Check value ranges
        logger.info(f"\nValue ranges:")
        logger.info(f"  Latent: [{outputs['z'].min():.3f}, {outputs['z'].max():.3f}]")
        logger.info(f"  Image recon: [{outputs['image_recon'].min():.3f}, {outputs['image_recon'].max():.3f}]")
        
        break  # Just test one batch
    
    logger.info("\n✓ Test completed successfully!")
    logger.info("\nModel is ready for training!")


if __name__ == "__main__":
    main()
