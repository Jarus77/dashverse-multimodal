"""
Training Loop for Multimodal Model (PRODUCTION)
Multimodal AI Project: Joint Image + Caption Generation

Features:
- Real caption tokenization with SimpleTokenizer
- Vocabulary building from training data
- Configurable batch size and hyperparameters
- Mixed precision training (AMP)
- Checkpoint management
- Validation with metrics
- W&B logging (optional)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import time
from collections import Counter

from model_architecture_large import (
    MultimodalModel, 
    EngravingDataset, 
    collate_fn,
    SimpleTokenizer,
    get_device,
    count_parameters
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Training configuration"""
    
    # Data
    metadata_path = Path("data/metadata/engraving_metadata.json")
    images_dir = Path("data/processed/engraving/resized")
    checkpoint_dir = Path("checkpoints")
    
    # Model
    latent_dim = 1024
    image_channels = 3
    vocab_size = 8000  # Maximum possible vocab size (will auto-adjust to actual)
    max_caption_length = 100
    embedding_dim = 512
    
    # Training
    batch_size = 16  # ✅ CONFIGURABLE - Adjust for your GPU
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-4
    
    # Optimization
    use_amp = True  # Mixed precision training
    gradient_clip = 1.0
    
    # Loss weights
    image_loss_weight = 1.0
    caption_loss_weight = 1.0
    alignment_loss_weight = 0.5
    
    # Data loading
    num_workers = 4
    pin_memory = True
    
    # Logging
    log_interval = 10
    save_interval = 5
    
    def __post_init__(self):
        """Note: vocab_size will auto-adjust to actual vocabulary in data"""
        pass


# ==============================================================================
# TOKENIZER UTILITY
# ==============================================================================

class ProductionTokenizer:
    """Production-ready tokenizer with vocabulary management"""
    
    def __init__(self, vocab_size: int = 8000):
        """
        Initialize tokenizer
        
        Args:
            vocab_size: Size of vocabulary
        """
        self.vocab_size = vocab_size
        self.word2idx = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.next_idx = 4
        self.is_built = False
    
    def build_vocab(self, captions: List[str], min_freq: int = 1):
        """
        Build vocabulary from captions
        
        Args:
            captions: List of caption strings
            min_freq: Minimum word frequency
        """
        logger.info(f"Building vocabulary from {len(captions)} captions...")
        
        # Count word frequencies
        word_freq = Counter()
        for caption in captions:
            words = caption.lower().strip().split()
            word_freq.update(words)
        
        # Add most common words to vocabulary
        num_added = 0
        for word, freq in word_freq.most_common(self.vocab_size - 4):
            if freq >= min_freq and self.next_idx < self.vocab_size:
                self.word2idx[word] = self.next_idx
                self.idx2word[self.next_idx] = word
                self.next_idx += 1
                num_added += 1
        
        self.is_built = True
        
        logger.info(f"✓ Vocabulary built:")
        logger.info(f"  Total tokens: {len(self.word2idx)}")
        logger.info(f"  Unique words added: {num_added}")
        logger.info(f"  Most common: {list(word_freq.most_common(5))}")
    
    def encode(self, caption: str, max_length: int = 100) -> torch.Tensor:
        """
        Encode caption to token tensor
        
        Args:
            caption: Caption string
            max_length: Maximum length to pad/truncate to
            
        Returns:
            Token tensor of shape (max_length,)
        """
        if not self.is_built:
            raise RuntimeError("Tokenizer vocabulary not built. Call build_vocab() first.")
        
        # Tokenize
        tokens = [1]  # <START>
        
        words = caption.lower().strip().split()
        for word in words:
            # Clean word
            word = word.strip('.,!?;:\'"')
            
            # Get token index
            if word:
                idx = self.word2idx.get(word, 3)  # <UNK> if not found
                tokens.append(idx)
        
        tokens.append(2)  # <END>
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))  # Pad with <PAD>
        else:
            tokens = tokens[:max_length]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def batch_encode(self, captions: List[str], max_length: int = 100) -> torch.Tensor:
        """
        Encode batch of captions
        
        Args:
            captions: List of caption strings
            max_length: Maximum length
            
        Returns:
            Token tensor of shape (batch_size, max_length)
        """
        batch_tokens = []
        for caption in captions:
            tokens = self.encode(caption, max_length)
            batch_tokens.append(tokens)
        
        return torch.stack(batch_tokens, dim=0)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode token tensor back to caption
        
        Args:
            tokens: Token tensor of shape (seq_len,)
            
        Returns:
            Caption string
        """
        words = []
        for idx in tokens:
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            
            # Skip special tokens
            if idx in [0, 1, 2]:  # <PAD>, <START>, <END>
                continue
            
            word = self.idx2word.get(idx, '<UNK>')
            words.append(word)
        
        return ' '.join(words)
    
    def save(self, path: str):
        """Save tokenizer to file"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': {int(k): v for k, v in self.idx2word.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        logger.info(f"✓ Tokenizer saved to {path}")
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        self.vocab_size = vocab_data['vocab_size']
        self.next_idx = max(self.idx2word.keys()) + 1
        self.is_built = True
        
        logger.info(f"✓ Tokenizer loaded from {path}")


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

class MultimodalLoss(nn.Module):
    """Combined loss for multimodal model"""
    
    def __init__(
        self,
        image_loss_weight: float = 1.0,
        caption_loss_weight: float = 1.0,
        alignment_loss_weight: float = 0.5,
        image_loss_type: str = "l2"
    ):
        """
        Initialize multimodal loss
        
        Args:
            image_loss_weight: Weight for image reconstruction
            caption_loss_weight: Weight for caption generation
            alignment_loss_weight: Weight for latent alignment
            image_loss_type: "l2", "l1", or "smooth_l1"
        """
        super().__init__()
        
        self.image_loss_weight = image_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.alignment_loss_weight = alignment_loss_weight
        
        # Image reconstruction loss
        if image_loss_type == "l2":
            self.image_loss_fn = nn.MSELoss()
        elif image_loss_type == "l1":
            self.image_loss_fn = nn.L1Loss()
        elif image_loss_type == "smooth_l1":
            self.image_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown image loss type: {image_loss_type}")
        
        # Caption loss (ignore padding tokens)
        self.caption_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    def compute_image_loss(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Image reconstruction loss"""
        return self.image_loss_fn(reconstructed, original)
    
    def compute_caption_loss(self, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Caption generation loss"""
        logits_flat = logits.view(-1, logits.size(-1))
        tokens_flat = tokens.view(-1)
        return self.caption_loss_fn(logits_flat, tokens_flat)
    
    def compute_alignment_loss(
        self,
        latent: torch.Tensor,
        reconstructed: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """Latent alignment loss"""
        latent_norm = torch.norm(latent, dim=1, keepdim=True)
        pixel_diff = torch.abs(reconstructed - original).mean()
        alignment = latent_norm.mean() - (1.0 - pixel_diff)
        return torch.abs(alignment)
    
    def forward(
        self,
        image_recon: torch.Tensor,
        image_original: torch.Tensor,
        caption_logits: torch.Tensor,
        caption_tokens: torch.Tensor,
        latent: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        image_loss = self.compute_image_loss(image_recon, image_original)
        caption_loss = self.compute_caption_loss(caption_logits, caption_tokens)
        alignment_loss = self.compute_alignment_loss(latent, image_recon, image_original)
        
        total_loss = (
            self.image_loss_weight * image_loss +
            self.caption_loss_weight * caption_loss +
            self.alignment_loss_weight * alignment_loss
        )
        
        return {
            'total': total_loss,
            'image': image_loss,
            'caption': caption_loss,
            'alignment': alignment_loss
        }


# ==============================================================================
# METRICS
# ==============================================================================

class MetricTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.losses = {
            'total': [],
            'image': [],
            'caption': [],
            'alignment': []
        }
        self.step = 0
    
    def update(self, loss_dict: Dict[str, float]):
        """Update metrics"""
        for key, value in loss_dict.items():
            if key in self.losses:
                self.losses[key].append(value)
        self.step += 1
    
    def get_avg(self, key: str = 'total') -> float:
        """Get average loss"""
        if not self.losses[key]:
            return 0.0
        return np.mean(self.losses[key])
    
    def reset(self):
        """Reset metrics"""
        self.losses = {k: [] for k in self.losses}
        self.step = 0
    
    def summary(self) -> str:
        """Get summary string"""
        return (
            f"Loss: {self.get_avg('total'):.4f} | "
            f"Img: {self.get_avg('image'):.4f} | "
            f"Cap: {self.get_avg('caption'):.4f} | "
            f"Align: {self.get_avg('alignment'):.4f}"
        )


# ==============================================================================
# TRAINING
# ==============================================================================

class Trainer:
    """Training manager"""
    
    def __init__(
        self,
        model: MultimodalModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: ProductionTokenizer,
        device: torch.device,
        config: Config
    ):
        """
        Initialize trainer
        
        Args:
            model: Multimodal model
            train_loader: Training data loader
            val_loader: Validation data loader
            tokenizer: Tokenizer for captions
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss function
        self.loss_fn = MultimodalLoss(
            image_loss_weight=config.image_loss_weight,
            caption_loss_weight=config.caption_loss_weight,
            alignment_loss_weight=config.alignment_loss_weight
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Metrics
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        
        logger.info(f"Trainer initialized with config:")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Use AMP: {config.use_amp}")
    
    def prepare_caption_tokens(self, captions: List[str]) -> torch.Tensor:
        """
        ✅ PRODUCTION: Real tokenization
        
        Args:
            captions: List of caption strings
            
        Returns:
            Token tensor (B, max_length)
        """
        tokens = self.tokenizer.batch_encode(
            captions,
            max_length=self.config.max_caption_length
        )
        return tokens.to(self.device)
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device, dtype=torch.float32)
            captions = batch['captions']
            
            # ✅ PRODUCTION: Real caption tokenization
            caption_tokens = self.prepare_caption_tokens(captions)
            
            # Forward pass
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images, caption_tokens)
                    losses = self.loss_fn(
                        outputs['image_recon'],
                        images,
                        outputs['caption_logits'],
                        caption_tokens,
                        outputs['z']
                    )
                    total_loss = losses['total']
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, caption_tokens)
                losses = self.loss_fn(
                    outputs['image_recon'],
                    images,
                    outputs['caption_logits'],
                    caption_tokens,
                    outputs['z']
                )
                total_loss = losses['total']
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                self.optimizer.step()
            
            # Update metrics
            loss_dict = {k: v.item() for k, v in losses.items()}
            self.train_metrics.update(loss_dict)
            
            # Update progress bar
            pbar.set_postfix_str(self.train_metrics.summary())
        
        avg_loss = self.train_metrics.get_avg('total')
        logger.info(f"Epoch {epoch+1} Train - {self.train_metrics.summary()}")
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Validate one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            images = batch['images'].to(self.device, dtype=torch.float32)
            captions = batch['captions']
            
            # ✅ PRODUCTION: Real caption tokenization
            caption_tokens = self.prepare_caption_tokens(captions)
            
            # Forward pass
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images, caption_tokens)
                    losses = self.loss_fn(
                        outputs['image_recon'],
                        images,
                        outputs['caption_logits'],
                        caption_tokens,
                        outputs['z']
                    )
            else:
                outputs = self.model(images, caption_tokens)
                losses = self.loss_fn(
                    outputs['image_recon'],
                    images,
                    outputs['caption_logits'],
                    caption_tokens,
                    outputs['z']
                )
            
            # Update metrics
            loss_dict = {k: v.item() for k, v in losses.items()}
            self.val_metrics.update(loss_dict)
            
            pbar.set_postfix_str(self.val_metrics.summary())
        
        avg_loss = self.val_metrics.get_avg('total')
        logger.info(f"Epoch {epoch+1} Val - {self.val_metrics.summary()}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_name = f"epoch_{epoch:03d}.pt"
        checkpoint_path = self.config.checkpoint_dir / checkpoint_name
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.train_metrics.get_avg('total'),
            'val_loss': self.val_metrics.get_avg('total'),
        }, checkpoint_path)
        
        logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.config.checkpoint_dir / "best.pt"
            torch.save(self.model.state_dict(), best_path)
            logger.info(f"✓ Best model saved: {best_path}")
    
    def train(self):
        """Train for configured number of epochs"""
        logger.info("="*70)
        logger.info(f"STARTING TRAINING - {self.config.num_epochs} EPOCHS")
        logger.info("="*70 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Train and validate
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
            
            # Check if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch time: {epoch_time:.1f}s | LR: {current_lr:.2e}\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main training script"""
    
    logger.info("="*70)
    logger.info("MULTIMODAL MODEL TRAINING PIPELINE (PRODUCTION)")
    logger.info("="*70 + "\n")
    
    config = Config()
    device = get_device()
    
    # ========== STEP 1: Load Data ==========
    logger.info("STEP 1: Loading dataset...")
    with open(config.metadata_path, 'r') as f:
        all_metadata = json.load(f)
    
    # Extract captions for tokenizer vocabulary building
    all_captions = [e['caption'] for e in all_metadata if e.get('caption')]
    logger.info(f"Total captions found: {len(all_captions)}")
    
    # ========== STEP 2: Build Tokenizer ==========
    logger.info("\nSTEP 2: Building tokenizer...")
    tokenizer = ProductionTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(all_captions, min_freq=1)
    
    # Create checkpoint directory
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer for inference
    tokenizer.save(str(config.checkpoint_dir / "tokenizer.json"))
    
    # ========== STEP 3: Create Data Loaders ==========
    logger.info("\nSTEP 3: Creating data loaders...")
    train_dataset = EngravingDataset(
        config.metadata_path,
        config.images_dir,
        split="train"
    )
    val_dataset = EngravingDataset(
        config.metadata_path,
        config.images_dir,
        split="val"
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    
    # ========== STEP 4: Build Model ==========
    logger.info("\nSTEP 4: Building model...")
    
    # Use actual vocabulary size from tokenizer (not config)
    actual_vocab_size = len(tokenizer.word2idx)
    logger.info(f"Using actual vocabulary size: {actual_vocab_size} (instead of {config.vocab_size})")
    
    model = MultimodalModel(
        latent_dim=config.latent_dim,
        image_channels=config.image_channels,
        vocab_size=actual_vocab_size,  # Use actual vocab size
        max_caption_length=config.max_caption_length
    ).to(device)
    
    total_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}\n")
    
    # ========== STEP 5: Test Tokenization ==========
    logger.info("STEP 5: Testing tokenization...")
    sample_caption = all_captions[0]
    tokens = tokenizer.encode(sample_caption, config.max_caption_length)
    decoded = tokenizer.decode(tokens)
    logger.info(f"Original: {sample_caption[:100]}")
    logger.info(f"Encoded shape: {tokens.shape}")
    logger.info(f"Decoded: {decoded[:100]}\n")
    
    # ========== STEP 6: Start Training ==========
    logger.info("STEP 6: Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        config=config
    )
    
    trainer.train()
    
    logger.info("\n✓ Training complete!")
    logger.info(f"Checkpoints saved to: {config.checkpoint_dir}")
    logger.info(f"Best model: {config.checkpoint_dir / 'best.pt'}")
    logger.info(f"Tokenizer: {config.checkpoint_dir / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
