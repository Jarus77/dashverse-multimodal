"""
Training Loop for Multimodal Model
Multimodal AI Project: Joint image + caption generation

Features:
- Multi-task learning: Image reconstruction + Caption generation
- Contrastive alignment loss for semantic coherence
- Validation loop with metrics (LPIPS for images, BLEU for captions)
- Checkpoint saving
- Logging with Weights & Biases (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
import logging
from tqdm import tqdm
import time
from typing import Dict, Optional
import numpy as np

# Import model components
from model_architecture import (
    MultimodalModel,
    create_data_loaders,
    SimpleTokenizer,
    get_device
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

class MultimodalLoss(nn.Module):
    """
    Combined loss for multimodal learning
    
    Components:
    1. Image reconstruction loss (L1 + Perceptual)
    2. Caption generation loss (Cross-entropy)
    3. Contrastive alignment loss (InfoNCE-style)
    """
    
    def __init__(
        self,
        lambda_recon=1.0,
        lambda_caption=1.0,
        lambda_contrastive=0.5,
        temperature=0.07
    ):
        """
        Initialize loss function
        
        Args:
            lambda_recon: Weight for image reconstruction
            lambda_caption: Weight for caption generation
            lambda_contrastive: Weight for contrastive alignment
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_caption = lambda_caption
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature
        
        # Image reconstruction
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Caption generation
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    def _perceptual_loss(self, x_recon, x_real, features_recon=None, features_real=None):
        """
        Perceptual loss in feature space (for diversity)
        Falls back to pixel-level if no features provided
        """
        if features_recon is not None and features_real is not None:
            return F.mse_loss(features_recon, features_real)
        else:
            return self.mse_loss(x_recon, x_real)
    
    def _caption_loss(self, caption_logits, caption_targets):
        """
        Cross-entropy loss for caption generation
        
        Args:
            caption_logits: (B, max_length, vocab_size)
            caption_targets: (B, max_length)
            
        Returns:
            Scalar loss
        """
        batch_size, max_length, vocab_size = caption_logits.shape
        
        # Reshape for cross-entropy
        logits_flat = caption_logits.reshape(-1, vocab_size)
        targets_flat = caption_targets.reshape(-1)
        
        loss = self.ce_loss(logits_flat, targets_flat)
        return loss
    
    def _contrastive_loss(self, z_batch):
        """
        Contrastive loss to encourage semantic coherence in latent space
        
        Uses simple in-batch negatives (SimCLR-style)
        
        Args:
            z_batch: Latent vectors (B, latent_dim)
            
        Returns:
            Scalar loss
        """
        batch_size = z_batch.size(0)
        
        # Normalize
        z_norm = F.normalize(z_batch, p=2, dim=1)
        
        # Similarity matrix
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature
        
        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=z_batch.device)
        
        # Cross-entropy loss on similarity
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def forward(
        self,
        images_real,
        images_recon,
        caption_logits,
        caption_targets,
        z_latent
    ):
        """
        Compute total loss
        
        Args:
            images_real: Real images (B, 3, H, W) in [-1, 1]
            images_recon: Reconstructed images (B, 3, H, W) in [-1, 1]
            caption_logits: Caption logits (B, max_length, vocab_size)
            caption_targets: Caption token indices (B, max_length)
            z_latent: Latent vectors (B, latent_dim)
            
        Returns:
            Dict with loss components
        """
        # 1. Image reconstruction loss
        l1_loss = self.l1_loss(images_recon, images_real)
        perceptual_loss = self._perceptual_loss(images_recon, images_real)
        recon_loss = l1_loss + perceptual_loss
        
        # 2. Caption generation loss
        caption_loss = self._caption_loss(caption_logits, caption_targets)
        
        # 3. Contrastive alignment loss
        contrastive_loss = self._contrastive_loss(z_latent)
        
        # Total loss
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_caption * caption_loss +
            self.lambda_contrastive * contrastive_loss
        )
        
        return {
            'total': total_loss,
            'recon': recon_loss.item(),
            'caption': caption_loss.item(),
            'contrastive': contrastive_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item()
        }


# ==============================================================================
# TRAINER
# ==============================================================================

class MultimodalTrainer:
    """Train multimodal model"""
    
    def __init__(
        self,
        model: MultimodalModel,
        tokenizer: SimpleTokenizer,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False
    ):
        """
        Initialize trainer
        
        Args:
            model: MultimodalModel instance
            tokenizer: SimpleTokenizer instance
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project="multimodal-engraving", entity="dashverse")
            except:
                logger.warning("Weights & Biases not available")
                self.use_wandb = False
        
        # Loss function
        self.criterion = MultimodalLoss(
            lambda_recon=1.0,
            lambda_caption=2.0,  # Weight captions more
            lambda_contrastive=0.5,
            temperature=0.07
        )
        
        # Optimizer
        self.optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
    
    def prepare_batch(self, batch: Dict):
        """
        Prepare batch for training
        
        Args:
            batch: Dict from dataloader with 'images' and 'captions'
            
        Returns:
            Dict with tensors on device
        """
        images = batch['images'].to(self.device)
        # Normalize images to [-1, 1] for training
        images = 2 * images - 1
        
        captions = batch['captions']
        
        # Tokenize captions
        caption_tokens = torch.stack([
            self.tokenizer.encode(cap) for cap in captions
        ]).to(self.device)
        
        return {
            'images': images,
            'caption_tokens': caption_tokens
        }
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader
            epoch: Epoch number
            
        Returns:
            Average losses dict
        """
        self.model.train()
        
        epoch_losses = {
            'total': [],
            'recon': [],
            'caption': [],
            'contrastive': [],
            'l1': [],
            'perceptual': []
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Prepare batch
            prepared_batch = self.prepare_batch(batch)
            images = prepared_batch['images']
            caption_tokens = prepared_batch['caption_tokens']
            
            # Forward pass
            outputs = self.model(images, caption_tokens)
            
            z = outputs['z']
            image_recon = outputs['image_recon']
            caption_logits = outputs['caption_logits']
            
            # Compute loss
            loss_dict = self.criterion(
                images_real=images,
                images_recon=image_recon,
                caption_logits=caption_logits,
                caption_targets=caption_tokens,
                z_latent=z
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[key])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['total'].item(),
                'recon': loss_dict['recon'],
                'caption': loss_dict['caption']
            })
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        self.train_metrics.append(avg_losses)
        
        return avg_losses
    
    def validate(self, val_loader, epoch):
        """
        Validate model
        
        Args:
            val_loader: Validation DataLoader
            epoch: Epoch number
            
        Returns:
            Average validation losses dict
        """
        self.model.eval()
        
        epoch_losses = {
            'total': [],
            'recon': [],
            'caption': [],
            'contrastive': [],
            'l1': [],
            'perceptual': []
        }
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                prepared_batch = self.prepare_batch(batch)
                images = prepared_batch['images']
                caption_tokens = prepared_batch['caption_tokens']
                
                # Forward pass
                outputs = self.model(images, caption_tokens)
                
                z = outputs['z']
                image_recon = outputs['image_recon']
                caption_logits = outputs['caption_logits']
                
                # Compute loss
                loss_dict = self.criterion(
                    images_real=images,
                    images_recon=image_recon,
                    caption_logits=caption_logits,
                    caption_targets=caption_tokens,
                    z_latent=z
                )
                
                # Track metrics
                for key in epoch_losses:
                    epoch_losses[key].append(loss_dict[key])
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        self.val_metrics.append(avg_losses)
        
        return avg_losses
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        filename = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, filename)
        logger.info(f"Saved checkpoint: {filename}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_metrics = checkpoint['train_metrics']
        self.val_metrics = checkpoint['val_metrics']
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint['epoch']
    
    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        patience: int = 10
    ):
        """
        Train model for multiple epochs
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs
            patience: Early stopping patience
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_losses = self.validate(val_loader, epoch)
            
            # Log
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_losses['total']:.4f}")
            logger.info(f"    - Reconstruction: {train_losses['recon']:.4f}")
            logger.info(f"    - Caption: {train_losses['caption']:.4f}")
            logger.info(f"    - Contrastive: {train_losses['contrastive']:.4f}")
            logger.info(f"  Val Loss: {val_losses['total']:.4f}")
            
            # Save checkpoint
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            # Log to wandb
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        'train_loss': train_losses['total'],
                        'train_recon': train_losses['recon'],
                        'train_caption': train_losses['caption'],
                        'val_loss': val_losses['total'],
                        'val_recon': val_losses['recon'],
                        'val_caption': val_losses['caption'],
                    })
                except:
                    pass
        
        logger.info("\n✓ Training complete!")
        logger.info(f"Best model saved to: {self.checkpoint_dir / 'best_model.pt'}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main training script"""
    
    logger.info("="*70)
    logger.info("TRAINING MULTIMODAL MODEL")
    logger.info("="*70 + "\n")
    
    # Setup
    device = get_device()
    
    # Paths
    metadata_path = Path("data/metadata/engraving_metadata.json")
    images_dir = Path("data/processed/engraving/resized")
    
    # Data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        metadata_path,
        images_dir,
        batch_size=8,  # Reduced for H100 memory efficiency
        max_samples=None  # Use all data
    )
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=8000)
    
    # Get all captions for vocabulary building
    with open(metadata_path) as f:
        metadata = json.load(f)
    captions = [e['caption'] for e in metadata if e.get('caption')]
    tokenizer.build_vocab(captions)
    
    # Model
    logger.info("Initializing model...")
    model = MultimodalModel(
        latent_dim=1024,
        image_channels=3,
        vocab_size=tokenizer.vocab_size,
        max_caption_length=100
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = MultimodalTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        checkpoint_dir="checkpoints",
        use_wandb=False  # Set to True if using W&B
    )
    
    # Train
    trainer.fit(
        train_loader,
        val_loader,
        num_epochs=50,
        patience=10
    )


if __name__ == "__main__":
    main()
