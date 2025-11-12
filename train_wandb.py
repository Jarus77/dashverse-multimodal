"""
W&B Integration for Multimodal Training
Comprehensive logging of all important parameters for rigorous analysis

Features:
- Full hyperparameter tracking
- Loss component monitoring
- Latent space analysis
- Gradient statistics
- Token & caption analysis
- Hardware monitoring
- Custom visualizations
- Model architecture logging
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
import psutil
import GPUtil

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("WARNING: wandb not installed. Install with: pip install wandb")

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
    vocab_size = 8000  # Will auto-adjust to actual
    max_caption_length = 100
    embedding_dim = 512
    
    # Training
    batch_size = 16  # ✅ CONFIGURABLE
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-4
    
    # Optimization
    use_amp = True
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


# ==============================================================================
# W&B SETUP & UTILITIES
# ==============================================================================

class WandBConfig:
    """W&B Configuration"""
    
    # W&B Settings
    project_name = "multimodal-engravings"
    entity = None  # Set to your username or team name
    experiment_name = "full-training-v1"
    
    # What to log
    log_model = True
    log_frequency = 10  # Log every N batches
    log_media_frequency = 100  # Log images every N batches
    save_model_frequency = 5  # Save checkpoint every N epochs
    
    # Advanced analysis
    track_gradients = True
    track_hardware = True
    track_latent_space = True
    track_caption_quality = True


def setup_wandb(config, model, config_dict: Dict):
    """
    Setup Weights & Biases
    
    Args:
        config: Training config
        model: Model to log
        config_dict: Config dictionary to log
    """
    if not HAS_WANDB:
        logger.warning("W&B not installed. Skipping W&B setup.")
        return None
    
    logger.info("="*70)
    logger.info("SETTING UP WEIGHTS & BIASES")
    logger.info("="*70)
    
    try:
        wandb.login()
        logger.info("✓ W&B login successful")
    except Exception as e:
        logger.error(f"W&B login failed: {e}")
        logger.info("Run 'wandb login' in terminal first")
        return None
    
    # Initialize W&B run
    run = wandb.init(
        project=WandBConfig.project_name,
        entity=WandBConfig.entity,
        name=WandBConfig.experiment_name,
        config=config_dict,
        notes="Comprehensive multimodal training with rigorous parameter logging"
    )
    
    logger.info(f"✓ W&B run initialized: {run.url}")
    
    # Log model architecture
    if WandBConfig.log_model:
        wandb.watch(model, log="all", log_freq=100)
        logger.info("✓ Model gradients being tracked")
    
    return run


# ==============================================================================
# TOKENIZER UTILITY
# ==============================================================================

class ProductionTokenizer:
    """Production-ready tokenizer with vocabulary management"""
    
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
        self.is_built = False
    
    def build_vocab(self, captions: List[str], min_freq: int = 1):
        """Build vocabulary from captions"""
        logger.info(f"Building vocabulary from {len(captions)} captions...")
        
        word_freq = Counter()
        for caption in captions:
            words = caption.lower().strip().split()
            word_freq.update(words)
        
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
        """Encode caption to token tensor"""
        if not self.is_built:
            raise RuntimeError("Tokenizer vocabulary not built.")
        
        tokens = [1]  # <START>
        
        words = caption.lower().strip().split()
        for word in words:
            word = word.strip('.,!?;:\'"')
            if word:
                idx = self.word2idx.get(word, 3)
                tokens.append(idx)
        
        tokens.append(2)  # <END>
        
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def batch_encode(self, captions: List[str], max_length: int = 100) -> torch.Tensor:
        """Encode batch of captions"""
        batch_tokens = []
        for caption in captions:
            tokens = self.encode(caption, max_length)
            batch_tokens.append(tokens)
        
        return torch.stack(batch_tokens, dim=0)
    
    def save(self, path: str):
        """Save tokenizer"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': {int(k): v for k, v in self.idx2word.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        logger.info(f"✓ Tokenizer saved to {path}")


# ==============================================================================
# ADVANCED METRICS TRACKING
# ==============================================================================

class AdvancedMetricsTracker:
    """Track detailed metrics for analysis"""
    
    def __init__(self):
        self.metrics = {
            # Loss components
            'loss_total': [],
            'loss_image': [],
            'loss_caption': [],
            'loss_alignment': [],
            
            # Latent space statistics
            'latent_mean': [],
            'latent_std': [],
            'latent_min': [],
            'latent_max': [],
            'latent_norm': [],
            
            # Gradient statistics
            'grad_mean': [],
            'grad_std': [],
            'grad_max': [],
            'grad_norm': [],
            
            # Image reconstruction
            'image_mse': [],
            'image_ssim': [],
            'image_l1': [],
            
            # Caption metrics
            'caption_perplexity': [],
            'caption_entropy': [],
            'token_accuracy': [],
            
            # Hardware
            'gpu_memory_used': [],
            'gpu_memory_percent': [],
            'cpu_percent': [],
            
            # Learning dynamics
            'learning_rate': [],
            'batch_size': [],
            'epoch': []
        }
        self.step = 0
    
    def update(self, metrics_dict: Dict):
        """Update metrics"""
        for key, value in metrics_dict.items():
            if key in self.metrics:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.metrics[key].append(float(value))
        self.step += 1
    
    def get_epoch_summary(self) -> Dict:
        """Get summary for current epoch"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values[-100:])  # Last 100 steps
                summary[f"{key}_std"] = np.std(values[-100:])
        return summary
    
    def reset_epoch(self):
        """Reset for new epoch"""
        pass


def compute_image_metrics(predicted: torch.Tensor, original: torch.Tensor) -> Dict:
    """
    Compute image reconstruction metrics
    
    Args:
        predicted: (B, 3, H, W) in [-1, 1]
        original: (B, 3, H, W) in [0, 1]
        
    Returns:
        Dict of metrics
    """
    # Normalize to [0, 1]
    pred_norm = (predicted + 1) / 2
    pred_norm = torch.clamp(pred_norm, 0, 1)
    
    # MSE
    mse = nn.functional.mse_loss(pred_norm, original)
    
    # L1
    l1 = nn.functional.l1_loss(pred_norm, original)
    
    # SSIM approximation (using correlation)
    mean_diff = (pred_norm - original).abs().mean()
    
    return {
        'image_mse': mse,
        'image_l1': l1,
        'image_diff': mean_diff
    }


def compute_caption_metrics(logits: torch.Tensor, tokens: torch.Tensor) -> Dict:
    """
    Compute caption generation metrics
    
    Args:
        logits: (B, seq_len, vocab_size)
        tokens: (B, seq_len)
        
    Returns:
        Dict of metrics
    """
    # Token accuracy (ignore padding)
    mask = tokens != 0
    predictions = logits.argmax(dim=-1)
    correct = (predictions == tokens) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    # Perplexity
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, tokens.unsqueeze(2)).squeeze(2)
    token_log_probs = token_log_probs * mask.float()
    perplexity = torch.exp(-token_log_probs.sum() / mask.sum())
    
    # Entropy
    probs = nn.functional.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    return {
        'caption_accuracy': accuracy,
        'caption_perplexity': perplexity,
        'caption_entropy': entropy
    }


def compute_latent_metrics(latent: torch.Tensor) -> Dict:
    """
    Compute latent space statistics
    
    Args:
        latent: (B, latent_dim)
        
    Returns:
        Dict of metrics
    """
    return {
        'latent_mean': latent.mean(),
        'latent_std': latent.std(),
        'latent_min': latent.min(),
        'latent_max': latent.max(),
        'latent_norm': torch.norm(latent, dim=1).mean()
    }


def get_gradient_stats(model: nn.Module) -> Dict:
    """
    Compute gradient statistics for analysis
    
    Args:
        model: Model to analyze
        
    Returns:
        Dict of gradient stats
    """
    grad_norms = []
    grad_means = []
    grad_stds = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(torch.norm(param.grad))
            grad_means.append(param.grad.mean().abs())
            grad_stds.append(param.grad.std())
    
    if grad_norms:
        return {
            'grad_mean': torch.stack(grad_means).mean(),
            'grad_std': torch.stack(grad_stds).mean(),
            'grad_max': torch.stack(grad_norms).max(),
            'grad_norm': torch.stack(grad_norms).mean()
        }
    return {}


def get_hardware_stats() -> Dict:
    """
    Get hardware utilization statistics
    
    Returns:
        Dict of hardware stats
    """
    # CPU
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # GPU
    gpu_memory_used = 0
    gpu_memory_percent = 0
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # First GPU
            gpu_memory_used = gpu.memoryUsed
            gpu_memory_percent = gpu.memoryUtil * 100
    except:
        pass
    
    return {
        'hardware_cpu_percent': cpu_percent,
        'hardware_gpu_memory_mb': gpu_memory_used,
        'hardware_gpu_memory_percent': gpu_memory_percent
    }


# ==============================================================================
# ENHANCED TRAINER WITH W&B
# ==============================================================================

class EnhancedTrainer:
    """Training manager with W&B logging"""
    
    def __init__(
        self,
        model: MultimodalModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        device: torch.device,
        config,
        run=None  # W&B run object
    ):
        """
        Initialize trainer with W&B
        
        Args:
            model: Multimodal model
            train_loader: Training data loader
            val_loader: Validation data loader
            tokenizer: Tokenizer
            device: Device to train on
            config: Training config
            run: W&B run object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.run = run
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        self.caption_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        
        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
        )
        
        # Scaler for mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Metrics trackers
        self.train_metrics = AdvancedMetricsTracker()
        self.val_metrics = AdvancedMetricsTracker()
        
        logger.info("✓ EnhancedTrainer initialized with W&B")
    
    def prepare_caption_tokens(self, captions: List[str]) -> torch.Tensor:
        """Prepare caption tokens"""
        tokens = self.tokenizer.batch_encode(
            captions,
            max_length=self.config.max_caption_length
        )
        return tokens.to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch with detailed logging"""
        self.model.train()
        self.train_metrics.reset_epoch()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train")
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device, dtype=torch.float32)
            captions = batch['captions']
            caption_tokens = self.prepare_caption_tokens(captions)
            
            # Forward pass
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images, caption_tokens)
                    
                    # Compute losses
                    image_loss = self.loss_fn(outputs['image_recon'], images)
                    caption_logits_flat = outputs['caption_logits'].view(-1, outputs['caption_logits'].size(-1))
                    caption_tokens_flat = caption_tokens.view(-1)
                    caption_loss = self.caption_loss_fn(caption_logits_flat, caption_tokens_flat)
                    
                    # Alignment loss
                    latent_norm = torch.norm(outputs['z'], dim=1, keepdim=True)
                    pixel_diff = torch.abs(outputs['image_recon'] - images).mean()
                    alignment_loss = torch.abs(latent_norm.mean() - (1.0 - pixel_diff))
                    
                    total_loss = (
                        1.0 * image_loss +
                        1.0 * caption_loss +
                        0.5 * alignment_loss
                    )
                
                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, caption_tokens)
                image_loss = self.loss_fn(outputs['image_recon'], images)
                caption_logits_flat = outputs['caption_logits'].view(-1, outputs['caption_logits'].size(-1))
                caption_tokens_flat = caption_tokens.view(-1)
                caption_loss = self.caption_loss_fn(caption_logits_flat, caption_tokens_flat)
                
                latent_norm = torch.norm(outputs['z'], dim=1, keepdim=True)
                pixel_diff = torch.abs(outputs['image_recon'] - images).mean()
                alignment_loss = torch.abs(latent_norm.mean() - (1.0 - pixel_diff))
                
                total_loss = image_loss + caption_loss + 0.5 * alignment_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Collect metrics for logging
            with torch.no_grad():
                # Loss metrics
                metrics = {
                    'loss_total': total_loss.item(),
                    'loss_image': image_loss.item(),
                    'loss_caption': caption_loss.item(),
                    'loss_alignment': alignment_loss.item(),
                }
                
                # Latent space
                latent_metrics = compute_latent_metrics(outputs['z'])
                metrics.update({f'train_{k}': v for k, v in latent_metrics.items()})
                
                # Image metrics
                image_metrics = compute_image_metrics(outputs['image_recon'], images)
                metrics.update({f'train_{k}': v for k, v in image_metrics.items()})
                
                # Caption metrics
                caption_metrics = compute_caption_metrics(outputs['caption_logits'], caption_tokens)
                metrics.update({f'train_{k}': v for k, v in caption_metrics.items()})
                
                # Gradient stats
                if WandBConfig.track_gradients:
                    grad_stats = get_gradient_stats(self.model)
                    metrics.update({f'train_{k}': v for k, v in grad_stats.items()})
                
                # Hardware stats
                if WandBConfig.track_hardware:
                    hw_stats = get_hardware_stats()
                    metrics.update({f'train_{k}': v for k, v in hw_stats.items()})
                
                # Learning rate
                metrics['train_learning_rate'] = self.optimizer.param_groups[0]['lr']
                metrics['train_epoch'] = epoch
                metrics['train_batch'] = batch_idx
                
                epoch_metrics.append(metrics)
            
            # Log to W&B every N batches
            if (batch_idx + 1) % WandBConfig.log_frequency == 0:
                if self.run:
                    avg_metrics = {
                        k: np.mean([m[k] for m in epoch_metrics[-WandBConfig.log_frequency:]])
                        for k in epoch_metrics[-1].keys()
                    }
                    self.run.log(avg_metrics, step=epoch * len(self.train_loader) + batch_idx)
            
            pbar.set_postfix({
                'loss': total_loss.item(),
                'img': image_loss.item(),
                'cap': caption_loss.item()
            })
        
        # Epoch summary
        epoch_summary = {
            'epoch': epoch,
            'train_loss_avg': np.mean([m['loss_total'] for m in epoch_metrics]),
            'train_loss_std': np.std([m['loss_total'] for m in epoch_metrics]),
        }
        
        return epoch_summary
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """Validate with detailed logging"""
        self.model.eval()
        self.val_metrics.reset_epoch()
        
        pbar = tqdm(self.val_loader, desc="Validation")
        epoch_metrics = []
        
        for batch in pbar:
            images = batch['images'].to(self.device, dtype=torch.float32)
            captions = batch['captions']
            caption_tokens = self.prepare_caption_tokens(captions)
            
            # Forward
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images, caption_tokens)
                    
                    image_loss = self.loss_fn(outputs['image_recon'], images)
                    caption_logits_flat = outputs['caption_logits'].view(-1, outputs['caption_logits'].size(-1))
                    caption_tokens_flat = caption_tokens.view(-1)
                    caption_loss = self.caption_loss_fn(caption_logits_flat, caption_tokens_flat)
                    
                    latent_norm = torch.norm(outputs['z'], dim=1, keepdim=True)
                    pixel_diff = torch.abs(outputs['image_recon'] - images).mean()
                    alignment_loss = torch.abs(latent_norm.mean() - (1.0 - pixel_diff))
                    
                    total_loss = image_loss + caption_loss + 0.5 * alignment_loss
            else:
                outputs = self.model(images, caption_tokens)
                image_loss = self.loss_fn(outputs['image_recon'], images)
                caption_logits_flat = outputs['caption_logits'].view(-1, outputs['caption_logits'].size(-1))
                caption_tokens_flat = caption_tokens.view(-1)
                caption_loss = self.caption_loss_fn(caption_logits_flat, caption_tokens_flat)
                
                latent_norm = torch.norm(outputs['z'], dim=1, keepdim=True)
                pixel_diff = torch.abs(outputs['image_recon'] - images).mean()
                alignment_loss = torch.abs(latent_norm.mean() - (1.0 - pixel_diff))
                
                total_loss = image_loss + caption_loss + 0.5 * alignment_loss
            
            # Metrics
            metrics = {
                'loss_total': total_loss.item(),
                'loss_image': image_loss.item(),
                'loss_caption': caption_loss.item(),
                'loss_alignment': alignment_loss.item(),
            }
            
            latent_metrics = compute_latent_metrics(outputs['z'])
            metrics.update({f'val_{k}': v for k, v in latent_metrics.items()})
            
            image_metrics = compute_image_metrics(outputs['image_recon'], images)
            metrics.update({f'val_{k}': v for k, v in image_metrics.items()})
            
            caption_metrics = compute_caption_metrics(outputs['caption_logits'], caption_tokens)
            metrics.update({f'val_{k}': v for k, v in caption_metrics.items()})
            
            if WandBConfig.track_hardware:
                hw_stats = get_hardware_stats()
                metrics.update({f'val_{k}': v for k, v in hw_stats.items()})
            
            metrics['val_epoch'] = epoch
            epoch_metrics.append(metrics)
            
            pbar.set_postfix({'loss': total_loss.item()})
        
        # Epoch summary
        epoch_summary = {
            'epoch': epoch,
            'val_loss_avg': np.mean([m['loss_total'] for m in epoch_metrics]),
            'val_loss_std': np.std([m['loss_total'] for m in epoch_metrics]),
        }
        
        # Log epoch to W&B
        if self.run:
            log_dict = {}
            for key in epoch_metrics[0].keys():
                values = [m[key] for m in epoch_metrics]
                log_dict[f'{key}_mean'] = np.mean(values)
                log_dict[f'{key}_std'] = np.std(values)
            log_dict['epoch'] = epoch
            self.run.log(log_dict, step=epoch)
        
        return epoch_summary
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint_name = f"epoch_{epoch:03d}.pt"
        checkpoint_path = self.config.checkpoint_dir / checkpoint_name
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        if is_best:
            best_path = self.config.checkpoint_dir / "best.pt"
            torch.save(self.model.state_dict(), best_path)
            logger.info(f"✓ Best model saved")
    
    def train(self):
        """Train with W&B logging"""
        logger.info("\n" + "="*70)
        logger.info("STARTING TRAINING WITH W&B LOGGING")
        logger.info("="*70 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_summary = self.train_epoch(epoch)
            
            # Validate
            val_summary = self.validate(epoch)
            
            # Schedule
            self.scheduler.step()
            
            # Checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
            
            if val_summary['val_loss_avg'] < best_val_loss:
                best_val_loss = val_summary['val_loss_avg']
                self.save_checkpoint(epoch, is_best=True)
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch {epoch+1} | "
                f"Train Loss: {train_summary['train_loss_avg']:.4f} | "
                f"Val Loss: {val_summary['val_loss_avg']:.4f} | "
                f"Time: {epoch_time:.1f}s | LR: {current_lr:.2e}"
            )
        
        if self.run:
            self.run.finish()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main training script with W&B logging"""
    
    logger.info("="*70)
    logger.info("MULTIMODAL TRAINING WITH W&B LOGGING")
    logger.info("="*70 + "\n")
    
    # Setup
    device = get_device()
    config = Config()
    
    # Load data
    logger.info("STEP 1: Loading dataset...")
    metadata_path = config.metadata_path
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)
    
    all_captions = [e['caption'] for e in all_metadata if e.get('caption')]
    logger.info(f"Total captions found: {len(all_captions)}\n")
    
    # Build tokenizer
    logger.info("STEP 2: Building tokenizer...")
    tokenizer = ProductionTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(all_captions, min_freq=1)
    
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(config.checkpoint_dir / "tokenizer.json"))
    
    actual_vocab_size = len(tokenizer.word2idx)
    logger.info(f"Using actual vocabulary size: {actual_vocab_size}\n")
    
    # Create data loaders
    logger.info("STEP 3: Creating data loaders...")
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
    logger.info(f"Batches per epoch: {len(train_loader)}\n")
    
    # Build model
    logger.info("STEP 4: Building model...")
    model = MultimodalModel(
        latent_dim=config.latent_dim,
        image_channels=config.image_channels,
        vocab_size=actual_vocab_size,
        max_caption_length=config.max_caption_length
    ).to(device)
    
    total_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}\n")
    
    # Setup W&B
    logger.info("STEP 5: Setting up W&B...\n")
    config_dict = {
        # Architecture
        "latent_dim": config.latent_dim,
        "vocab_size": actual_vocab_size,
        "embedding_dim": config.embedding_dim,
        "max_caption_length": config.max_caption_length,
        "total_parameters": total_params,
        
        # Training
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealing",
        
        # Loss weights
        "image_loss_weight": config.image_loss_weight,
        "caption_loss_weight": config.caption_loss_weight,
        "alignment_loss_weight": config.alignment_loss_weight,
        
        # Data
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "use_amp": config.use_amp,
    }
    
    run = setup_wandb(config, model, config_dict)
    
    # Train
    logger.info("\nSTEP 6: Starting training...\n")
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        config=config,
        run=run
    )
    
    trainer.train()
    
    logger.info("\n✓ Training complete!")
    if run:
        logger.info(f"Dashboard: {run.url}")


# ==============================================================================
# IMPORTS AT TOP (add if missing)
# ==============================================================================

# Add these imports if you see errors:
# from pathlib import Path
# from collections import Counter
# import psutil
# import GPUtil


if __name__ == "__main__":
    main()
