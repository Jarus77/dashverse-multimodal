# Training Configuration & Important Notes - RESOLVED ✅

## What Was Fixed

### 1. ✅ BATCH SIZE CONFIGURATION

**Location:** `train_production.py` → `Config` class

```python
class Config:
    batch_size = 16  # ✅ CONFIGURABLE - Adjust for your GPU
```

**GPU Memory Requirements:**
| Batch Size | GPU Memory | Speed | Quality |
|-----------|-----------|-------|---------|
| 8 | ~12GB | Slower | Similar |
| 16 | ~20GB | **Good** | **Recommended** |
| 32 | ~35GB | Faster | Better |
| 64 | ~65GB | Fastest | Best |

**For H100 (80GB):** You can safely use batch_size=32 or even 64

**To change:**
```python
class Config:
    batch_size = 32  # Change this line
```

---

### 2. ✅ CAPTION TOKENIZATION (MAJOR FIX)

**Problem:** Previous version used dummy tokens

**Solution:** Implemented `ProductionTokenizer` with:
- ✅ Vocabulary building from training captions
- ✅ Word-to-index mapping
- ✅ Proper encoding/decoding
- ✅ Batch processing
- ✅ Persistence (save/load)

**How it works:**

```python
# Step 1: Build vocab from all captions
tokenizer = ProductionTokenizer(vocab_size=8000)
tokenizer.build_vocab(all_captions, min_freq=1)

# Step 2: Encode caption to tokens
tokens = tokenizer.encode("An engraving depicting a woman")
# Returns: tensor([1, 234, 445, 112, 556, 2, 0, 0, ...])
#          <START>, <word>, <word>, <word>, <word>, <END>, <PAD>, ...

# Step 3: Decode tokens back
caption = tokenizer.decode(tokens)
# Returns: "an engraving depicting a woman"
```

**Vocabulary Building Process:**
1. Count word frequencies across all 5,141 captions
2. Select top 7,996 most common words (8,000 - 4 special tokens)
3. Create mapping: word → index
4. Handle out-of-vocabulary with `<UNK>` token

**Token Structure:**
- `<PAD>` (0): Padding token
- `<START>` (1): Sentence start
- `<END>` (2): Sentence end
- `<UNK>` (3): Unknown word
- Words 4-7,999: Your vocabulary

---

### 3. ✅ TRAINING PIPELINE IMPROVEMENTS

**New Training Pipeline:**

```
STEP 1: Load dataset
  ↓
STEP 2: Build tokenizer from all captions
  ↓
STEP 3: Create data loaders
  ↓
STEP 4: Initialize model
  ↓
STEP 5: Test tokenization (validate encoding/decoding)
  ↓
STEP 6: Start training with REAL tokens
```

**Key Features:**

| Feature | Before | After |
|---------|--------|-------|
| Tokenization | Dummy tokens | Real tokens from vocab |
| Batch Processing | Manual loop | `batch_encode()` |
| Vocab Building | N/A | Automatic from captions |
| Token Persistence | N/A | Save/load from JSON |
| Error Handling | Basic | Comprehensive |
| Validation | N/A | Pre-training test |

---

## Configuration Reference

### Batch Size (GPU Memory vs Speed)

```python
class Config:
    # For H100 80GB - RECOMMENDED
    batch_size = 32  # Good balance of speed and memory
    
    # For RTX 3090 24GB
    batch_size = 8   # Lower memory requirement
    
    # For multiple GPUs
    batch_size = 64  # Can go higher with DataParallel
```

### Learning Rate & Optimization

```python
class Config:
    # Current (good defaults)
    learning_rate = 1e-3       # Adam initial LR
    weight_decay = 1e-4        # L2 regularization
    gradient_clip = 1.0        # Prevent gradient explosion
    
    # For slower convergence
    learning_rate = 5e-4       # Lower LR, more stable
    
    # For faster convergence (risky)
    learning_rate = 5e-3       # Higher LR, may diverge
```

### Training Duration

```python
class Config:
    num_epochs = 100           # Full training
    
    # For testing (quick run)
    num_epochs = 5             # Just 5 epochs to verify
```

### Loss Weights

```python
class Config:
    image_loss_weight = 1.0      # Image reconstruction priority
    caption_loss_weight = 1.0    # Caption generation priority
    alignment_loss_weight = 0.5  # Latent space alignment
    
    # More weight on images
    image_loss_weight = 2.0      # Better image quality
    caption_loss_weight = 0.5
    
    # More weight on captions
    image_loss_weight = 0.5
    caption_loss_weight = 2.0    # Better caption quality
```

---

## Running Training

### Basic Usage

```bash
cd ~/Documents/dashverse
python train_production.py
```

### Custom Configuration

Create a custom script:

```python
from train_production import Trainer, Config, ProductionTokenizer
from model_architecture_large import MultimodalModel, get_device
import torch
import json

# Load config
config = Config()

# Customize
config.batch_size = 32
config.num_epochs = 50
config.learning_rate = 5e-4

# ... rest of training code
```

---

## Expected Output

### Initialization Phase

```
STEP 1: Loading dataset...
Total captions found: 5141

STEP 2: Building tokenizer...
✓ Vocabulary built:
  Total tokens: 8000
  Unique words added: 7996
  Most common: [('an', 5141), ('engraving', 4892), ('depicting', 3456), ...]

STEP 3: Creating data loaders...
Train samples: 4627
Val samples: 514
Batches per epoch: 289  (4627 / 16)

STEP 4: Building model...
Total parameters: 108,708,035

STEP 5: Testing tokenization...
Original: An engraving depicting a woman in a blue dress
Encoded shape: torch.Size([100])
Decoded: an engraving depicting a woman in a blue dress

STEP 6: Starting training...
```

### Training Phase (First Few Epochs)

```
Epoch 1 Train: 
  Avg Loss: 2.4532 | Img: 0.8234 | Cap: 1.2341 | Align: 0.3957
Epoch 1 Val: 
  Avg Loss: 2.1823 | Img: 0.7234 | Cap: 1.1234 | Align: 0.3355
✓ Best model saved: checkpoints/best.pt
Epoch time: 45.2s | LR: 1.00e-03

Epoch 2 Train:
  Avg Loss: 2.1245 | Img: 0.7123 | Cap: 1.0834 | Align: 0.3288
Epoch 2 Val:
  Avg Loss: 1.9834 | Img: 0.6756 | Cap: 1.0123 | Align: 0.3055
✓ Best model saved: checkpoints/best.pt
Epoch time: 44.8s | LR: 9.98e-04
```

**Loss should generally decrease:**
- Image loss: 0.82 → 0.15
- Caption loss: 1.23 → 0.25
- Alignment: 0.39 → 0.05

---

## Checkpoint Files

After training:

```
checkpoints/
├── epoch_000.pt         # Epoch 0
├── epoch_001.pt         # Epoch 1
├── epoch_002.pt         # Epoch 2
├── ...
├── best.pt              # Best validation model ⭐
└── tokenizer.json       # Saved vocabulary
```

### Loading a Checkpoint

```python
checkpoint = torch.load("checkpoints/best.pt")
model.load_state_dict(checkpoint)
model.eval()
```

### Using Saved Tokenizer

```python
from train_production import ProductionTokenizer

tokenizer = ProductionTokenizer()
tokenizer.load("checkpoints/tokenizer.json")

# Now you can encode/decode
tokens = tokenizer.encode("Your caption here")
```

---

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
config.batch_size = 8

# Reduce model size (latent_dim)
config.latent_dim = 512  # Instead of 1024
```

### Training Loss Not Decreasing

```python
# Lower learning rate
config.learning_rate = 5e-4

# Increase gradient clip
config.gradient_clip = 0.5
```

### Tokenizer Not Built Error

```python
# Make sure metadata.json is in correct location
config.metadata_path = Path("data/metadata/engraving_metadata.json")

# Verify it has 'caption' field
```

### CUDA Out of Memory During Validation

```python
# Reduce batch size for validation
# Edit Trainer class:
val_loader = DataLoader(..., batch_size=8)  # Lower than train
```

---

## Next Steps After Training

1. **Inference Script** - Generate new image+caption pairs
2. **Evaluation Metrics** - FID, BLEU, alignment scores
3. **Gradio Demo** - Interactive web interface
4. **Model Deployment** - Export for production

---

## Summary of Changes

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| Batch Size | Fixed @ 16 | Configurable | ✅ |
| Tokenization | Dummy tokens | Real vocab-based | ✅ |
| Vocab Building | N/A | Auto from captions | ✅ |
| Token Encoding | N/A | Proper implementation | ✅ |
| Error Handling | Basic | Comprehensive | ✅ |
| Data Pipeline | Simple | 6-step structured | ✅ |
| Testing | No pre-training test | Pre-training validation | ✅ |

---

**All important notes resolved! Ready to train!** 🚀
