# 🚀 Quick Start Guide - Multimodal AI Training Pipeline

## ⚡ TL;DR - Get Started in 5 Minutes

```bash
# 1. Dataset preparation (already done ✅)
python dataset_preparation_v2.py

# 2. Generate captions (already done ✅)
python caption_generation.py

# 3. Train model (THIS IS NEXT)
python training_loop.py

# Done! Model saves to checkpoints/best_model.pt
```

---

## 📋 Checklist: What's Done

- ✅ **Step 1: Dataset Preparation**
  - 5,141 engravings downloaded & preprocessed
  - 512×512 resolution
  - Metadata created

- ✅ **Step 2: Caption Generation**
  - BLIP2 captions generated
  - Train/val split (90/10)
  - JSONL exported

- ✅ **Step 3: Model Architecture**
  - Large model (1024-dim latent)
  - 8,000 vocabulary optimized
  - Data loaders ready

- ✅ **Step 4: Training Loop**
  - Multi-task losses implemented
  - Checkpointing & early stopping
  - Ready to train!

---

## 🎯 Current Status: Ready to Train

Your data pipeline is complete. The model is ready to learn!

---

## 🏃 Step-by-Step Training

### 1. Navigate to Your Project
```bash
cd ~/Documents/dashverse
```

### 2. Run Training
```bash
python training_loop.py
```

### 3. Monitor Training
```
Epoch 1/50
  Train Loss: 2.1234
    - Reconstruction: 1.2345
    - Caption: 5.1234
    - Contrastive: 0.8234
  Val Loss: 1.9876

Epoch 2/50
  Train Loss: 1.8765
  ...

[Training continues...]

Epoch 40/50 ✅ BEST MODEL
  Val Loss: 0.2891
  → Saved to: checkpoints/best_model.pt
```

**Expected Duration**: ~4-5 hours on H100

### 4. Training Logs
Check logs in real-time:
```bash
tail -f checkpoints/training.log
```

---

## 📊 Understanding the Output

### Loss Components

**Total Loss = 1.0 × Recon + 2.0 × Caption + 0.5 × Contrastive**

| Loss | Meaning | Target |
|------|---------|--------|
| **Recon** | Image quality | ↓ Decrease → Sharper images |
| **Caption** | Text accuracy | ↓ Decrease → Better captions |
| **Contrastive** | Latent organization | ↓ Decrease → Better alignment |

### Healthy Training Signs ✅
- All losses decrease consistently
- No NaN or infinity values
- Val loss follows train loss (slight lag ok)
- By epoch 20: Noticeable improvement

### Warning Signs ⚠️
- Losses stay flat (learning rate too low)
- Loss spikes (gradient explosion)
- Val loss diverges from train (overfitting)

---

## 💾 Checkpoints

After training, you'll have:

```
checkpoints/
├── checkpoint_epoch_005.pt    (every 5 epochs)
├── checkpoint_epoch_010.pt
├── checkpoint_epoch_015.pt
├── checkpoint_epoch_020.pt
├── ...
└── best_model.pt              ⭐ Use this for inference
```

**To resume training from checkpoint:**
```python
trainer.load_checkpoint("checkpoints/checkpoint_epoch_020.pt")
trainer.fit(train_loader, val_loader, num_epochs=50)
```

---

## 🔍 Troubleshooting

### Issue: GPU Out of Memory
**Solution:**
```python
# In training_loop.py, change:
batch_size=8  →  batch_size=4
```

### Issue: Slow Training
**Solution:**
```bash
# Check GPU utilization
nvidia-smi watch

# Make sure H100 is being used (not CPU)
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Training Hangs
**Solution:**
```bash
# Kill and restart
# Training script saves checkpoints, so you won't lose progress
pkill -f training_loop.py
python training_loop.py  # Resumes from best checkpoint
```

### Issue: Bad Loss Values
**Check:**
1. Data loading: `images = 2 * images - 1` (normalized to [-1, 1])
2. Tokenization: Captions → token indices
3. Model device: GPU vs CPU mismatch

---

## 📈 Expected Timeline

| Time | Milestone |
|------|-----------|
| **Epoch 1** | High loss (~2.0), random tokens |
| **Epoch 5** | Loss decreasing (~1.2) |
| **Epoch 10** | Better images/captions (~0.8) |
| **Epoch 20** | Good progress (~0.45) |
| **Epoch 30** | Strong results (~0.35) |
| **Epoch 40** | Excellent results (~0.29) ⭐ |
| **Epoch 50** | Final (~0.28) |

---

## 🎨 What Happens During Training

### Model Learning
```
Random Weights (bad outputs)
        ↓ Epoch 1-5
Poor Reconstruction (blurry)
        ↓ Epoch 5-15
Getting Better (recognizable)
        ↓ Epoch 15-30
Good Quality Images + Captions
        ↓ Epoch 30-40
Excellent Coherence ✅
        ↓ Epoch 40+
Fine-tuning (marginal gains)
```

### Latent Space Development
```
Epoch 1:  Random, no structure
Epoch 10: Partial clustering
Epoch 20: Clear semantic regions
Epoch 40: Well-organized space ✅
```

---

## 🔐 Saving Training Progress

Training script **automatically saves**:
- Checkpoint every 5 epochs
- Best model when validation improves
- Metrics to JSON

**Manual save:**
```python
trainer.save_checkpoint(epoch=25, is_best=True)
```

---

## 📊 Monitoring Training

### Option 1: Live Console Output
```bash
python training_loop.py 2>&1 | tee training.log
```

### Option 2: Tensorboard (add to code)
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('Loss/train', loss, epoch)
writer.close()
```

### Option 3: Weights & Biases (optional)
```python
# In training_loop.py, change:
use_wandb=False  →  use_wandb=True

# Then view at: https://wandb.ai
```

---

## 🚀 After Training: What's Next?

Once you have `best_model.pt`, you can:

### 1. Generate Images + Captions
```python
import torch
from model_architecture import MultimodalModel, SimpleTokenizer

model = MultimodalModel(...)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))

# Generate from random seed
seed = torch.randn(1, 1024)
image = model.decode_image(seed)
caption = model.decode_text(seed)
```

### 2. Build Interactive Demo (Gradio)
```python
import gradio as gr

def generate(seed_value):
    # Generate image + caption
    return image, caption_text

gr.Interface(generate, 
             inputs=gr.Number(label="Seed"),
             outputs=[gr.Image(), gr.Textbox()]).launch()
```

### 3. Evaluate Results
- Visual inspection of generated images
- Caption quality assessment
- Semantic coherence scoring

---

## 📚 Key Files

| File | Purpose | Status |
|------|---------|--------|
| `dataset_preparation_v2.py` | Download & process images | ✅ Done |
| `caption_generation.py` | Generate captions with BLIP2 | ✅ Done |
| `model_architecture.py` | Define model & data loaders | ✅ Ready |
| `training_loop.py` | Training engine | ✅ Ready to run |
| `checkpoints/best_model.pt` | Trained model | ⏳ After training |
| `inference.py` | Generate from seed | 📝 Next |
| `gradio_demo.py` | Web interface | 📝 Next |

---

## 💡 Pro Tips

### Tip 1: Start Fresh or Resume?
```python
# Start fresh (delete old checkpoints)
import shutil
shutil.rmtree("checkpoints")

# OR resume from best model
trainer.load_checkpoint("checkpoints/best_model.pt")
```

### Tip 2: Adjust Loss Weights
```python
# In training_loop.py, MultimodalLoss init:
self.lambda_recon = 1.0        # Image quality
self.lambda_caption = 2.0      # Caption quality (default 2x)
self.lambda_contrastive = 0.5  # Latent alignment
```

### Tip 3: Shorter Testing Run
```python
# Test with subset of data
trainer.fit(train_loader, val_loader, 
            num_epochs=3,  # Just 3 epochs
            patience=1)    # Early stop quickly
```

---

## ✅ Verification Checklist

Before running training, verify:

- [ ] Python >= 3.10
- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Data exists: `ls data/processed/engraving/resized/ | wc -l` (should be ~5141)
- [ ] Metadata exists: `ls data/metadata/engraving_metadata.json`
- [ ] PyTorch installed: `python -c "import torch; print(torch.version)"`
- [ ] 80GB+ VRAM available on H100
- [ ] ~2-3 hours free time for training

---

## 🎯 Success Criteria

Your training is successful if:

✅ Losses decrease consistently over epochs
✅ By epoch 30, total loss < 0.5
✅ Generated images look like engravings (recognizable)
✅ Captions contain meaningful words
✅ No NaN or infinity in losses
✅ Model converges before patience limit

---

## 📞 When Stuck

1. **Check losses**: Are they decreasing?
2. **Check GPU**: `nvidia-smi` (should show >10GB usage)
3. **Check data**: Load one batch manually and inspect
4. **Check logs**: `tail -50 training.log`
5. **Restart**: Kill process and retry from checkpoint

---

## 🚀 Ready?

```bash
cd ~/Documents/dashverse
python training_loop.py
```

**Good luck! Your H100 will crush this in 4-5 hours!** 🔥

---

## 📞 Quick Reference Commands

```bash
# Check GPU
nvidia-smi

# Monitor live
nvidia-smi dmon -s pucm

# Check data
python -c "import torch; from model_architecture import create_data_loaders; create_data_loaders('data/metadata/engraving_metadata.json', 'data/processed/engraving/resized')"

# Restart training
pkill -f training_loop.py
python training_loop.py

# View best model
ls -lh checkpoints/best_model.pt
```

---

**Last Updated**: November 10, 2025
**Status**: Ready to Train 🚀
