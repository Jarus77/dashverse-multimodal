# 📊 Complete W&B Training Guide - Everything You Need

## Files Ready for Download

| File | Purpose | Size |
|------|---------|------|
| **[train_wandb.py](computer:///mnt/user-data/outputs/train_wandb.py)** | ⭐ Main training script with W&B | 40 KB |
| **[WANDB_QUICKSTART.md](computer:///mnt/user-data/outputs/WANDB_QUICKSTART.md)** | 🚀 Get started in 5 minutes | 15 KB |
| **[WANDB_GUIDE.md](computer:///mnt/user-data/outputs/WANDB_GUIDE.md)** | 📚 Detailed W&B reference | 20 KB |
| **[model_architecture_large.py](computer:///mnt/user-data/outputs/model_architecture_large.py)** | 🧠 Model architecture (already have) | 21 KB |

---

## What You'll Log

### 📈 Real-Time Metrics (Updated Every Batch)

```
Loss Components:
  loss_total        → Sum of all losses
  loss_image        → Image reconstruction error (MSE)
  loss_caption      → Caption generation error (CrossEntropy)
  loss_alignment    → Latent space coherence loss

Latent Space:
  latent_mean       → Average value (should hover ~0)
  latent_std        → Standard deviation (should grow then stabilize)
  latent_min/max    → Min/max values
  latent_norm       → L2 norm of vectors

Image Quality:
  image_mse         → Mean squared error
  image_l1          → L1 distance
  image_diff        → Pixel-wise difference

Caption Quality:
  caption_accuracy  → Token prediction accuracy (% correct)
  caption_perplexity → Model uncertainty (exp of -log_prob)
  caption_entropy   → Information entropy

Gradients:
  grad_norm         → L2 norm (indicator of learning rate)
  grad_mean         → Average magnitude
  grad_max          → Largest gradient (spike detection)
  grad_std          → Gradient variance

Hardware:
  gpu_memory_used   → MB (should be stable)
  gpu_memory_percent → 0-100% (aim for 80-90%)
  cpu_percent       → CPU usage (aim for 20-40%)

Learning:
  learning_rate     → Current LR from scheduler
  batch_size        → Batch size used
  epoch             → Current epoch
```

---

## Setup Process (Step-by-Step)

### 1️⃣ Install Requirements

```bash
# All at once
pip install wandb psutil GPUtil
```

**Verify:**
```bash
python -c "import wandb, psutil, GPUtil; print('✓ All installed')"
```

### 2️⃣ Create W&B Account

1. Go to [wandb.ai/signup](https://wandb.ai/signup)
2. Sign up with email or GitHub
3. Verify email

### 3️⃣ Get API Key

1. Login to wandb.ai
2. Go to [wandb.ai/settings/profile](https://wandb.ai/settings/profile)
3. Find "API keys" section
4. Click "New key"
5. Copy the 40-character key

### 4️⃣ Login via CLI

```bash
wandb login
```

When prompted:
```
paste your API key: [paste 40-char key]
```

Result:
```
✓ Successfully authenticated
```

### 5️⃣ Download Files

Download these from `/mnt/user-data/outputs/`:
- train_wandb.py
- model_architecture_large.py (already have)

```bash
cd ~/Documents/dashverse
# Download the 2 files above
```

### 6️⃣ Run Training

```bash
python train_wandb.py
```

**First run will:**
1. Load 5,141 captions
2. Build vocabulary (1,266 tokens)
3. Create data loaders
4. Initialize model (85M parameters)
5. Setup W&B run
6. Start training!

### 7️⃣ Monitor Live

Open the URL shown in console:
```
✓ W&B run initialized: https://wandb.ai/YOUR_USERNAME/multimodal-engravings/runs/xxx
```

---

## What to Analyze

### ✅ Immediately After Training Starts

**Check (every 5 epochs):**
1. Is training loss decreasing? (YES → ✅)
2. Are gradients flowing? (grad_norm > 0 → ✅)
3. Is GPU being used? (gpu_memory_percent > 70% → ✅)

### ✅ After 20 Epochs

**Check:**
1. Loss decreased by 30%+? (e.g., 2.4 → 1.7 → ✅)
2. Validation loss similar to training? (Gap < 0.5 → ✅)
3. Latent space stabilized? (latent_std changing < 10% → ✅)

### ✅ After 50 Epochs

**Analyze:**
1. Which loss component dominates? (loss_image vs loss_caption)
2. Caption accuracy increasing? (% accuracy should increase)
3. Overfitting starting? (val_loss - train_loss > 1.0?)

### ✅ After 100 Epochs

**Final Analysis:**
1. Best validation loss achieved?
2. Gradient norm healthy (< 1.0)?
3. GPU memory efficient?
4. Any metrics show anomalies?

---

## Common Analysis Questions & How to Answer

### Q1: Is my learning rate correct?

**Check:**
- `grad_norm` trend in W&B
  - Too high LR → grad_norm spikes or explodes
  - Too low LR → loss barely decreases
  - Right LR → grad_norm decreases smoothly

**Action:**
- Spiky gradients? → Reduce LR: `learning_rate = 5e-4`
- Slow learning? → Increase LR: `learning_rate = 5e-3`

---

### Q2: Which part of the model is working?

**Check:**
- Image loss: 0.45 ✅ (working)
- Caption loss: 0.82 ❌ (struggling)
- Alignment: 0.12 ✅ (working)

**Conclusion:** Image encoder good, caption decoder needs work

**Action:**
- Increase `caption_loss_weight = 2.0`
- Run experiment again

---

### Q3: Is model overfitting?

**Check:**
```
W&B Charts:
Train loss: 0.50
Val loss: 0.75
Gap: 0.25 → Some overfitting

If gap > 1.0 → Severe overfitting!
```

**Action:**
- Increase `weight_decay` from 1e-4 → 1e-3
- Reduce batch size to add noise
- Add regularization

---

### Q4: Is GPU memory being used efficiently?

**Check:**
```
gpu_memory_percent = 45% → Underutilized! Can go bigger
gpu_memory_percent = 88% → Good!
gpu_memory_percent = 98% → Too risky
```

**Action:**
- 45%? → Increase batch_size: 16 → 32
- 98%? → Decrease batch_size: 16 → 8

---

### Q5: Did vocab building work?

**Check:**
```
Total tokens: 1266
Unique words: 1262
Most common: [('a', 7067), ('an', 5232), ...]

If Total tokens < 200 → Problem!
If Total tokens > 8000 → Using all available vocab
```

**Verdict:** 1,266 is perfect! ✅

---

## Comparing Multiple Runs

### Experiment: Different Batch Sizes

```bash
# Run 1: batch_size = 8
# Edit config.batch_size = 8
python train_wandb.py
# Wait for completion...

# Run 2: batch_size = 16
# Edit config.batch_size = 16
python train_wandb.py
# Wait for completion...

# Run 3: batch_size = 32
# Edit config.batch_size = 32
python train_wandb.py
```

### Analysis in W&B

1. Go to W&B project page
2. Click all 3 runs to select them
3. Click "Compare"
4. See side-by-side:
   - Final loss for each batch size
   - Training time per epoch
   - Peak GPU memory

### Finding Optimal Value

```
Batch Size vs Final Loss:
  8  → Loss: 0.89 (slow training)
  16 → Loss: 0.82 ✅ (best balance)
  32 → Loss: 0.80 (fast but risky)

Conclusion: Batch size 16 is optimal for your hardware!
```

---

## Dashboard Setup

### Create Custom Dashboard

1. In W&B project, click **+ Create** → **Custom Dashboard**
2. Add panels:

**Panel 1: Loss Curves**
```
Line chart
Y: loss_total, loss_image, loss_caption
X: step
```

**Panel 2: Latent Space Evolution**
```
Stacked area chart
Y: latent_mean, latent_std, latent_norm
X: step
```

**Panel 3: Training Health**
```
Line chart with dual Y-axis
Left Y: loss_total
Right Y: grad_norm
```

**Panel 4: Hardware Efficiency**
```
Line chart
Y: gpu_memory_percent, cpu_percent
X: step
```

---

## Exporting Results

### Download Metrics as CSV

1. In W&B dashboard
2. Right-click any chart
3. "Download → CSV"
4. Open in Excel/Python for analysis

### Create PDF Report

1. Click "Report" tab
2. Drag charts into report
3. Add markdown notes
4. Export as PDF
5. Share with team

### Example Report

```
# Multimodal Training Results

## Summary
- Final validation loss: 1.82
- Training time: 11.5 hours
- Best batch size: 16
- Optimal learning rate: 1e-3

## Key Metrics
- Image MSE: 0.15 ✅ (good)
- Caption accuracy: 87.2% ✅ (good)
- GPU utilization: 85% ✅ (efficient)

## Recommendations
1. Training is healthy - no anomalies detected
2. Consider next: fine-tune on specific engraving styles
3. Implement inference pipeline for production
```

---

## Troubleshooting

### Problem: "wandb: offline"

**Solution:**
```bash
wandb online
wandb login
python train_wandb.py
```

### Problem: "ImportError: No module named 'wandb'"

**Solution:**
```bash
pip install wandb
python -c "import wandb; print(wandb.__version__)"
```

### Problem: "API key invalid"

**Solution:**
```bash
# Re-login
wandb login --relogin
# Get new key from wandb.ai/settings/profile
```

### Problem: "No metrics appearing in dashboard"

**Solution:**
- Wait 60 seconds (first batch takes time)
- Refresh browser (Ctrl+R)
- Check internet connection
- Verify `run = wandb.init(...)` succeeded

### Problem: "Out of memory (OOM)"

**Solution:**
```python
class Config:
    batch_size = 8  # Reduce from 16
```

### Problem: "Training too slow"

**Solution:**
```python
class Config:
    num_workers = 8        # Increase from 4
    batch_size = 32        # Increase from 16
    log_frequency = 50     # Log less frequently
```

---

## Configuration Presets

### 🟢 Conservative (Stable)
```python
class Config:
    batch_size = 8
    learning_rate = 5e-4
    gradient_clip = 0.5
    # Result: Slow but very stable
```

### 🟡 Balanced (Recommended)
```python
class Config:
    batch_size = 16
    learning_rate = 1e-3
    gradient_clip = 1.0
    # Result: Good speed + stability (USE THIS)
```

### 🔴 Aggressive (Fast)
```python
class Config:
    batch_size = 32
    learning_rate = 5e-3
    gradient_clip = 2.0
    # Result: Fast but risky - may diverge!
```

---

## Key Metrics Reference

| Metric | Ideal Range | Problem If |
|--------|------------|-----------|
| loss_total | Decreasing | Increasing or flat |
| loss_image | < 0.5 | > 1.0 |
| loss_caption | < 0.3 | > 0.8 |
| loss_alignment | < 0.1 | > 0.5 |
| latent_std | 0.4 - 1.0 | < 0.2 or oscillating |
| grad_norm | 0.01 - 1.0 | Spikes or → 0 |
| image_mse | 0.05 - 0.20 | > 0.5 |
| caption_accuracy | > 80% | < 50% |
| gpu_memory_percent | 70% - 90% | > 98% (OOM risk) |
| cpu_percent | 20% - 40% | > 80% (bottleneck) |

---

## Next Steps After Training

1. ✅ **Save best model**: `checkpoints/best.pt`
2. ✅ **Export metrics**: CSV from W&B
3. ✅ **Create report**: Document findings
4. ✅ **Build inference**: Generate new image+caption pairs
5. ✅ **Deploy**: Package for production

---

## Quick Reference Commands

```bash
# Check W&B is working
wandb login

# Run training with W&B
python train_wandb.py

# Check GPU usage during training
nvidia-smi -l 1  # Update every 1 second

# View W&B dashboard
# Opens URL from console output

# Stop training
Ctrl+C (checkpoint still saved)
```

---

## Final Checklist

Before starting training:
- ✅ W&B account created
- ✅ API key configured (`wandb login`)
- ✅ Files downloaded
- ✅ Dependencies installed (`pip install wandb psutil GPUtil`)
- ✅ Config reviewed and adjusted if needed
- ✅ Data paths verified
- ✅ Model weights initialized

During training:
- ✅ Monitor W&B dashboard
- ✅ Check loss decreasing
- ✅ Verify gradients flowing
- ✅ Watch GPU memory

After training:
- ✅ Review final metrics
- ✅ Export data for analysis
- ✅ Compare with other runs
- ✅ Document findings
- ✅ Move best model to production

---

## Resources

- **W&B Docs:** https://docs.wandb.ai/
- **W&B Examples:** https://github.com/wandb/examples
- **PyTorch Docs:** https://pytorch.org/docs/
- **Weights & Biases FAQ:** https://docs.wandb.ai/guides/general/references/faq

---

**You're all set!** 🚀📊

Download files and run:
```bash
python train_wandb.py
```

Questions? Check WANDB_GUIDE.md or WANDB_QUICKSTART.md! 📚
