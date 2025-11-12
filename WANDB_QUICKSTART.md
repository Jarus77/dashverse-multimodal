# 🚀 Quick Start: W&B Training with Rigorous Analysis

## What You'll Get

A complete training pipeline that logs:
- ✅ Loss components (image, caption, alignment)
- ✅ Latent space statistics (mean, std, norm)
- ✅ Image reconstruction metrics (MSE, L1)
- ✅ Caption quality metrics (accuracy, perplexity, entropy)
- ✅ Gradient statistics (norm, mean, max)
- ✅ Hardware utilization (GPU memory, CPU)
- ✅ Learning rate schedule tracking
- ✅ Real-time visualizations

---

## Installation (First Time)

### Step 1: Install W&B

```bash
pip install wandb psutil GPUtil
```

### Step 2: Setup W&B Account

```bash
wandb login
# Follow prompts to create account and login
```

**You'll see:**
```
wandb: Logging into wandb.ai. (Learn how to run `wandb offline` to turn off wandb cloud.)
wandb: Go to https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter: [paste key here]
wandb: Successfully authenticated. Logged in as: your_username
```

### Step 3: Verify Setup

```bash
python -c "import wandb; print('✓ W&B ready')"
python -c "import psutil; print('✓ psutil ready')"
```

---

## Running Training

### 1. Copy Files

Download these files from outputs:
- `train_wandb.py` ⭐ (NEW - with W&B)
- `model_architecture_large.py` ✅ (Already have)

```bash
cd ~/Documents/dashverse
# Copy train_wandb.py here
```

### 2. Customize Config (Optional)

Edit `train_wandb.py`:

```python
class Config:
    batch_size = 16      # ← Adjust for your GPU
    num_epochs = 100     # ← Or 50 for quick test
    learning_rate = 1e-3
    # ... rest of config
```

### 3. Start Training

```bash
python train_wandb.py
```

**Expected Output:**

```
======================================================================
MULTIMODAL TRAINING WITH W&B LOGGING
======================================================================

Using GPU: NVIDIA H100 80GB HBM3

STEP 1: Loading dataset...
Total captions found: 5141

STEP 2: Building tokenizer...
Building vocabulary from 5141 captions...
✓ Vocabulary built:
  Total tokens: 1266
  Unique words added: 1262
  Most common: [('a', 7067), ('an', 5232), ('engraving', 5141), ...]

STEP 3: Creating data loaders...
Train samples: 4627
Val samples: 514
Batches per epoch: 289

STEP 4: Building model...
Total parameters: 85,000,000

STEP 5: Setting up W&B...
======================================================================
SETTING UP WEIGHTS & BIASES
======================================================================
✓ W&B login successful
✓ W&B run initialized: https://wandb.ai/your-username/multimodal-engravings/runs/xxx
✓ Model gradients being tracked

STEP 6: Starting training...

Epoch 1 Train: 100%|█████| 289/289 [05:23<00:00, 0.89 it/s]
Epoch 1 | Train Loss: 2.4532 | Val Loss: 2.1823 | Time: 325.5s | LR: 1.00e-03
```

**Copy the W&B URL and open in browser!**

---

## Monitoring in Real-Time

### 1. Open W&B Dashboard

Click the URL from console:
```
https://wandb.ai/your-username/multimodal-engravings/runs/xxx
```

### 2. Watch Metrics Update

Every 10 batches:
- Loss curves update
- Latent statistics show in real-time
- Gradient norms appear
- GPU usage displays

### 3. Refresh Browser

As training progresses, refresh to see:
- ✅ Training loss decreasing
- ✅ Validation loss improving
- ✅ Latent space stabilizing
- ✅ Gradients flowing properly

---

## Understanding the Dashboard

### Main Charts (Auto-created)

**Loss Components:**
```
loss_total       → Total loss (should decrease)
loss_image       → Image reconstruction error
loss_caption     → Caption generation error
loss_alignment   → Latent space coherence
```

**Latent Space:**
```
latent_mean      → Average latent value (hover ~0)
latent_std       → Spread of values (should grow initially)
latent_norm      → Magnitude of vectors (should be consistent)
```

**Image Quality:**
```
image_mse        → Mean squared error (should decrease)
image_l1         → L1 distance (should decrease)
```

**Caption Quality:**
```
caption_accuracy   → Token prediction accuracy (should increase)
caption_perplexity → Model confidence (should decrease)
caption_entropy    → Prediction uncertainty (should decrease)
```

**Gradients:**
```
grad_norm        → L2 norm of gradients (indicator of training health)
grad_mean        → Average gradient magnitude
grad_max         → Largest gradient (check for spikes)
```

**Hardware:**
```
gpu_memory_percent → GPU utilization (aim for 80-90%)
gpu_memory_used    → Memory usage in MB
cpu_percent        → CPU usage percentage
```

---

## Rigorous Analysis - Example Workflows

### Analysis 1: Is Training Healthy?

**What to check:**
1. **Loss curves:** Do they decrease smoothly?
   - If NO → learning rate too high/low
   - If YES → training is healthy ✅

2. **Gradient norm:** Is it stable?
   - If spikes → gradient explosion (reduce LR)
   - If → 0 → gradient vanishing (increase LR)
   - If stable → good ✅

3. **GPU memory:** Is it growing slowly?
   - If stable → normal ✅
   - If sudden spike → potential memory leak

**Action:** If not healthy, adjust learning rate and rerun.

---

### Analysis 2: Which Loss Dominates?

**What to check:**
```
Epoch 50 losses:
  loss_image    = 0.45  ← Image reconstruction working well
  loss_caption  = 0.82  ← Captions struggling
  loss_alignment = 0.12 ← Alignment good
```

**Conclusion:** Caption generation needs improvement

**Actions:**
- Increase `caption_loss_weight` from 1.0 → 2.0
- Add more caption-specific regularization
- Check tokenizer vocabulary coverage
- Run experiment with adjusted weights

---

### Analysis 3: Is Latent Space Learning?

**What to check:**
```
Epoch 1:   latent_std = 0.15  (low - initializing)
Epoch 10:  latent_std = 0.45  (growing - learning!)
Epoch 50:  latent_std = 0.62  (stabilizing - good!)
Epoch 100: latent_std = 0.63  (steady - converged!)
```

**Conclusion:** Latent space is learning properly ✅

**If instead you see:**
```
Epoch 1:   latent_std = 0.15
Epoch 50:  latent_std = 0.14  ← Not growing!
Epoch 100: latent_std = 0.14  ← PROBLEM!
```

**Actions:**
- Increase latent_dim from 1024 → 2048
- Check if encoder is learning (grad_norm > 0?)
- Verify loss is backpropagating properly

---

### Analysis 4: Overfitting Detection

**What to check:**
```
Epoch 10:
  train_loss = 1.23
  val_loss   = 1.19
  gap = 0.04 (small - good generalization!)

Epoch 50:
  train_loss = 0.45
  val_loss   = 0.89
  gap = 0.44 (large - OVERFITTING!)
```

**Actions:**
- Add regularization (increase weight_decay)
- Reduce model size (latent_dim: 1024 → 512)
- Use dropout (if implemented)
- Increase training data

---

### Analysis 5: Hardware Efficiency

**What to check:**
```
GPU Memory: 75% → Good! Room for larger batch
GPU Memory: 95% → Risky! Might OOM on noise

CPU Usage: 20% → Not data-loading bound
CPU Usage: 95% → Data loading is bottleneck (increase num_workers)
```

**Actions to optimize:**
```python
# More workers if CPU bound
config.num_workers = 8  # Instead of 4

# Larger batch if GPU underutilized
config.batch_size = 32  # Instead of 16
```

---

## Creating Custom Analysis Dashboard

### Example: Loss Component Stacking

In W&B:
1. Click **Create chart**
2. Select **Stacked area**
3. Y-axis metrics:
   - loss_image
   - loss_caption
   - loss_alignment
4. See which component dominates!

### Example: Correlation Analysis

In W&B:
1. Click **Analyze**
2. Scatter plot:
   - X: learning_rate
   - Y: final_loss
   - Color: batch_size
3. Discover optimal hyperparameters!

### Example: Hardware vs Performance

In W&B:
1. Create line chart with dual Y-axes:
   - Left Y: training_loss (primary)
   - Right Y: gpu_memory_percent (secondary)
2. Identify memory-performance tradeoffs

---

## Comparing Multiple Runs

### Run Multiple Experiments

```bash
# Run 1: Default config
python train_wandb.py

# Run 2: Different batch size
# (Edit config.batch_size = 32, then run)
python train_wandb.py

# Run 3: Different learning rate
# (Edit config.learning_rate = 5e-4, then run)
python train_wandb.py
```

### Compare in W&B

1. Go to project page
2. Click **Runs** tab
3. Select runs you want to compare
4. See side-by-side metrics!
5. Identify best configuration

---

## Exporting Results for Publication

### Export Data

In W&B dashboard:
1. Right-click any chart
2. Click **Download as CSV**
3. Use in Excel, Python, etc.

### Create Report

In W&B:
1. Click **Report**
2. Add charts and markdown text
3. Document findings
4. Share link with collaborators

### Example Report Content

```
# Multimodal Training Report

## Executive Summary
- Best validation loss: 1.82 (Batch size: 32, LR: 1e-3)
- Training time: 12 hours on H100
- Final accuracy: 85.3%

## Key Findings
1. Image loss dominated early training
2. Latent space stabilized after 30 epochs
3. Batch size 32 was optimal for this dataset

## Recommendations
- For production: use batch_size=64 on 2x H100s
- Further work: implement attention mechanisms
- Dataset: consider 10K+ samples for robustness
```

---

## Troubleshooting

### "wandb: offline"
W&B is in offline mode. To use cloud:
```bash
wandb online
wandb login
```

### "ConnectionError: Failed to upload"
Check internet connection or use offline mode:
```bash
wandb offline
python train_wandb.py
```

### "API key invalid"
Re-login:
```bash
wandb login --relogin
# Paste new API key
```

### "No metrics appearing"
Wait 30 seconds, then refresh browser. First batch takes time to upload.

---

## Performance Tips

### Optimize Logging

Reduce logging frequency for faster training:
```python
class WandBConfig:
    log_frequency = 50  # Log every 50 batches (instead of 10)
```

### Disable Expensive Logging

```python
class WandBConfig:
    track_gradients = False  # Can be expensive
    track_hardware = True    # Lightweight, keep on
```

### Save Bandwidth

```python
class WandBConfig:
    log_model = False  # Don't track model gradients
```

---

## Advanced: Custom Metrics

Add your own metrics to log:

```python
# In trainer.train_epoch():
custom_metrics = {
    'my_custom_metric': some_value,
    'latent_diversity': compute_diversity(latent),
    'caption_length_avg': avg_caption_length,
}

self.run.log(custom_metrics)
```

---

## Summary Checklist

- ✅ Install W&B and dependencies
- ✅ Login to W&B (`wandb login`)
- ✅ Download `train_wandb.py`
- ✅ Adjust config if needed
- ✅ Run: `python train_wandb.py`
- ✅ Open W&B dashboard URL
- ✅ Watch metrics in real-time
- ✅ Analyze results after training
- ✅ Compare runs and export findings

---

## Next Steps

1. **Train:** Let model run for 20 epochs to warm up
2. **Analyze:** Check if losses decrease properly
3. **Adjust:** If needed, tweak hyperparameters
4. **Compare:** Run with different configs
5. **Publish:** Export results and share with team

---

**Questions?** Check WANDB_GUIDE.md for detailed explanations! 📊🚀
