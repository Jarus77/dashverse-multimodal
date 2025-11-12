# Weights & Biases (W&B) Setup Guide

## What is W&B?

**Weights & Biases** is a platform for:
- 🎯 Tracking experiments and metrics in real-time
- 📊 Creating custom dashboards and charts
- 💾 Versioning models and datasets
- 🔍 Analyzing hyperparameter impacts
- 👥 Collaborating with teams

**Perfect for rigorous parameter analysis!**

---

## Step 1: Install W&B

```bash
pip install wandb
```

Verify:
```bash
python -c "import wandb; print(wandb.__version__)"
```

---

## Step 2: Create W&B Account

1. Go to [wandb.ai](https://wandb.ai)
2. Click "Sign Up"
3. Create account (free tier is enough)
4. Verify email

---

## Step 3: Get API Key

1. Go to [wandb.ai/settings/profile](https://wandb.ai/settings/profile)
2. Scroll to "API keys"
3. Click "New key"
4. Copy the key

---

## Step 4: Login to W&B

In terminal:
```bash
wandb login
# Paste your API key when prompted
```

You'll see:
```
Successfully authenticated
```

**Done!** Now ready to log training.

---

## What We're Logging

### 📊 Loss Components
```
- total_loss      (sum of all losses)
- loss_image      (reconstruction error)
- loss_caption    (generation error)
- loss_alignment  (latent coherence)
```

### 🧠 Latent Space Analysis
```
- latent_mean     (average latent value)
- latent_std      (variability in latent space)
- latent_min/max  (range of values)
- latent_norm     (magnitude of latent vectors)
```

### 🖼️ Image Reconstruction
```
- image_mse       (Mean Squared Error)
- image_l1        (L1 distance)
- image_diff      (pixel-wise difference)
```

### 📝 Caption Quality
```
- caption_accuracy    (token prediction accuracy)
- caption_perplexity  (model confidence)
- caption_entropy     (prediction uncertainty)
```

### 🔍 Gradient Analysis
```
- grad_mean       (average gradient magnitude)
- grad_std        (gradient variance)
- grad_max        (largest gradient)
- grad_norm       (L2 norm of all gradients)
```

### 💻 Hardware Monitoring
```
- gpu_memory_used     (MB)
- gpu_memory_percent  (0-100%)
- cpu_percent         (CPU usage)
```

### 📈 Learning Dynamics
```
- learning_rate    (current LR from scheduler)
- batch_size       (batch size)
- epoch            (current epoch)
```

---

## Usage in Python

### Simple Usage

```python
# In your main() function:
if HAS_WANDB:
    import wandb
    wandb.login()
    run = wandb.init(
        project="multimodal-engravings",
        name="exp-1",
        config={
            "batch_size": 16,
            "learning_rate": 1e-3,
            "epochs": 100
        }
    )
else:
    run = None

# Then pass run to trainer
trainer = EnhancedTrainer(..., run=run)
trainer.train()
```

### Logging During Training

**Automatic:** Every batch (controlled by `WandBConfig.log_frequency`)

```python
# In trainer:
if batch_idx % 10 == 0:
    self.run.log({
        'loss': total_loss.item(),
        'learning_rate': lr,
        'epoch': epoch
    })
```

### Logging Images

```python
if batch_idx % 100 == 0:
    self.run.log({
        'original_image': wandb.Image(images[0].cpu()),
        'reconstructed_image': wandb.Image(outputs['image_recon'][0].cpu())
    })
```

---

## Understanding the Dashboard

### 1. Real-time Metrics
Click **Charts** → See live plots of:
- Loss over time
- Latent statistics
- Gradient norms
- Hardware usage

### 2. Hyperparameter Correlation
**Analyze** → **Scatter plots**
- X-axis: learning_rate
- Y-axis: final_loss
- See which parameters matter!

### 3. Compare Runs
Run multiple experiments → **Compare** tab
- Same config, different seeds?
- Different learning rates?
- See which combination works best!

### 4. Model Evolution
**Graphs** → Watch:
- Loss decreasing (good!)
- Latent space growing (expected!)
- Gradients flowing properly

---

## Advanced Analysis

### Config Dictionary to Log

```python
config_dict = {
    # Architecture
    "latent_dim": 1024,
    "vocab_size": 1266,
    "embedding_dim": 512,
    "image_encoder_layers": 5,
    "transformer_layers": 4,
    
    # Training
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealing",
    
    # Loss weights
    "image_loss_weight": 1.0,
    "caption_loss_weight": 1.0,
    "alignment_loss_weight": 0.5,
    
    # Data
    "train_samples": 4627,
    "val_samples": 514,
    "total_parameters": 85_000_000,
}

wandb.init(project="...", config=config_dict)
```

### Custom Metrics

```python
# After each epoch
metrics_to_log = {
    # Averages
    "epoch_train_loss_avg": train_loss,
    "epoch_val_loss_avg": val_loss,
    
    # Trends
    "gradient_norm_trend": grad_norm,
    "latent_std_trend": latent_std,
    
    # Ratios
    "val_train_loss_ratio": val_loss / train_loss,
    "image_caption_loss_ratio": img_loss / cap_loss,
}

wandb.log(metrics_to_log)
```

---

## Rigorously Analyzing Parameters

### 1. Loss Component Analysis

**Question:** Which loss component matters most?

**How to analyze:**
```
Go to W&B Dashboard
→ Charts
→ Create scatter: x=epoch, y=loss_image, y=loss_caption, y=loss_alignment
```

If `loss_caption` dominates, your captions need work.
If `loss_image` dominates, your image reconstruction needs work.

### 2. Latent Space Learning

**Question:** Is latent space learning properly?

**How to analyze:**
```
Charts:
- latent_std vs epoch (should grow initially, then stabilize)
- latent_norm vs epoch (should follow smooth trajectory)
- latent_mean vs epoch (should hover around 0)
```

### 3. Gradient Flow

**Question:** Are gradients flowing properly?

**How to analyze:**
```
Charts:
- grad_norm vs epoch (should decrease smoothly)
- grad_max vs epoch (check for spikes - gradient explosions?)
- grad_mean vs epoch (should be consistent)

If grad_norm suddenly spikes → gradient explosion
If grad_norm → 0 → gradient vanishing
```

### 4. Hyperparameter Sensitivity

**Question:** Which hyperparameter has biggest impact?

**How to analyze:**
```
Run 5 experiments with different configs:
- batch_size: 8, 16, 32, 64, 128
- learning_rate: 1e-4, 5e-4, 1e-3, 5e-3

Then in W&B:
Analyze → Scatter plot
x-axis: batch_size
y-axis: final_validation_loss

See which batch size gives best loss!
```

### 5. Overfitting Detection

**Question:** Is model overfitting?

**How to analyze:**
```
Charts:
- train_loss vs val_loss

If gap is growing → overfitting
If gap is shrinking → good regularization
```

### 6. Caption Quality Evolution

**Question:** Are captions getting better?

**How to analyze:**
```
Charts:
- caption_accuracy vs epoch (should increase)
- caption_perplexity vs epoch (should decrease)
- caption_entropy vs epoch (should decrease)
```

### 7. Hardware Efficiency

**Question:** Is GPU being used efficiently?

**How to analyze:**
```
Charts:
- gpu_memory_percent vs epoch
- gpu_memory_percent vs batch_size (run multiple batch sizes)

Aim: ~80-90% GPU utilization (not 100%, leaves room for stability)
```

---

## Using W&B for Debugging

### Gradient Clipping Effective?

```
Before: grad_max = 15.2  (high spikes)
After: grad_max = 1.0    (clamped nicely)
```

### Learning Rate Schedule Working?

```
Charts: learning_rate vs epoch
Should follow cosine annealing pattern ✓
```

### Batch Size Too Large?

```
If OOM errors:
- Log event in W&B
- See gpu_memory_percent spike
- Reduce batch_size + rerun
```

---

## Creating a Dashboard Summary

Create custom dashboard:

1. **Go to W&B project**
2. **Click "Create chart"**
3. **Add panels:**

```
Panel 1: Loss Curves
  - train_loss_avg (line chart)
  - val_loss_avg (line chart)
  
Panel 2: Components
  - loss_image
  - loss_caption
  - loss_alignment
  
Panel 3: Latent Space
  - latent_mean
  - latent_std
  - latent_norm
  
Panel 4: Gradients
  - grad_norm
  - grad_max
  
Panel 5: Hardware
  - gpu_memory_percent
  - cpu_percent
```

---

## Sharing Results

### 1. Share Dashboard Link
```
Copy URL from W&B project
Send to collaborators
They can see all logs, charts, comparisons
```

### 2. Export Data
```
In W&B:
Right-click chart → Download CSV
Analyze in Excel, Python, etc.
```

### 3. Create Report
```
In W&B:
Create report with text + charts
Add insights and conclusions
```

---

## Running Training with W&B

```bash
# Make sure logged in
wandb login

# Run training (auto-logs to W&B)
python train_wandb.py
```

**You'll see:**
```
Initializing wandb run...
✓ W&B run initialized: https://wandb.ai/your-username/multimodal-engravings/runs/xxx

Training started...
Logging every 10 batches to W&B
```

**Check dashboard:**
- Go to URL above
- Refresh every few seconds
- Watch metrics update in real-time!

---

## Tips for Rigorous Analysis

1. **Log everything:** More data = better insights
2. **Multiple runs:** Run same config 3x, see variance
3. **Systematic sweeps:** Change one param at a time
4. **Document findings:** Add notes to runs
5. **Save best config:** Remember what worked
6. **Share with team:** W&B makes collaboration easy

---

## Troubleshooting

### "wandb: ERROR Failed to initialize"

**Solution:**
```bash
wandb login  # Login again
wandb offline  # Or use offline mode for testing
```

### "ERROR: Failed to save run with error"

**Solution:**
```python
# Disable W&B temporarily
HAS_WANDB = False
# Or fix internet connection
```

### "How do I see my metrics?"

**Solution:**
1. Go to https://wandb.ai
2. Click your project
3. Select your run
4. Click "Charts" tab
5. Metrics appear automatically

---

## Next Steps

1. ✅ Install: `pip install wandb`
2. ✅ Login: `wandb login`
3. ✅ Run training: `python train_wandb.py`
4. ✅ Open W&B dashboard and watch!
5. ✅ Analyze results after training

**Happy experimenting!** 🚀📊
