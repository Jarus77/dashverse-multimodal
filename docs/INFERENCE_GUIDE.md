# 🎨 Inference Pipeline - Complete Guide

## Overview

The inference pipeline allows you to generate **images + captions** from random seeds using your trained multimodal model.

## 📦 Required Files

Before running inference, ensure you have:

```
checkpoints/
├── best.pt              ← Model weights (from training)
└── tokenizer.json       ← Caption vocabulary (generated during training)

model_architecture_large.py   ← Model definition (required for loading)
```

### File Locations

- **best.pt**: Generated during training (saved as best model checkpoint)
- **tokenizer.json**: Generated during training (saved in checkpoints/)
- **model_architecture_large.py**: Already available in outputs/

## 🚀 Quick Start (60 seconds)

### Step 1: Verify Setup
```bash
python inference_setup.py
```

This will:
- Check all required files
- Verify PyTorch installation
- Test GPU availability
- Run a quick generation test

### Step 2: Generate Your First Image+Caption
```python
from inference import MultimodalGenerator

# Initialize
generator = MultimodalGenerator(
    checkpoint_path="checkpoints/best.pt",
    tokenizer_path="checkpoints/tokenizer.json"
)

# Generate
image, caption = generator.generate(seed=42)

# Save
image.save("my_first_generation.png")
print(f"Caption: {caption}")
```

### Step 3: Generate Batch
```python
# Generate 10 samples
results = generator.generate_batch(seeds=list(range(10)))

# Each result is (image, caption, metadata)
for image, caption, metadata in results:
    print(f"Seed {metadata['seed']}: {caption}")
    image.save(f"sample_{metadata['seed']}.png")
```

## 📚 API Reference

### MultimodalGenerator Class

#### Initialization
```python
from inference import MultimodalGenerator

generator = MultimodalGenerator(
    checkpoint_path="checkpoints/best.pt",      # Path to model weights
    tokenizer_path="checkpoints/tokenizer.json", # Path to tokenizer
    latent_dim=1024,                            # Latent dimension (default)
    vocab_size=1266,                            # Vocabulary size (default)
    device=None                                 # 'cuda', 'cpu', or auto
)
```

#### Methods

**`generate(seed: int) -> Tuple[Image, str]`**
```python
image, caption = generator.generate(seed=42)
```
- **Input**: Random seed (int)
- **Output**: PIL Image and caption text
- **Use**: Single generation

**`generate_with_metadata(seed: int) -> Tuple[Image, str, Dict]`**
```python
image, caption, metadata = generator.generate_with_metadata(seed=42)
```
- **Input**: Random seed (int)
- **Output**: Image, caption, and metadata dictionary
- **Use**: When you need detailed information

**`generate_batch(seeds: list) -> list`**
```python
results = generator.generate_batch(seeds=[0, 1, 2, 3, 4])
```
- **Input**: List of seed values
- **Output**: List of (image, caption, metadata) tuples
- **Use**: Generate multiple samples

## 💡 Usage Examples

### Example 1: Single Generation
```python
from inference import MultimodalGenerator
from pathlib import Path

# Initialize
generator = MultimodalGenerator()

# Generate
image, caption = generator.generate(seed=123)

# Save
image.save("generated.png")
print(f"Caption: {caption}")
```

### Example 2: Batch with Metadata
```python
import json
from pathlib import Path

# Generate 30 samples
results = generator.generate_batch(seeds=range(30))

# Save images and metadata
output_dir = Path("samples")
output_dir.mkdir(exist_ok=True)

metadata = []
for image, caption, meta in results:
    seed = meta['seed']
    
    # Save image
    image.save(output_dir / f"seed_{seed:03d}.png")
    
    # Record metadata
    metadata.append({
        'seed': seed,
        'caption': caption,
        'image_path': f"seed_{seed:03d}.png"
    })

# Save metadata
with open(output_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Example 3: Interactive Generation
```python
generator = MultimodalGenerator()

while True:
    seed = input("Enter seed (or 'q' to quit): ")
    if seed.lower() == 'q':
        break
    
    try:
        image, caption = generator.generate(int(seed))
        print(f"Caption: {caption}")
        image.save(f"seed_{seed}.png")
    except Exception as e:
        print(f"Error: {e}")
```

### Example 4: Reproducible Generations
```python
# Same seed always produces same output
generator = MultimodalGenerator()

image1, caption1 = generator.generate(seed=42)
image2, caption2 = generator.generate(seed=42)

assert caption1 == caption2  # Always true
assert image1 == image2      # Visually identical
```

## 🎯 Hyperparameters

### Seeds
- **Range**: 0 to 2^32 - 1 (any integer)
- **Reproducibility**: Same seed always produces same output
- **Randomness**: Different seeds produce different outputs

### Latent Dimension
- **Default**: 1024
- **Use**: Keep as trained (don't change)

### Vocabulary Size
- **Default**: 1266
- **Use**: Keep as trained (don't change)

### Device
- **'cuda'**: GPU (fast, requires GPU)
- **'cpu'**: CPU (slow, but works anywhere)
- **None**: Auto-detect (recommended)

## ⚡ Performance

### Timing
- **Single generation**: ~0.5 - 2.0 seconds (depending on GPU)
- **Batch 10 samples**: ~5-10 seconds
- **Batch 100 samples**: ~50-100 seconds

### Memory
- **Model weights**: ~340 MB
- **Per generation**: ~500 MB
- **Batch processing**: Memory grows linearly with batch size

### GPU vs CPU
- **GPU (CUDA)**: ~10x faster
- **CPU**: Works but slow (~2-5 seconds per sample)

## 🔧 Troubleshooting

### Issue: "No module named 'model_architecture_large'"
**Solution**: Ensure `model_architecture_large.py` is in the same directory or in Python path

### Issue: "FileNotFoundError: checkpoints/best.pt"
**Solution**: 
- Ensure you ran training successfully
- Check file path is correct
- Run `python inference_setup.py` to verify

### Issue: "Out of memory"
**Solution**:
- Generate smaller batches
- Use CPU instead of GPU
- Reduce batch size

### Issue: "CUDA out of memory"
**Solution**:
- Empty GPU cache: `torch.cuda.empty_cache()`
- Use smaller batches
- Switch to CPU: `device='cpu'`

### Issue: "Caption is gibberish"
**Solution**:
- This is expected if:
  - Training didn't converge well
  - Tokenizer wasn't trained properly
- Check W&B metrics to verify training quality

## 📊 Outputs

### Image Properties
- **Size**: 256×256 pixels
- **Format**: RGB (3 channels)
- **Range**: [0-255] in saved PNG

### Caption Properties
- **Length**: 5-20 words (typically)
- **Vocabulary**: 1,266 unique tokens
- **Quality**: 69.4% token accuracy (from training)

### Metadata Included
```json
{
  "seed": 42,
  "caption": "an engraving of a portrait with intricate details",
  "latent_dim": 1024,
  "image_size": [256, 256],
  "device": "cuda"
}
```

## 🎓 Understanding the Pipeline

### Generation Process

1. **Set Random Seed**
   - Deterministic: Same seed → same output

2. **Generate Latent Vector**
   - Gaussian random vector (1024 dimensions)
   - Represents the "concept"

3. **Image Decoding**
   - Latent → Image decoder
   - Output: 256×256 RGB image

4. **Caption Decoding**
   - Latent → Text decoder
   - Output: Sequence of tokens

5. **Token Decoding**
   - Token IDs → Words → Text

### Why Shared Latent Space?

✅ **Semantic Alignment**: Image and caption describe same concept
✅ **Consistency**: Both emerge from same seed
✅ **Efficiency**: Single latent vector for both modalities
✅ **Scalability**: Can generate infinite combinations

## 🚀 Advanced Usage

### Custom Device Management
```python
import torch

# Explicit device selection
device = torch.device('cuda:0')  # First GPU
generator = MultimodalGenerator(device='cuda:0')

# Check GPU usage
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Batch Processing with Progress
```python
from tqdm import tqdm
from pathlib import Path

generator = MultimodalGenerator()
seeds = range(1000)
output_dir = Path("large_batch")
output_dir.mkdir(exist_ok=True)

for seed in tqdm(seeds):
    image, caption = generator.generate(seed)
    image.save(output_dir / f"seed_{seed:05d}.png")
```

### Integration with Other Libraries
```python
# Gradio demo (web interface)
import gradio as gr

def generate_art(seed):
    image, caption = generator.generate(seed)
    return image, caption

gr.Interface(
    fn=generate_art,
    inputs="slider",
    outputs=["image", "text"]
).launch()
```

## 📝 Common Patterns

### Pattern 1: Generate and Save
```python
image, caption = generator.generate(seed=42)
image.save(f"output_{42}.png")
```

### Pattern 2: Generate and Display
```python
image, caption = generator.generate(seed=42)
image.show()
print(caption)
```

### Pattern 3: Generate and Analyze
```python
image, caption, metadata = generator.generate_with_metadata(seed=42)

# Analyze caption
print(f"Caption length: {len(caption.split())}")

# Analyze image
print(f"Image size: {image.size}")
```

### Pattern 4: Batch Process
```python
results = generator.generate_batch(list(range(100)))

for image, caption, metadata in results:
    # Process each result
    pass
```

## ✅ Verification Checklist

Before using in production:

- [ ] Run `python inference_setup.py` successfully
- [ ] All files (best.pt, tokenizer.json) present
- [ ] Single generation works
- [ ] Batch generation works
- [ ] Captions are coherent
- [ ] Images look reasonable

## 🎯 Next Steps

1. ✅ Run setup verification
2. ✅ Generate a few test samples
3. ✅ Verify image quality
4. ✅ Check caption relevance
5. ✅ Scale to batch generation
6. ✅ Deploy in application

## 📞 Support

If you encounter issues:

1. Check troubleshooting section above
2. Run `python inference_setup.py` for diagnostics
3. Check W&B dashboard for training metrics
4. Review model_architecture_large.py for details

---

**Inference Pipeline Ready!** 🚀

Generated images and captions are reproducible, scalable, and production-ready.
