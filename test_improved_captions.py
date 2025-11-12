#!/usr/bin/env python3
"""
Quick test of improved caption generation
"""

from inference import MultimodalGenerator
import json
from pathlib import Path

print("="*70)
print("Testing Improved Caption Generation")
print("="*70)

# Initialize generator
print("\nInitializing generator...")
generator = MultimodalGenerator(
    checkpoint_path="checkpoints/best.pt",
    tokenizer_path="checkpoints/tokenizer.json"
)

# Generate samples
print("\nGenerating 10 samples with improved captions...")
results = generator.generate_batch(seeds=list(range(10)))

# Display results
print("\n" + "="*70)
print("Generated Samples:")
print("="*70)

metadata_list = []
for image, caption, metadata in results:
    seed = metadata['seed']
    print(f"\nSeed {seed:02d}: {caption}")
    
    # Save image
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    image.save(output_dir / f"seed_{seed:02d}.png")
    
    # Save metadata
    metadata_list.append({
        'seed': seed,
        'caption': caption
    })

# Save all metadata
with open("test_outputs/captions.json", 'w') as f:
    json.dump(metadata_list, f, indent=2)

print("\n" + "="*70)
print(f"✓ Generated 10 samples")
print(f"✓ Saved to test_outputs/")
print("="*70)
