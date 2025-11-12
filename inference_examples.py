"""
INFERENCE PIPELINE - USAGE GUIDE & EXAMPLES

This script shows how to use the multimodal generator for image+caption generation.
"""

# ================================================================================
# EXAMPLE 1: Simple Usage (Single Generation)
# ================================================================================

def example_1_simple():
    """Generate a single image+caption from a seed"""
    from inference import MultimodalGenerator
    
    # Initialize generator
    generator = MultimodalGenerator(
        checkpoint_path="checkpoints/best.pt",
        tokenizer_path="checkpoints/tokenizer.json"
    )
    
    # Generate from seed
    image, caption = generator.generate(seed=42)
    
    # Display result
    print(f"Caption: {caption}")
    print(f"Image size: {image.size}")
    
    # Save image
    image.save("generated_image.png")
    
    return image, caption


# ================================================================================
# EXAMPLE 2: Batch Generation
# ================================================================================

def example_2_batch():
    """Generate multiple images+captions"""
    from inference import MultimodalGenerator
    from pathlib import Path
    import json
    
    # Initialize generator
    generator = MultimodalGenerator(
        checkpoint_path="checkpoints/best.pt",
        tokenizer_path="checkpoints/tokenizer.json"
    )
    
    # Generate batch
    seeds = [0, 1, 2, 3, 4, 5]
    results = generator.generate_batch(seeds)
    
    # Save results
    output_dir = Path("batch_output")
    output_dir.mkdir(exist_ok=True)
    
    metadata_list = []
    for image, caption, metadata in results:
        seed = metadata['seed']
        
        # Save image
        image.save(output_dir / f"image_seed_{seed}.png")
        
        # Collect metadata
        metadata_list.append({
            'seed': seed,
            'caption': caption
        })
    
    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"✓ Saved {len(results)} images to {output_dir}/")


# ================================================================================
# EXAMPLE 3: Interactive Loop
# ================================================================================

def example_3_interactive():
    """Interactive generation - generate from user-provided seeds"""
    from inference import MultimodalGenerator
    
    # Initialize once
    print("Loading generator (this may take a minute)...")
    generator = MultimodalGenerator(
        checkpoint_path="checkpoints/best.pt",
        tokenizer_path="checkpoints/tokenizer.json"
    )
    
    print("\n✓ Generator ready!\n")
    
    while True:
        try:
            # Get user input
            seed_input = input("Enter seed (0-10000) or 'quit' to exit: ").strip()
            
            if seed_input.lower() == 'quit':
                break
            
            seed = int(seed_input)
            
            # Generate
            print(f"\nGenerating from seed {seed}...")
            image, caption = generator.generate(seed=seed)
            
            # Display result
            print(f"\nCaption: {caption}")
            print(f"Image size: {image.size}\n")
            
            # Save option
            save = input("Save image? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"generated_seed_{seed}.png"
                image.save(filename)
                print(f"✓ Saved to {filename}\n")
        
        except ValueError:
            print("Invalid input. Please enter a number.\n")
        except Exception as e:
            print(f"Error: {e}\n")


# ================================================================================
# EXAMPLE 4: With Metadata
# ================================================================================

def example_4_with_metadata():
    """Generate with full metadata for analysis"""
    from inference import MultimodalGenerator
    import json
    from pathlib import Path
    
    # Initialize generator
    generator = MultimodalGenerator(
        checkpoint_path="checkpoints/best.pt",
        tokenizer_path="checkpoints/tokenizer.json"
    )
    
    # Generate with metadata
    image, caption, metadata = generator.generate_with_metadata(seed=123)
    
    print("Generated metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Save with metadata
    output_dir = Path("detailed_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save image
    image.save(output_dir / "image.png")
    
    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir}/")


# ================================================================================
# EXAMPLE 5: Programmatic API
# ================================================================================

def example_5_api():
    """Use as API for other applications"""
    from inference import MultimodalGenerator
    
    class ArtGeneratorAPI:
        def __init__(self):
            self.generator = MultimodalGenerator(
                checkpoint_path="checkpoints/best.pt",
                tokenizer_path="checkpoints/tokenizer.json"
            )
        
        def generate_art(self, seed: int):
            """Generate art from seed"""
            image, caption = self.generator.generate(seed=seed)
            
            # Return as dict for API response
            return {
                'status': 'success',
                'seed': seed,
                'caption': caption,
                'image': image  # PIL Image object
            }
    
    # Usage
    api = ArtGeneratorAPI()
    result = api.generate_art(seed=999)
    
    print(f"API Response:")
    print(f"  Status: {result['status']}")
    print(f"  Seed: {result['seed']}")
    print(f"  Caption: {result['caption']}")


# ================================================================================
# EXAMPLE 6: With Error Handling
# ================================================================================

def example_6_robust():
    """Robust usage with error handling"""
    from inference import MultimodalGenerator
    from pathlib import Path
    
    try:
        # Check if files exist
        if not Path("checkpoints/best.pt").exists():
            print("Error: checkpoints/best.pt not found")
            return
        
        if not Path("checkpoints/tokenizer.json").exists():
            print("Error: checkpoints/tokenizer.json not found")
            return
        
        # Initialize
        print("Initializing generator...")
        generator = MultimodalGenerator(
            checkpoint_path="checkpoints/best.pt",
            tokenizer_path="checkpoints/tokenizer.json"
        )
        
        # Generate multiple with error handling
        for seed in range(5):
            try:
                image, caption = generator.generate(seed=seed)
                image.save(f"output_{seed}.png")
                print(f"✓ Seed {seed}: {caption}")
            except Exception as e:
                print(f"✗ Seed {seed} failed: {e}")
    
    except Exception as e:
        print(f"Initialization failed: {e}")


# ================================================================================
# EXAMPLE 7: Memory-Efficient Batch
# ================================================================================

def example_7_memory_efficient():
    """Generate batch with memory optimization"""
    from inference import MultimodalGenerator
    from pathlib import Path
    import json
    
    generator = MultimodalGenerator(
        checkpoint_path="checkpoints/best.pt",
        tokenizer_path="checkpoints/tokenizer.json"
    )
    
    output_dir = Path("batch_memory_efficient")
    output_dir.mkdir(exist_ok=True)
    
    metadata = []
    
    # Process in small batches to save memory
    batch_size = 5
    total_seeds = 100
    
    for batch_start in range(0, total_seeds, batch_size):
        batch_end = min(batch_start + batch_size, total_seeds)
        seeds = list(range(batch_start, batch_end))
        
        print(f"Processing seeds {batch_start}-{batch_end-1}...")
        
        results = generator.generate_batch(seeds)
        
        for image, caption, meta in results:
            seed = meta['seed']
            image.save(output_dir / f"seed_{seed:05d}.png")
            metadata.append({'seed': seed, 'caption': caption})
    
    # Save all metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Generated {total_seeds} images")


# ================================================================================
# MAIN - Run Examples
# ================================================================================

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("MULTIMODAL GENERATOR - USAGE EXAMPLES")
    print("="*70)
    
    examples = {
        '1': ('Simple Usage', example_1_simple),
        '2': ('Batch Generation', example_2_batch),
        '3': ('Interactive Loop', example_3_interactive),
        '4': ('With Metadata', example_4_with_metadata),
        '5': ('Programmatic API', example_5_api),
        '6': ('Robust Usage', example_6_robust),
        '7': ('Memory Efficient', example_7_memory_efficient),
    }
    
    print("\nAvailable examples:\n")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print(f"  q. Quit\n")
    
    choice = input("Choose example (1-7): ").strip()
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}\n")
        try:
            func()
            print(f"\n✓ Example completed successfully")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    elif choice.lower() != 'q':
        print("Invalid choice")
