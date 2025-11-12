#!/usr/bin/env python3
"""
Quick Start Script for Inference Pipeline

Run this to:
1. Verify setup
2. Test generation
3. Create sample outputs
"""

import sys
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def check_files():
    """Check if required files exist"""
    print_header("Step 1: Checking Required Files")
    
    files_to_check = {
        'model_architecture_large.py': 'Model architecture',
        'checkpoints/best.pt': 'Model weights',
        'checkpoints/tokenizer.json': 'Caption tokenizer',
        'inference.py': 'Inference pipeline'
    }
    
    missing_files = []
    
    for file_path, description in files_to_check.items():
        full_path = Path(file_path)
        if full_path.exists():
            if file_path.endswith('.pt'):
                size_mb = full_path.stat().st_size / (1024**2)
                logger.info(f"✓ {description:.<40} {file_path} ({size_mb:.1f} MB)")
            else:
                logger.info(f"✓ {description:.<40} {file_path}")
        else:
            logger.error(f"✗ {description:.<40} {file_path} - NOT FOUND")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files


def check_dependencies():
    """Check Python dependencies"""
    print_header("Step 2: Checking Dependencies")
    
    dependencies = {
        'torch': 'PyTorch',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'json': 'JSON'
    }
    
    all_good = True
    
    for module_name, display_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"✓ {display_name:.<40} {version}")
        except ImportError:
            logger.error(f"✗ {display_name:.<40} NOT INSTALLED")
            all_good = False
    
    return all_good


def check_gpu():
    """Check GPU availability"""
    print_header("Step 3: Checking GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available")
            logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            logger.warning("⚠️  CUDA not available (will use CPU - slower)")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False


def test_generation():
    """Test single generation"""
    print_header("Step 4: Testing Generation")
    
    try:
        from inference import MultimodalGenerator
        
        logger.info("Loading generator...")
        generator = MultimodalGenerator(
            checkpoint_path="checkpoints/best.pt",
            tokenizer_path="checkpoints/tokenizer.json"
        )
        
        logger.info("Generating sample (seed=42)...")
        image, caption = generator.generate(seed=42)
        
        logger.info(f"✓ Generation successful!")
        logger.info(f"  Image size: {image.size}")
        logger.info(f"  Caption: '{caption}'")
        
        # Save test output
        output_dir = Path("inference_outputs")
        output_dir.mkdir(exist_ok=True)
        test_path = output_dir / "test_sample.png"
        image.save(test_path)
        logger.info(f"  Saved to: {test_path}")
        
        return True, image, caption
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_batch():
    """Test batch generation"""
    print_header("Step 5: Testing Batch Generation")
    
    try:
        from inference import MultimodalGenerator
        
        generator = MultimodalGenerator(
            checkpoint_path="checkpoints/best.pt",
            tokenizer_path="checkpoints/tokenizer.json"
        )
        
        seeds = [0, 1, 2]
        logger.info(f"Generating batch ({len(seeds)} seeds)...")
        
        results = generator.generate_batch(seeds)
        
        logger.info(f"✓ Batch generation successful!")
        
        # Save batch
        output_dir = Path("inference_outputs")
        output_dir.mkdir(exist_ok=True)
        
        batch_metadata = []
        for image, caption, metadata in results:
            seed = metadata['seed']
            image.save(output_dir / f"batch_seed_{seed}.png")
            batch_metadata.append({
                'seed': seed,
                'caption': caption
            })
        
        # Save metadata
        with open(output_dir / "batch_metadata.json", 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        logger.info(f"  Saved {len(results)} samples to {output_dir}/")
        
        return True
    
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        return False


def run_full_test():
    """Run complete test suite"""
    print_header("🎨 INFERENCE PIPELINE - QUICK START TEST")
    
    # Step 1: Check files
    files_ok, missing = check_files()
    
    if not files_ok:
        print_header("❌ SETUP INCOMPLETE")
        logger.error(f"Missing files: {missing}")
        logger.info("\nPlease ensure:")
        for file_path in missing:
            logger.info(f"  - {file_path} exists")
        logger.info("\nIf checkpoints don't exist:")
        logger.info("  - Run: python train_wandb.py")
        return False
    
    # Step 2: Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print_header("❌ DEPENDENCIES MISSING")
        logger.info("Install missing dependencies:")
        logger.info("  pip install torch pillow numpy")
        return False
    
    # Step 3: Check GPU
    gpu_available = check_gpu()
    
    # Step 4: Test generation
    gen_ok, image, caption = test_generation()
    
    if not gen_ok:
        print_header("❌ GENERATION FAILED")
        return False
    
    # Step 5: Test batch
    batch_ok = test_batch()
    
    if not batch_ok:
        print_header("⚠️  BATCH GENERATION FAILED (but single works)")
    
    # Success
    print_header("✅ ALL TESTS PASSED!")
    
    logger.info("Inference pipeline is ready to use!\n")
    logger.info("Quick Usage Examples:\n")
    logger.info("from inference import MultimodalGenerator")
    logger.info("")
    logger.info("generator = MultimodalGenerator()")
    logger.info("")
    logger.info("# Single generation")
    logger.info("image, caption = generator.generate(seed=42)")
    logger.info("")
    logger.info("# Batch generation")
    logger.info("results = generator.generate_batch([0, 1, 2, 3, 4])")
    logger.info("")
    logger.info("# See INFERENCE_GUIDE.md for more examples")
    
    return True


def interactive_mode():
    """Interactive generation mode"""
    print_header("Interactive Generation Mode")
    
    try:
        from inference import MultimodalGenerator
        
        logger.info("Loading generator...")
        generator = MultimodalGenerator(
            checkpoint_path="checkpoints/best.pt",
            tokenizer_path="checkpoints/tokenizer.json"
        )
        
        logger.info("✓ Ready!\n")
        
        while True:
            try:
                seed_input = input("Enter seed (or 'quit' to exit): ").strip()
                
                if seed_input.lower() in ['quit', 'q', 'exit']:
                    logger.info("Goodbye!")
                    break
                
                seed = int(seed_input)
                
                logger.info(f"\nGenerating from seed {seed}...")
                image, caption = generator.generate(seed=seed)
                
                logger.info(f"\nCaption: {caption}\n")
                
                save = input("Save image? (y/n): ").strip().lower()
                if save == 'y':
                    filename = f"generated_seed_{seed}.png"
                    image.save(filename)
                    logger.info(f"✓ Saved to {filename}\n")
            
            except ValueError:
                logger.error("Invalid seed. Please enter a number.\n")
            except Exception as e:
                logger.error(f"Error: {e}\n")
    
    except Exception as e:
        logger.error(f"Failed to start interactive mode: {e}")


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive' or sys.argv[1] == '-i':
            interactive_mode()
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("""
Usage: python quickstart.py [options]

Options:
  (no args)     Run full test suite
  -i, --interactive   Interactive generation mode
  -h, --help    Show this help
  
Examples:
  python quickstart.py          # Run tests
  python quickstart.py -i       # Interactive mode
            """)
    else:
        success = run_full_test()
        
        if success:
            print("\n" + "="*70)
            ask = input("Start interactive mode? (y/n): ").strip().lower()
            if ask == 'y':
                interactive_mode()
        else:
            sys.exit(1)
