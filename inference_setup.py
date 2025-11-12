"""
Setup & Verification Script for Inference Pipeline
Ensures all required files are in place
"""

import json
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================================
# VERIFICATION
# ================================================================================

def verify_setup():
    """Verify all required files and dependencies"""
    
    logger.info("="*70)
    logger.info("INFERENCE PIPELINE SETUP VERIFICATION")
    logger.info("="*70)
    
    all_good = True
    
    # 1. Check model architecture
    logger.info("\n1. Checking Model Architecture...")
    if Path("model_architecture_large.py").exists():
        logger.info("   ✓ model_architecture_large.py found")
    else:
        logger.error("   ✗ model_architecture_large.py NOT found!")
        all_good = False
    
    # 2. Check checkpoint directory
    logger.info("\n2. Checking Checkpoint Directory...")
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        logger.info(f"   ✓ checkpoints/ directory exists")
        
        # Check for best.pt
        if (checkpoint_dir / "best.pt").exists():
            size_mb = (checkpoint_dir / "best.pt").stat().st_size / (1024**2)
            logger.info(f"   ✓ best.pt found ({size_mb:.1f} MB)")
        else:
            logger.error("   ✗ best.pt NOT found!")
            all_good = False
        
        # Check for tokenizer.json
        if (checkpoint_dir / "tokenizer.json").exists():
            logger.info(f"   ✓ tokenizer.json found")
            
            # Verify tokenizer
            try:
                with open(checkpoint_dir / "tokenizer.json", 'r') as f:
                    tokenizer_data = json.load(f)
                
                vocab_size = len(tokenizer_data['word2idx'])
                logger.info(f"     - Vocabulary size: {vocab_size}")
                
                if vocab_size > 1000:
                    logger.info(f"     ✓ Vocabulary size looks good")
                else:
                    logger.warning(f"     ⚠️  Vocabulary size seems small ({vocab_size})")
            except Exception as e:
                logger.error(f"   ✗ Error reading tokenizer: {e}")
                all_good = False
        else:
            logger.error("   ✗ tokenizer.json NOT found!")
            all_good = False
    else:
        logger.error("   ✗ checkpoints/ directory NOT found!")
        all_good = False
    
    # 3. Check inference.py
    logger.info("\n3. Checking Inference Script...")
    if Path("inference.py").exists():
        logger.info("   ✓ inference.py found")
    else:
        logger.error("   ✗ inference.py NOT found!")
        all_good = False
    
    # 4. Check PyTorch and CUDA
    logger.info("\n4. Checking PyTorch Setup...")
    logger.info(f"   PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"   ✓ CUDA available")
        logger.info(f"     - GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"     - CUDA version: {torch.version.cuda}")
    else:
        logger.warning("   ⚠️  CUDA not available (will use CPU - slower)")
    
    # 5. Create output directory
    logger.info("\n5. Creating Output Directory...")
    output_dir = Path("inference_outputs")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"   ✓ Created: {output_dir}/")
    
    # Summary
    logger.info("\n" + "="*70)
    if all_good:
        logger.info("✅ ALL CHECKS PASSED - Ready for inference!")
    else:
        logger.error("❌ SOME CHECKS FAILED - See above for details")
    logger.info("="*70 + "\n")
    
    return all_good


# ================================================================================
# QUICK TEST
# ================================================================================

def quick_test():
    """Quick test of inference pipeline"""
    
    logger.info("="*70)
    logger.info("QUICK INFERENCE TEST")
    logger.info("="*70)
    
    try:
        # Import generator
        from inference import MultimodalGenerator
        
        logger.info("\n1. Loading generator...")
        generator = MultimodalGenerator(
            checkpoint_path="checkpoints/best.pt",
            tokenizer_path="checkpoints/tokenizer.json"
        )
        
        logger.info("\n2. Generating sample...")
        image, caption = generator.generate(seed=42)
        
        logger.info(f"\n✓ Generation successful!")
        logger.info(f"  Image size: {image.size}")
        logger.info(f"  Caption: '{caption}'")
        
        # Save test output
        output_dir = Path("inference_outputs")
        test_path = output_dir / "test_sample.png"
        image.save(test_path)
        logger.info(f"\n✓ Saved test image to: {test_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    
    # Run verification
    verified = verify_setup()
    
    if verified:
        # Run quick test
        logger.info("\nRunning quick test...\n")
        success = quick_test()
        
        if success:
            logger.info("\n" + "="*70)
            logger.info("🎉 SETUP COMPLETE! Ready to use inference pipeline")
            logger.info("="*70)
            logger.info("\nUsage examples:\n")
            logger.info("from inference import MultimodalGenerator\n")
            logger.info("# Initialize")
            logger.info("generator = MultimodalGenerator()")
            logger.info("  \n# Generate single sample")
            logger.info("image, caption = generator.generate(seed=42)")
            logger.info("\n# Generate batch")
            logger.info("results = generator.generate_batch(seeds=[0,1,2,3,4])")
            logger.info("\nRun: python inference.py\n")
    else:
        logger.error("\nFix the issues above before proceeding")
