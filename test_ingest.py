#!/usr/bin/env python3
"""
Test ingest logic - media conversion pipeline
Run with: python3 test_ingest.py
"""

import os
import sys
from pathlib import Path

def test_ingest_logic():
    """Test media conversion components"""
    print("=" * 60)
    print("Testing Ingest Logic (Media Conversion)")
    print("=" * 60)
    
    try:
        from ingest_logic import image_to_png, VIDEO_EXTS, IMAGE_EXTS
        print("✅ Ingest logic module imported")
    except ImportError as e:
        print(f"❌ Failed to import ingest_logic: {e}")
        return False
    
    # Test 1: Check supported formats
    print("\n📋 Checking supported formats...")
    print(f"   Image formats: {IMAGE_EXTS}")
    print(f"   Video formats: {VIDEO_EXTS}")
    
    # Test 2: Image to PNG conversion
    print("\n🖼️  Testing image_to_png()...")
    
    # Create test image
    try:
        from PIL import Image, ImageDraw
        
        test_img_path = Path("test_diagram.png")
        img = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw simple shapes
        draw.rectangle([50, 50, 150, 100], outline='blue', width=2)
        draw.rectangle([170, 50, 270, 100], outline='red', width=2)
        draw.text((60, 110), "Test Image", fill='black')
        
        img.save(test_img_path)
        print(f"   Created test image: {test_img_path}")
        
        # Test conversion
        output = image_to_png(test_img_path, 'test_output')
        print(f"   ✅ Conversion successful")
        print(f"      Output: {output[0]}")
        
        # Clean up test file
        test_img_path.unlink()
        
        return True
        
    except ImportError:
        print("   ⚠️  PIL not installed. Install with: pip install Pillow")
        return False
    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        return False

def main():
    print()
    if test_ingest_logic():
        print("\n✅ All ingest logic tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
