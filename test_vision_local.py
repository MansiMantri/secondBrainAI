#!/usr/bin/env python3
"""
Test vision engine (local Ollama)
Run with: python3 test_vision_local.py
"""

import os
import sys
from pathlib import Path

def test_ollama_connection():
    """Test Ollama connection"""
    print("\n🦙 Testing Ollama connection...")
    try:
        from vision_engine_local import check_ollama_models
        
        models = check_ollama_models()
        
        if models.get('error'):
            print(f"   ❌ Ollama error: {models['error']}")
            return False
        
        vision_models = models.get('vision', [])
        text_models = models.get('text', [])
        
        print("   ✅ Ollama connected successfully")
        print(f"   Vision models ({len(vision_models)}): {vision_models}")
        print(f"   Text models ({len(text_models)}): {text_models}")
        
        if not vision_models:
            print("   ⚠️  No vision models found. Install with: ollama pull llava:7b")
            return False
        
        if not text_models:
            print("   ⚠️  No text models found. Install with: ollama pull llama2:7b")
            return False
        
        return True
    
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_vision_analysis():
    """Test visual analysis"""
    print("\n🖼️  Testing visual analysis...")
    
    try:
        from vision_engine_local import visual_summary_from_image_local
    except ImportError as e:
        print(f"   ❌ Failed to import: {e}")
        return False
    
    # Create test image
    try:
        from PIL import Image, ImageDraw
        
        test_img_path = Path("test_diagram.png")
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple diagram
        draw.rectangle([50, 50, 150, 100], outline='blue', width=3)
        draw.rectangle([250, 50, 350, 100], outline='red', width=3)
        draw.line([(150, 75), (250, 75)], fill='black', width=2)
        draw.rectangle([150, 150, 250, 200], outline='green', width=3)
        draw.line([(100, 100), (175, 150)], fill='black', width=2)
        draw.line([(300, 100), (225, 150)], fill='black', width=2)
        
        img.save(test_img_path)
        print(f"   Created test diagram: {test_img_path}")
        
    except ImportError:
        print("   ⚠️  PIL not installed. Can't create test image.")
        return False
    except Exception as e:
        print(f"   ❌ Failed to create test image: {e}")
        return False
    
    # Test analysis
    try:
        print("   Analyzing image (this may take 30-60 seconds)...")
        summary = visual_summary_from_image_local(
            str(test_img_path),
            model_name='llava:7b'
        )
        
        if summary:
            print(f"   ✅ Analysis successful")
            print(f"      Summary (first 100 chars): {summary[:100]}...")
            test_img_path.unlink()
            return True
        else:
            print(f"   ❌ Empty result")
            test_img_path.unlink()
            return False
    
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        if test_img_path.exists():
            test_img_path.unlink()
        return False

def test_synthesis():
    """Test text synthesis"""
    print("\n📝 Testing text synthesis...")
    
    try:
        from vision_engine_local import video_summary_from_frame_summaries_local
        
        test_summaries = [
            "A diagram showing three interconnected components labeled Input, Process, and Output",
            "The same diagram with arrows showing data flow from Input to Process to Output"
        ]
        
        print("   Synthesizing summary (this may take 30-60 seconds)...")
        result = video_summary_from_frame_summaries_local(
            test_summaries,
            frame_labels=["Frame 1", "Frame 2"],
            model_name='llama2:7b'
        )
        
        if result:
            print(f"   ✅ Synthesis successful")
            print(f"      Result (first 100 chars): {result[:100]}...")
            return True
        else:
            print(f"   ❌ Empty result")
            return False
    
    except Exception as e:
        print(f"   ❌ Synthesis failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing Vision Engine (Local - Ollama)")
    print("=" * 60)
    print()
    
    results = {
        "Ollama Connection": test_ollama_connection(),
        "Visual Analysis": test_vision_analysis(),
        "Text Synthesis": test_synthesis()
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\n✅ All vision engine tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
