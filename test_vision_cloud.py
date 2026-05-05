#!/usr/bin/env python3
"""
Test vision engine (cloud Gemini API)
Run with: GEMINI_API_KEY=your-key python3 test_vision_cloud.py
"""

import os
import sys
from pathlib import Path

def test_api_key():
    """Test Gemini API key"""
    print("\n🔑 Testing API key...")
    try:
        from vision_engine import _load_api_key
        
        api_key = _load_api_key()
        print("   ✅ API key loaded")
        print(f"      Key begins with: {api_key[:10]}...")
        return True
    
    except RuntimeError as e:
        print(f"   ❌ API key failed: {e}")
        print()
        print("   Set API key with:")
        print("   export GEMINI_API_KEY=your-key-here")
        print("   OR create .env file with:")
        print("   GEMINI_API_KEY=your-key-here")
        return False

def test_model_candidates():
    """Test model candidate selection"""
    print("\n🤖 Testing model candidates...")
    try:
        from vision_engine import _candidate_model_names
        
        candidates = _candidate_model_names(None)
        print(f"   ✅ Model candidates loaded ({len(candidates)} total)")
        print(f"      Primary: {candidates[0]}")
        print(f"      Fallbacks: {candidates[1:3]}")
        print(f"      Last resort: {candidates[-1]}")
        return True
    
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_vision_analysis():
    """Test visual analysis with Gemini"""
    print("\n🖼️  Testing visual analysis...")
    
    try:
        from vision_engine import visual_summary_from_image
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
        
        # Add text labels
        draw.text((60, 60), "Input", fill='black')
        draw.text((270, 60), "Process", fill='black')
        draw.text((160, 160), "Output", fill='black')
        
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
        print("   Analyzing image with Gemini API...")
        print("   (this may take 10-20 seconds)")
        
        summary = visual_summary_from_image(
            str(test_img_path),
            temperature=0.2
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
    """Test text synthesis with Gemini"""
    print("\n📝 Testing text synthesis...")
    
    try:
        from vision_engine import video_summary_from_frame_summaries
        
        test_summaries = [
            "The first frame shows a diagram with three interconnected components: Input, Process, and Output boxes",
            "The second frame shows the same diagram with arrows indicating data flows from each component"
        ]
        
        print("   Synthesizing summary with Gemini API...")
        print("   (this may take 10-20 seconds)")
        
        result = video_summary_from_frame_summaries(
            test_summaries,
            frame_labels=["Frame 1", "Frame 2"]
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
    print("Testing Vision Engine (Cloud - Gemini API)")
    print("=" * 60)
    print()
    
    # Check API key first
    if not test_api_key():
        return 1
    
    results = {
        "Model Candidates": test_model_candidates(),
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
        print("\n✅ All cloud vision tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
