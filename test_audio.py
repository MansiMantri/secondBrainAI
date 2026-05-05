#!/usr/bin/env python3
"""
Test audio engine - Whisper transcription
Run with: python3 test_audio.py
"""

import os
import sys
from pathlib import Path

def test_audio_engine():
    """Test audio transcription"""
    print("=" * 60)
    print("Testing Audio Engine (Whisper Transcription)")
    print("=" * 60)
    
    # Test 1: Check Whisper installation
    print("\n📦 Checking Whisper installation...")
    try:
        import whisper
        print("   ✅ Whisper library loaded")
        print("   Available models: tiny, base, small, medium, large")
    except ImportError as e:
        print(f"   ❌ Whisper not installed: {e}")
        print("   Install with: pip install openai-whisper")
        return False
    
    # Test 2: Check audio_engine module
    print("\n🔊 Loading audio_engine module...")
    try:
        from audio_engine import transcribe_audio
        print("   ✅ audio_engine module imported")
    except ImportError as e:
        print(f"   ❌ Failed to import: {e}")
        return False
    
    # Test 3: Create test audio file
    print("\n🎵 Creating test audio...")
    try:
        import numpy as np
        from scipy.io import wavfile
        
        # Create simple sine wave (A4 note)
        sample_rate = 16000
        duration = 1  # seconds
        frequency = 440  # Hz
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (32767 * 0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
        
        test_audio_path = Path("test_audio.wav")
        wavfile.write(str(test_audio_path), sample_rate, audio_data)
        print(f"   ✅ Test audio created: {test_audio_path}")
        
    except ImportError:
        print("   ⚠️  NumPy/SciPy not available. Using existing test_audio.wav if available.")
        test_audio_path = Path("test_audio.wav")
        if not test_audio_path.exists():
            print("   ⚠️  No test audio file found. Skipping transcription test.")
            return True
    except Exception as e:
        print(f"   ❌ Failed to create test audio: {e}")
        return False
    
    # Test 4: Test transcription
    print("\n🎙️  Testing transcription (using 'tiny' model for speed)...")
    try:
        print("   This may take 20-30 seconds...")
        result = transcribe_audio(str(test_audio_path), model_size='tiny')
        print(f"   ✅ Transcription successful")
        if result:
            print(f"      Result: '{result}'")
        else:
            print(f"      Result: (empty - expected for sine wave)")
        
        # Clean up
        test_audio_path.unlink()
        return True
        
    except Exception as e:
        print(f"   ❌ Transcription failed: {e}")
        # Clean up
        if test_audio_path.exists():
            test_audio_path.unlink()
        return False

def main():
    print()
    if test_audio_engine():
        print("\n✅ All audio engine tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
