# SecondBrainAI Component Testing Guide

Complete testing procedures for validating all components in SecondBrainAI (both local and cloud versions).

---

## 📋 Testing Overview

| Component | Test Type | Command |
|-----------|-----------|---------|
| **Ollama Setup** | Dependency | `ollama list` |
| **Vision Engine (Local)** | Module | `python3 test_vision_local.py` |
| **Vision Engine (Cloud)** | Module | `python3 test_vision_cloud.py` |
| **Audio Engine** | Module | `python3 test_audio.py` |
| **Ingest Logic** | Module | `python3 test_ingest.py` |
| **Full Pipeline (Local)** | Integration | `streamlit run app_local.py` |
| **Full Pipeline (Cloud)** | Integration | `streamlit run app.py` |

---

## 🔧 Pre-Testing Setup

### 1. Verify Environment
```bash
# Activate virtual environment (if using one)
source secondbrain_env/bin/activate

# Check Python version
python3 --version  # Should be 3.8+

# Check required tools installed
ollama --version
ffmpeg -version
```

### 2. Ensure Ollama is Running
```bash
# Start Ollama service (if not already running)
ollama serve

# In another terminal, verify models are available
ollama list

# Expected output:
# NAME            ID              SIZE      MODIFIED
# llava:7b        8dd30f6b0cb1    4.7 GB    2 hours ago
# llama2:7b       78e26419b446    3.8 GB    1 hour ago
```

---

## 📁 Create Test Data

### Create Sample Image (for vision testing)
```bash
python3 << 'EOF'
from PIL import Image, ImageDraw, ImageFont

# Create a simple diagram-like image
img = Image.new('RGB', (400, 300), color='white')
draw = ImageDraw.Draw(img)

# Draw rectangles (simulating a diagram)
draw.rectangle([50, 50, 150, 100], outline='black', width=2)
draw.rectangle([200, 50, 300, 100], outline='black', width=2)
draw.rectangle([125, 150, 225, 200], outline='black', width=2)

# Draw lines (connections)
draw.line([(150, 75), (200, 75)], fill='black', width=2)
draw.line([(100, 100), (175, 150)], fill='black', width=2)
draw.line([(300, 75), (225, 150)], fill='black', width=2)

# Add text labels
draw.text((70, 60), "Input", fill='black')
draw.text((220, 60), "Process", fill='black')
draw.text((150, 160), "Output", fill='black')

img.save('test_diagram.png')
print("✅ Test diagram created: test_diagram.png")
EOF
```

### Create Sample Audio (for transcription testing)
```bash
python3 << 'EOF'
import numpy as np
from scipy.io import wavfile
import os

# Create a simple WAV file (sine wave)
sample_rate = 16000
duration = 2  # seconds
frequency = 440  # Hz (A note)

t = np.linspace(0, duration, int(sample_rate * duration))
audio = (32767 * 0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)

wavfile.write('test_audio.wav', sample_rate, audio)
print("✅ Test audio file created: test_audio.wav (sine wave)")
EOF
```

---

## 🧪 Component Testing

### Test 1: Ingest Logic (Media Conversion)
```bash
python3 << 'EOF'
print("=" * 50)
print("Testing Ingest Logic (Media Conversion)")
print("=" * 50)

from ingest_logic import image_to_png, pdf_slides_to_images
from pathlib import Path
import os

# Test 1a: Image to PNG conversion
print("\n1️⃣  Testing image_to_png()...")
try:
    if os.path.exists('test_diagram.png'):
        output = image_to_png('test_diagram.png', 'test_output')
        print(f"✅ Image conversion successful")
        print(f"   Input: test_diagram.png")
        print(f"   Output: {output[0]}")
    else:
        print("⚠️  test_diagram.png not found. Create it first with: 'python3 << EOF ... EOF'")
except Exception as e:
    print(f"❌ Image conversion failed: {e}")

# Test 1b: Document metrics
print("\n2️⃣  Testing document analysis...")
try:
    from ingest_logic import VIDEO_EXTS, IMAGE_EXTS
    print(f"✅ Supported video formats: {VIDEO_EXTS}")
    print(f"✅ Supported image formats: {IMAGE_EXTS}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n✅ Ingest logic tests complete!")
EOF
```

### Test 2: Audio Engine (Transcription)
```bash
python3 << 'EOF'
print("=" * 50)
print("Testing Audio Engine (Transcription)")
print("=" * 50)

from audio_engine import transcribe_audio
import os

print("\n1️⃣  Checking Whisper model...")
try:
    import whisper
    print("✅ Whisper library loaded successfully")
    
    # List available models
    print("   Available Whisper models: tiny, base, small, medium, large")
except ImportError as e:
    print(f"❌ Whisper not installed: {e}")
    print("   Install with: pip install openai-whisper")

print("\n2️⃣  Testing transcription (requires audio file)...")
try:
    if os.path.exists('test_audio.wav'):
        print("   Transcribing test_audio.wav...")
        print("   (This may take 30 seconds for base model)")
        
        # Test with tiny model (fastest)
        result = transcribe_audio('test_audio.wav', model_size='tiny')
        print(f"✅ Transcription successful")
        print(f"   Result: {result[:100] if result else 'Empty'}")
    else:
        print("⚠️  test_audio.wav not found. Create with media upload or audio generation.")
except Exception as e:
    print(f"❌ Transcription failed: {e}")

print("\n✅ Audio engine tests complete!")
EOF
```

### Test 3: Vision Engine - Local (Ollama)
```bash
python3 << 'EOF'
print("=" * 50)
print("Testing Vision Engine - Local (Ollama)")
print("=" * 50)

from vision_engine_local import (
    visual_summary_from_image_local,
    video_summary_from_frame_summaries_local,
    check_ollama_models
)
import os

print("\n1️⃣  Checking Ollama connection...")
try:
    models = check_ollama_models()
    if models.get('error'):
        print(f"❌ Ollama error: {models['error']}")
    else:
        print("✅ Ollama connected successfully")
        print(f"   Vision models: {models.get('vision', [])}")
        print(f"   Text models: {models.get('text', [])}")
except Exception as e:
    print(f"❌ Connection failed: {e}")

print("\n2️⃣  Testing visual_summary_from_image()...")
try:
    if os.path.exists('test_diagram.png'):
        print("   Analyzing test_diagram.png...")
        print("   (This may take 30-60 seconds with llava:7b)")
        
        summary = visual_summary_from_image_local(
            'test_diagram.png',
            model_name='llava:7b'
        )
        print(f"✅ Visual analysis successful")
        print(f"   Summary: {summary[:150]}...")
    else:
        print("⚠️  test_diagram.png not found")
except Exception as e:
    print(f"❌ Visual analysis failed: {e}")

print("\n3️⃣  Testing video_summary_from_frame_summaries()...")
try:
    test_summaries = [
        "Shows a diagram with three connected components",
        "Demonstrates data flow between components"
    ]
    print("   Creating synthesis from sample summaries...")
    
    result = video_summary_from_frame_summaries_local(
        test_summaries,
        frame_labels=["Frame 1", "Frame 2"],
        model_name='llama2:7b'
    )
    print(f"✅ Synthesis successful")
    print(f"   Result: {result[:150]}...")
except Exception as e:
    print(f"❌ Synthesis failed: {e}")

print("\n✅ Vision engine (local) tests complete!")
EOF
```

### Test 4: Vision Engine - Cloud (Gemini)
```bash
python3 << 'EOF'
print("=" * 50)
print("Testing Vision Engine - Cloud (Gemini)")
print("=" * 50)

from vision_engine import visual_summary_from_image, _load_api_key
import os

print("\n1️⃣  Checking Gemini API key...")
try:
    api_key = _load_api_key()
    print("✅ Gemini API key loaded")
    print(f"   Key starts with: {api_key[:10]}...")
except RuntimeError as e:
    print(f"❌ API key not found: {e}")
    print("   Set GEMINI_API_KEY or GOOGLE_API_KEY in environment")

print("\n2️⃣  Testing visual_summary_from_image()...")
try:
    if os.path.exists('test_diagram.png'):
        print("   Analyzing test_diagram.png...")
        print("   (This may take 10-15 seconds)")
        
        summary = visual_summary_from_image('test_diagram.png')
        print(f"✅ Visual analysis successful")
        print(f"   Summary: {summary[:150]}...")
    else:
        print("⚠️  test_diagram.png not found")
except Exception as e:
    print(f"❌ Visual analysis failed: {e}")

print("\n✅ Vision engine (cloud) tests complete!")
EOF
```

---

## 🎬 Full Pipeline Testing

### Test 5: Local Pipeline (Ollama-based)
```bash
echo "Starting Ollama service..."
ollama serve &  # Run in background

echo "Waiting for Ollama to start..."
sleep 5

echo "Starting SecondBrainAI Local..."
streamlit run app_local.py

# Access at: http://localhost:8501
```

### Test 6: Cloud Pipeline (Gemini-based)
```bash
# Set API key
export GEMINI_API_KEY="your-api-key-here"

echo "Starting SecondBrainAI Cloud..."
streamlit run app.py

# Access at: http://localhost:8501
```

---

## 🎯 Testing Checklist

### Local Version Testing (Ollama)
- [ ] Ollama models are installed (`ollama list` shows models)
- [ ] Vision model (llava:7b) analyzes test image correctly
- [ ] Text model (llama2:7b) synthesizes frame summaries
- [ ] Streamlit UI starts without errors
- [ ] Ollama status pane shows connected models
- [ ] File upload works (test with PNG/JPG)
- [ ] Image summarization completes (wait for spinner)
- [ ] Results display in collapsible sections

### Cloud Version Testing (Gemini)
- [ ] GEMINI_API_KEY environment variable is set
- [ ] API connection is successful (no "missing key" error)
- [ ] Streamlit UI starts without errors
- [ ] File upload works (test with PDF/PNG/Video)
- [ ] Image summarization completes
- [ ] Video processing extracts frames
- [ ] Audio transcription completes (if video has audio)
- [ ] Final video summary is generated and displayed

### Performance Baselines
| Component | Local (7B) | Cloud (Gemini) |
|-----------|-----------|----------------|
| Image analysis | 30-60 sec | 5-10 sec |
| Video synthesis | 30-60 sec | 10-15 sec |
| Audio transcription | 30-120 sec | 30-120 sec |
| Full 5-frame video | 5-10 min | 2-3 min |

---

## 🐛 Troubleshooting During Testing

### "Ollama not found" / "Connection refused"
```bash
# Start Ollama
ollama serve

# Or with background daemon
ollama serve &
sleep 5

# Verify connection
ollama list
```

### "Model not found" / "No vision models"
```bash
# Download required models
ollama pull llava:7b
ollama pull llama2:7b

# Verify
ollama list
```

### "Module not found" errors
```bash
# Reinstall dependencies
pip install -r requirements_local.txt --force-reinstall

# Or for cloud version:
pip install -r requirements.txt --force-reinstall
```

### "API key not found" (Cloud version)
```bash
# Set API key
export GEMINI_API_KEY="your-actual-key"

# Verify it's set
echo $GEMINI_API_KEY

# Or create .env file
echo "GEMINI_API_KEY=your-key-here" > .env
```

### "FFmpeg not found" (for video processing)
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg

# Windows
winget install ffmpeg
```

### Slow performance / High memory usage
```bash
# Use minimum models
ollama pull llava:7b  # Instead of llava:13b
ollama pull llama2:7b  # Instead of llama2:13b

# Reduce max frames in UI
# Edit app_local.py line ~60: change value=8 to value=4

# Close other applications
# Free up system RAM before processing
```

---

## 📊 Test Results Template

Document your test results:

```
TEST RESULTS - [DATE]
====================

Environment:
- OS: [macOS/Linux/Windows]
- Python: [version]
- Ollama: [installed/status]
- Gemini API: [configured/key present]

Component Tests:
- Ingest Logic: [PASS/FAIL]
- Audio Engine: [PASS/FAIL]
- Vision Local: [PASS/FAIL]
- Vision Cloud: [PASS/FAIL]

Integration Tests:
- Local Pipeline: [PASS/FAIL]
- Cloud Pipeline: [PASS/FAIL]

Performance:
- Image analysis time: [X seconds]
- Video synthesis time: [X seconds]
- Full pipeline time: [X minutes]

Issues Found:
1. [Issue description]
   Resolution: [What fixed it]

Notes:
- [Any observations]
- [Model performance notes]
```

---

## ✅ Sign-Off

Once all tests pass, you've validated:
- ✅ All AI components work independently
- ✅ External models (Ollama/Gemini) are accessible
- ✅ Media processing pipeline functions correctly
- ✅ Full end-to-end workflow produces expected output
- ✅ UI elements render properly
- ✅ Error handling works as expected

Ready for production use! 🎉