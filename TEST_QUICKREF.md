# SecondBrainAI Testing Quick Reference

Fast guide to testing all components.

---

## 🚀 Quick Start - Run All Tests

### Everything at Once
```bash
# Test everything (local + cloud, includes slow vision tests)
python3 run_all_tests.py both

# Test only local version (Ollama)
python3 run_all_tests.py local

# Test only cloud version (Gemini)
python3 run_all_tests.py cloud

# Quick checks only (skips slow vision analysis)
python3 run_all_tests.py both --skip-slow
```

---

## 🧪 Individual Component Tests

### Test Media Processing
```bash
python3 test_ingest.py
# Verifies: Image formats, file handling, media conversion
# Time: ~5 seconds
```

### Test Audio Transcription
```bash
python3 test_audio.py
# Verifies: Whisper model, audio processing
# Time: ~30 seconds
```

### Test Local Vision (Ollama)
```bash
python3 test_vision_local.py
# Verifies: Ollama connection, image analysis, text synthesis
# Time: ~2-3 minutes (depends on model size)
# Requires: Ollama running, llava:7b and llama2:7b installed
```

### Test Cloud Vision (Gemini)
```bash
export GEMINI_API_KEY="your-key-here"
python3 test_vision_cloud.py
# Verifies: API connection, image analysis, text synthesis
# Time: ~1-2 minutes
# Requires: Valid Gemini API key
```

---

## ▶️ Full Pipeline Tests

### Run Local Version
```bash
# Start Ollama first
ollama serve &

# Then run app
streamlit run app_local.py

# Open browser to http://localhost:8501
# Test with: PNG, JPG, or MP4 file
```

### Run Cloud Version
```bash
# Set API key
export GEMINI_API_KEY="your-key"

# Run app
streamlit run app.py

# Open browser to http://localhost:8501
# Test with: PDF, PNG, JPG, or MP4 file
```

---

## 📊 Test Coverage Matrix

| Component | Test Script | Local | Cloud | Time |
|-----------|------------|-------|-------|------|
| **Media Processing** | `test_ingest.py` | ✅ | ✅ | <1 min |
| **Audio Processing** | `test_audio.py` | ✅ | ✅ | <1 min |
| **Image Analysis** | `test_vision_local.py` | ✅ | - | 1-2 min |
| **Image Analysis** | `test_vision_cloud.py` | - | ✅ | 1-2 min |
| **Full App** | `streamlit run app_local.py` | ✅ | - | 10-15 min |
| **Full App** | `streamlit run app.py` | - | ✅ | 5-10 min |

---

## ✅ Expected Results

### Successful Component Test Output
```
============================================================
Testing [Component Name]
============================================================

✅ Module imported
✅ [Subtest 1] passed
✅ [Subtest 2] passed
...

✅ All [Component] tests passed!
```

### Successful Full Pipeline Output
- **Ollama Status**: Shows connected models
- **File Upload**: Accepts media files
- **Processing**: Shows progress spinners
- **Results**: Displays summaries in expandable sections
- **No Errors**: Console shows no red error messages

---

## 🔧 Quick Diagnostics

### Check System Setup
```bash
# Python version (3.8+ required)
python3 --version

# Ollama status (for local testing)
ollama list

# API key set (for cloud testing)
echo $GEMINI_API_KEY

# FFmpeg installed (for video processing)
ffmpeg -version

# Required Python packages
pip list | grep -E "streamlit|ollama|PIL|whisper"
```

### Common Issues

**"Module not found" errors**
```bash
pip install -r requirements_local.txt
# OR
pip install -r requirements.txt
```

**"Ollama connection refused"**
```bash
ollama serve  # Start Ollama in another terminal
```

**"No vision/text models found"**
```bash
ollama pull llava:7b
ollama pull llama2:7b
```

**"API key not found"**
```bash
export GEMINI_API_KEY="your-actual-key"
```

---

## 🎬 Test Scenario: Complete Workflow

### Scenario 1: Test Image Summarization
1. Create test image: `python3 test_ingest.py` creates one
2. Analyze locally: `python3 test_vision_local.py`
3. Test in UI: Upload `test_diagram.png` to `streamlit run app_local.py`
4. Verify: Summary appears in ~1-2 minutes

### Scenario 2: Test Video Processing
1. Get sample video (MP4 file)
2. Run full app: `streamlit run app_local.py`
3. Upload video (max 8 frames default)
4. Wait for:
   - Frame extraction (~10 sec)
   - Frame analysis (~2 min)
   - Audio transcription (~1 min)
   - Video synthesis (~1 min)
5. Total time: ~4-5 minutes

### Scenario 3: Test Cloud vs Local
1. Run local: `python3 test_vision_local.py` → ~2 min
2. Run cloud: `python3 test_vision_cloud.py` → ~1 min
3. Compare results in both

---

## 🎯 Pass/Fail Criteria

### ✅ Test Passes If:
- Script completes without Python errors
- Shows "✅ PASS" or "✅ successful" messages
- No red error text in output
- Returns exit code 0

### ❌ Test Fails If:
- Script crashes with traceback
- Shows "❌ FAIL" messages
- Error messages visible in output
- Returns non-zero exit code

---

## 📝 Reporting Test Results

When documenting results:
```
Platform: [macOS/Linux/Windows]
Python: [version]
Date: [date]

Test Results:
- Media Processing: ✅ PASS
- Audio Engine: ✅ PASS
- Vision Local: ✅ PASS
- Vision Cloud: ✅ PASS
- Full App Local: ✅ PASS
- Full App Cloud: ✅ PASS

Performance:
- Image analysis: 45 seconds
- Video synthesis: 60 seconds
- Full 5-frame video: 4 minutes 30 seconds

Issues: None
```

---

## 🔗 Related Docs
- **Setup Guide**: See `SETUP_LOCAL.md` for installation steps
- **Full Testing Guide**: See `TESTING.md` for comprehensive details
- **Code Documentation**: See individual module docstrings