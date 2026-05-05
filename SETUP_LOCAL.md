# SecondBrainAI Local Setup Guide

## Prerequisites Setup for Local AI-Powered Video/Image Summarization

This guide covers the complete setup process for running SecondBrainAI locally using Ollama models instead of cloud APIs. This provides complete privacy and zero API costs.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Ollama Installation](#ollama-installation)
3. [AI Model Downloads](#ai-model-downloads)
4. [Python Environment Setup](#python-environment-setup)
5. [Project Setup](#project-setup)
6. [Verification & Testing](#verification--testing)
7. [Troubleshooting](#troubleshooting)
8. [Team Development Notes](#team-development-notes)

---

## 🖥️ System Requirements

### Minimum Requirements (lighter optional stack; see [AI Model Downloads](#ai-model-downloads))
- **Operating System**: macOS 12+, Ubuntu 18.04+, Windows 10/11
- **RAM**: 16GB (recommended 32GB for default models below)
- **Storage**: ~25GB free for smaller models; **~35–45GB** recommended for default `llama3` + `llava:13b` pulls + data
- **CPU**: Modern multi-core processor
- **GPU**: Optional (NVIDIA/AMD with CUDA/Metal support for faster inference)

### Recommended Requirements (default setup: Llama 3 + LLaVA 13B)
- **RAM**: 32GB
- **Storage**: 45GB+ free space for models + processed uploads
- **GPU**: Apple Silicon / NVIDIA RTX 20-series or better when available (Ollama uses GPU automatically when supported)

### Supported Platforms
- ✅ macOS (Intel/Apple Silicon)
- ✅ Linux (Ubuntu, Fedora, CentOS)
- ✅ Windows 10/11 (WSL recommended)

---

## 🦙 Ollama Installation

Ollama is the local AI model runtime that replaces cloud APIs.

### macOS Installation
```bash
# Using Homebrew (recommended)
brew install ollama

# Start Ollama service
brew services start ollama
# OR run manually:
ollama serve
```

### Linux Installation
```bash
# Download and install
curl -fsSL https://ollama.ai/install.sh | sh

# Start service (Ubuntu/Debian)
sudo systemctl start ollama
# OR run manually:
ollama serve
```

### Windows Installation
1. Download installer from [ollama.ai/download](https://ollama.ai/download)
2. Run installer and follow setup wizard
3. Ollama will start automatically

### Verification
```bash
# Check Ollama version
ollama --version

# Test Ollama service
ollama list
```
Expected output: Shows installed models (initially empty)

---

## 🤖 AI Model Downloads

SecondBrainAI’s defaults (see `vision_engine_local.py`) assume **strong text + vision** models. Install these first for the best summaries and alignment with the PDF/page pipeline.

### Default / Recommended (best quality — matches automated `setup_local.sh`)
```bash
# Text: summaries, per-page merge, combined summary, Ask/Q&A
ollama pull llama3

# Vision: per-page / per-frame image understanding (figures, layout)
ollama pull llava:13b
```

These are the models **`setup_local.sh` pulls** when they are not already installed (`ollama list` is checked first). Expect **roughly 12–18 GB** total depending on Ollama tags and quantization.

### Low-resource alternative (smaller downloads)
If disk or RAM is tight, use smaller models and select them in the app sidebar:
```bash
ollama pull llava:7b
ollama pull llama2:7b
```

### Optional Models (Advanced Users)
```bash
# Newer Llama 3.x variants (often listed as separate tags in `ollama list`)
ollama pull llama3.2
ollama pull llama3.1

# Alternative instruct models
ollama pull qwen2.5
ollama pull mistral

# Best vision quality (slowest / largest)
ollama pull llava:34b

# Lightweight vision alternatives
ollama pull moondream
ollama pull bakllava

# Code-focused (optional)
ollama pull codellama:7b
```

### Model download times (order of magnitude)
- `llama3`: ~4–5 GB, varies by tag
- `llava:13b`: ~8 GB
- `llava:7b`: ~4–5 GB
- `llama2:7b`: ~4 GB

### Verification
```bash
# List all downloaded models
ollama list

# You should see at least llama3 (or another llama3* tag) and llava:13b after default setup
```

---

## 🐍 Python Environment Setup

### Python Version Requirements
- **Python**: 3.8 or higher (3.9+ recommended)
- **Pip**: Latest version

### Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv secondbrain_env

# Activate environment
# macOS/Linux:
source secondbrain_env/bin/activate
# Windows:
# secondbrain_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Install Dependencies
```bash
# Install local requirements
pip install -r requirements_local.txt

# Verify installations
python3 -c "import ollama, streamlit, PIL; print('All imports successful')"
```

### requirements_local.txt Contents
```
# Core dependencies
streamlit
pypdfium2
pillow
python-dotenv
imageio
imageio-ffmpeg

# Audio transcription (local)
openai-whisper

# Local AI models
ollama

# Optional: Vector database
chromadb
```

---

## 📁 Project Setup

### Clone/Download Project
```bash
# If using git
git clone <repository-url>
cd secondBrainAI

# If downloaded as ZIP, extract and navigate
cd path/to/secondBrainAI
```

### Project Structure
```
secondBrainAI/
├── app_local.py              # Local Streamlit app (main entry point)
├── vision_engine_local.py    # Ollama-based vision engine
├── requirements_local.txt    # Local dependencies
├── README_local.md          # This setup guide
├── app.py                   # Original Gemini version
├── vision_engine.py         # Original Gemini engine
├── audio_engine.py          # Audio transcription (shared)
├── ingest_logic.py          # Media processing (shared)
└── data/                    # Data directory (created automatically)
    ├── uploads/            # Uploaded files
    └── processed/          # Processed results
```

### Environment Variables (Optional)
Create `.env` file in project root:
```bash
# No API keys needed for local version
# But you can still set these if switching between versions
# GEMINI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
```

---

## ✅ Verification & Testing

### 1. Test Ollama Connection
```bash
# Check models are loaded
ollama list

# Test basic Ollama functionality
echo "Hello, describe a cat" | ollama run llama3
```

### 2. Test Python Environment
```bash
# Test imports
python3 -c "
import ollama
import streamlit
from PIL import Image
from vision_engine_local import check_ollama_models
print('All imports successful')
models = check_ollama_models()
print('Available models:', models)
"
```

### 3. Test Vision Engine
```bash
# Create a test image (optional)
python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (100, 100), color='red')
draw = ImageDraw.Draw(img)
draw.text((10, 40), 'TEST', fill='white')
img.save('test_image.png')
print('Test image created')
"

# Test vision analysis (requires test image)
python3 -c "
from vision_engine_local import visual_summary_from_image
try:
    result = visual_summary_from_image('test_image.png', model_name='llava:13b')
    print('Vision test successful:', result[:100] + '...')
except Exception as e:
    print('Vision test failed:', e)
"
```

### 4. Run the Application
```bash
# Primary UI (recommended)
streamlit run app.py

# Minimal demo
streamlit run app_local.py

# Expected: Opens browser; sidebar model lists should include llama3 / llava:13b after setup
```

### 5. Full Pipeline Test
1. Open the running Streamlit app
2. Upload a test image or video
3. Verify processing completes without errors
4. Check that summaries are generated

---

## 🔧 Troubleshooting

### Ollama Issues

**"Ollama command not found"**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Add to PATH if needed
export PATH=$PATH:/usr/local/bin
```

**"Ollama service not running"**
```bash
# Start Ollama
ollama serve

# Or as background service
# macOS: brew services start ollama
# Linux: sudo systemctl start ollama
```

**Model download fails**
```bash
# Check internet connection
ping 8.8.8.8

# Retry download
ollama pull llava:13b

# Check disk space
df -h
```

### Python Issues

**"Module not found" errors**
```bash
# Reinstall requirements
pip install -r requirements_local.txt --force-reinstall

# Check Python version
python3 --version

# Update pip
pip install --upgrade pip
```

**PIL/Pillow issues**
```bash
# Install Pillow explicitly
pip install Pillow

# For macOS with Apple Silicon
pip install --no-cache-dir Pillow
```

### Performance Issues

**Slow processing**
- Use smaller models: `llava:7b` instead of `llava:13b`
- Reduce max frames in UI (4-6 instead of 8)
- Close other applications
- Check GPU usage: Ollama automatically uses GPU if available

**Memory errors**
```bash
# Check available RAM
vm_stat  # macOS
free -h  # Linux

# Use smaller models
ollama pull llava:7b
ollama pull llama2:7b

# Reduce batch processing
# Edit app_local.py to lower max_images default
```

### Model-Specific Issues

**LLaVA not working well**
```bash
# Try alternative vision models
ollama pull moondream
ollama pull bakllava

# Update app_local.py to use different model
# Change DEFAULT_VISION_MODEL = "moondream"
```

**Poor text synthesis quality**
```bash
# Prefer stronger instruct models (pull then select in sidebar)
ollama pull llama3
ollama pull qwen2.5
ollama pull mistral

# Optional legacy / niche
ollama pull codellama:7b

# Adjust temperature in vision_engine_local.py if needed
# Lower temperature (0.1-0.3) for more focused synthesis
```

---

## 👥 Team Development Notes

### Version Control
```bash
# Add local files to git
git add app_local.py vision_engine_local.py requirements_local.txt README_local.md

# Commit with descriptive message
git commit -m "Add local Ollama-based alternative to Gemini API

- vision_engine_local.py: Ollama integration for vision analysis
- app_local.py: Local Streamlit app without API dependencies
- requirements_local.txt: Local-only dependencies
- README_local.md: Comprehensive setup guide

Features:
- Complete privacy (no data sent to cloud)
- Zero API costs
- Same functionality as cloud version
- Model selection in UI
- Ollama status monitoring"

# Push to repository
git push origin main
```

### Branching Strategy
```bash
# Create feature branch for local development
git checkout -b feature/local-ollama-support

# After testing, merge to main
git checkout main
git merge feature/local-ollama-support
```

### Environment Consistency
```bash
# Create .env.example for team
cp .env .env.example
# Remove actual API keys, keep structure

# Document which models team should install (defaults match setup_local.sh)
echo "# Recommended Ollama models (see SETUP_LOCAL.md)
ollama pull llama3
ollama pull llava:13b" > models_required.txt
```

### CI/CD Considerations
- Local version doesn't require API key secrets
- Models are downloaded per developer machine
- Consider adding model download to setup scripts
- Test both cloud and local versions in CI

### Documentation Updates
- Update main README.md to mention local alternative
- Add badges showing both cloud and local options
- Include performance comparisons
- Document model recommendations for different use cases

---

## 🚀 Quick Setup Script

The repository includes **`setup_local.sh`** in the `secondBrainAI` folder. It installs Ollama (macOS via Homebrew when available), starts the service, pulls **`llama3`** and **`llava:13b`** when missing, creates `secondbrain_env`, and installs `requirements_local.txt`.

```bash
cd secondBrainAI
chmod +x setup_local.sh
bash setup_local.sh
```

After it finishes: `source secondbrain_env/bin/activate` then `streamlit run app.py`.

---

## 📞 Support & Resources

### Getting Help
1. Check this guide's troubleshooting section
2. Verify Ollama status: `ollama list`
3. Test models individually, e.g. `ollama run llava:13b` (vision) and `ollama run llama3` (text)
4. Check system resources: RAM, disk space, GPU

### Useful Links
- [Ollama Documentation](https://github.com/jmorganca/ollama)
- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [SecondBrainAI Issues](https://github.com/your-repo/issues)

### Performance Optimization
- Use SSD storage for models
- Enable GPU acceleration if available
- Monitor RAM usage during processing
- Consider model quantization for lower memory usage

---

*Last updated: May 4, 2026*  
*Tested on: macOS 14.0, Ubuntu 22.04, Windows 11 (WSL)*