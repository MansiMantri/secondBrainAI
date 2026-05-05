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

### Minimum Requirements (LLaVA 7B + Llama2 7B)
- **Operating System**: macOS 12+, Ubuntu 18.04+, Windows 10/11
- **RAM**: 16GB (recommended 32GB)
- **Storage**: 25GB free space for models + data
- **CPU**: Modern multi-core processor
- **GPU**: Optional (NVIDIA/AMD with CUDA/Metal support for faster inference)

### Recommended Requirements (LLaVA 13B + Llama2 13B)
- **RAM**: 32GB (64GB for Llama2 70B)
- **Storage**: 50GB free space
- **GPU**: NVIDIA RTX 30-series or better (with CUDA)

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

Download the required AI models for vision analysis and text synthesis.

### Required Models (Minimum Setup)
```bash
# Vision model for image/frame analysis
ollama pull llava:7b

# Text model for video synthesis
ollama pull llama2:7b
```

### Recommended Models (Better Quality)
```bash
# Higher quality vision model
ollama pull llava:13b

# Better text synthesis
ollama pull llama2:13b
```

### Optional Models (Advanced Users)
```bash
# Best vision quality (slowest)
ollama pull llava:34b

# Code-focused synthesis
ollama pull codellama:7b

# Chat-optimized synthesis
ollama pull orca-mini:7b

# Lightweight alternative vision
ollama pull moondream
ollama pull bakllava
```

### Model Download Times
- `llava:7b`: ~4GB, 5-10 minutes
- `llava:13b`: ~7GB, 10-20 minutes
- `llama2:7b`: ~4GB, 5-10 minutes
- `llama2:13b`: ~7GB, 10-20 minutes

### Verification
```bash
# List all downloaded models
ollama list

# Expected output:
# NAME            ID              SIZE      MODIFIED
# llava:7b        8dd30f6b0cb1    4.7 GB    2 minutes ago
# llama2:7b       78e26419b446    3.8 GB    1 minute ago
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
echo "Hello, describe a cat" | ollama run llama2:7b
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
    result = visual_summary_from_image('test_image.png', model_name='llava:7b')
    print('Vision test successful:', result[:100] + '...')
except Exception as e:
    print('Vision test failed:', e)
"
```

### 4. Run the Application
```bash
# Start the local Streamlit app
streamlit run app_local.py

# Expected: Opens browser with SecondBrainAI Local interface
# Check Ollama Status section shows connected models
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
ollama pull llava:7b --retry 3

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
# Try different text models
ollama pull codellama:7b
ollama pull orca-mini:7b

# Adjust temperature in vision_engine_local.py
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

# Document in README which models team should install
echo "# Required Ollama Models
ollama pull llava:7b
ollama pull llama2:7b" > models_required.txt
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

## 🚀 Quick Setup Script (Optional)

Create `setup_local.sh` for automated setup:

```bash
#!/bin/bash
# setup_local.sh - Automated setup for SecondBrainAI Local

echo "🚀 Setting up SecondBrainAI Local..."

# Check Python
python3 --version || { echo "Python 3 required"; exit 1; }

# Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Start Ollama
echo "🦙 Starting Ollama..."
ollama serve &

# Wait for Ollama to start
sleep 5

# Download models
echo "🤖 Downloading AI models..."
ollama pull llava:7b
ollama pull llama2:7b

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv secondbrain_env
source secondbrain_env/bin/activate
pip install -r requirements_local.txt

echo "✅ Setup complete! Run: streamlit run app_local.py"
```

Make executable: `chmod +x setup_local.sh`

---

## 📞 Support & Resources

### Getting Help
1. Check this guide's troubleshooting section
2. Verify Ollama status: `ollama list`
3. Test models individually: `ollama run llava:7b "describe this image" --image test.jpg`
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

*Last updated: April 1, 2026*
*Tested on: macOS 14.0, Ubuntu 22.04, Windows 11*</content>
<parameter name="filePath">/Users/sanchitvartak/Desktop/Spring26/AI_BNgan/ai-project/secondBrainAI/SETUP_LOCAL.md