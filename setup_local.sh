#!/bin/bash
# setup_local.sh - Automated setup for SecondBrainAI Local
# Run with: bash setup_local.sh
#
# Pulls default Ollama models aligned with vision_engine_local.py:
#   - llama3          (text: summaries, page merge, Q&A)
#   - llava:13b       (vision: per-page / frame analysis)
# Low-resource machines can skip and pull llava:7b + a smaller text model instead.

set -e  # Exit on any error

echo "🚀 Setting up SecondBrainAI Local..."
echo "====================================="

# Check Python
echo "🐍 Checking Python version..."
python3 --version || { echo "❌ Python 3 required. Please install Python 3.8+"; exit 1; }

# Check if we're in the right directory
if [ ! -f "app_local.py" ]; then
    echo "❌ Please run this script from the secondBrainAI directory"
    exit 1
fi

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh/"
            exit 1
        fi
    else
        # Linux/Windows (WSL)
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
else
    echo "✅ Ollama already installed"
fi

# Start Ollama service
echo "🦙 Starting Ollama service..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew services start ollama 2>/dev/null || ollama serve &
elif command -v systemctl &> /dev/null; then
    sudo systemctl start ollama 2>/dev/null || ollama serve &
else
    ollama serve &
fi

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 5

# Test Ollama connection
echo "🔍 Testing Ollama connection..."
if ! ollama list &> /dev/null; then
    echo "❌ Ollama not responding. Please check installation."
    exit 1
fi
echo "✅ Ollama connected"

# Download recommended models (match app defaults in vision_engine_local.py)
echo "🤖 Downloading AI models..."
echo "   Default pulls: llama3 (text) + llava:13b (vision)."
echo "   This may take 20–60+ minutes depending on bandwidth; total size is often ~12–18 GB."
echo ""

# Text: prefer Llama 3 family (skip pull if any llama3* tag is already installed)
if ollama list 2>/dev/null | awk '{print $1}' | grep -qE '^llama3'; then
    echo "   ✅ Llama 3 text model — already present"
else
    echo "   📥 Downloading Llama 3 text model (llama3)..."
    ollama pull llama3
fi

# Vision: LLaVA 13B (stronger than 7B for figures/layout)
if ollama list 2>/dev/null | awk '{print $1}' | grep -qxF "llava:13b"; then
    echo "   ✅ LLaVA 13B vision model — already present"
else
    echo "   📥 Downloading LLaVA 13B vision model (llava:13b)..."
    ollama pull llava:13b
fi

echo ""
echo "   Optional (smaller / older — use if low RAM or disk):"
echo "     ollama pull llava:7b"
echo "     ollama pull llama2:7b"
echo "   Optional (alternatives): ollama pull qwen2.5  OR  ollama pull mistral"

# Setup Python virtual environment
echo "🐍 Setting up Python environment..."
if [ ! -d "secondbrain_env" ]; then
    python3 -m venv secondbrain_env
fi

# Activate environment and install dependencies
echo "📦 Installing Python dependencies..."
source secondbrain_env/bin/activate
pip install --upgrade pip
pip install -r requirements_local.txt

# Test the setup
echo "🧪 Testing setup..."
python3 -c "
import ollama
import streamlit
from PIL import Image
from vision_engine_local import check_ollama_models
print('✅ All Python imports successful')
models = check_ollama_models()
if models.get('error'):
    print('❌ Ollama connection issue:', models['error'])
    exit(1)
print('✅ Ollama models available:', len(models.get('vision', [])), 'vision,', len(models.get('text', [])), 'text')
"

echo ""
echo "🎉 Setup complete!"
echo "=================="
echo "To run SecondBrainAI Local:"
echo "1. Activate the environment: source secondbrain_env/bin/activate"
echo "2. Start the full UI: streamlit run app.py"
echo "   (minimal demo: streamlit run app_local.py)"
echo "3. Open your browser to the displayed URL"
echo "4. In the sidebar, confirm Vision ≈ llava:13b and Text ≈ llama3 (or your installed tags)"
echo ""
echo "For team members, share SETUP_LOCAL.md for manual setup instructions."
