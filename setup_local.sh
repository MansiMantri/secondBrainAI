#!/bin/bash
# setup_local.sh - Automated setup for SecondBrainAI Local
# Run with: bash setup_local.sh

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

# Download required models
echo "🤖 Downloading AI models..."
echo "   This may take 10-30 minutes depending on your internet speed..."

# Download vision model
if ! ollama list | grep -q "llava:7b"; then
    echo "   📥 Downloading LLaVA vision model (4.7GB)..."
    ollama pull llava:7b
else
    echo "   ✅ LLaVA vision model already downloaded"
fi

# Download text model
if ! ollama list | grep -q "llama2:7b"; then
    echo "   📥 Downloading Llama2 text model (3.8GB)..."
    ollama pull llama2:7b
else
    echo "   ✅ Llama2 text model already downloaded"
fi

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
echo "2. Start the app: streamlit run app_local.py"
echo "3. Open your browser to the displayed URL"
echo ""
echo "For team members, share SETUP_LOCAL.md for manual setup instructions."</content>
<parameter name="filePath">/Users/sanchitvartak/Desktop/Spring26/AI_BNgan/ai-project/secondBrainAI/setup_local.sh