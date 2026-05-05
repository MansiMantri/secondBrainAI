# SecondBrainAI Local Version

Run the complete video/image summarization pipeline locally using Ollama models instead of cloud APIs.

## 🚀 Quick Start

### 1. Install Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 2. Install Required Models
```bash
# Vision model for image understanding
ollama pull llava:7b

# Text model for synthesis (optional, defaults to llama2)
ollama pull llama2:7b
```

### 3. Install Python Dependencies
```bash
pip install -r requirements_local.txt
```

### 4. Run the Application
```bash
streamlit run app_local.py
```

## 🔧 Alternative Models

### Vision Models (for image/frame analysis)
```bash
# LLaVA variants (recommended)
ollama pull llava:7b          # 7B parameters, good balance
ollama pull llava:13b         # 13B parameters, better quality
ollama pull llava:34b         # 34B parameters, best quality (slow)

# Other vision models
ollama pull bakllava         # Alternative vision model
ollama pull moondream         # Lightweight vision model
```

### Text Models (for video synthesis)
```bash
# General purpose
ollama pull llama2:7b
ollama pull llama2:13b

# Instruction-tuned
ollama pull codellama:7b      # Code-focused
ollama pull orca-mini:7b      # Chat-optimized

# Larger models for better synthesis
ollama pull llama2:70b        # Very high quality (requires lots of RAM)
```

## 📋 System Requirements

### Minimum (LLaVA 7B + Llama2 7B)
- RAM: 16GB
- Storage: 20GB free space
- OS: macOS, Linux, or Windows

### Recommended (LLaVA 13B + Llama2 13B)
- RAM: 32GB
- Storage: 40GB free space
- GPU: Optional but recommended for faster inference

## 🔄 How It Works

1. **Frame Extraction**: Videos are sampled into evenly-spaced frames
2. **Vision Analysis**: Each frame analyzed by LLaVA for visual content
3. **Audio Transcription**: Whisper extracts speech (if available)
4. **Synthesis**: Llama2 combines visuals + audio into narrative

## ⚙️ Configuration

Models can be changed in the Streamlit UI:
- **Vision Model**: Select from available Ollama vision models
- **Text Model**: Select from available text models for synthesis

## 🆚 Comparison: Cloud vs Local

| Aspect | Gemini API | Local Ollama |
|--------|------------|--------------|
| **Cost** | API charges | Free (one-time setup) |
| **Privacy** | Data sent to Google | Local processing only |
| **Setup** | API key required | Ollama + models download |
| **Speed** | Fast (cloud) | Variable (depends on hardware) |
| **Offline** | Requires internet | Works offline |
| **Customization** | Limited | Full control |

## 🐛 Troubleshooting

### Ollama Connection Issues
```bash
# Start Ollama service
ollama serve

# Check models
ollama list

# Test a model
ollama run llava:7b "Describe this image" --image test.jpg
```

### Memory Issues
- Use smaller models: `llava:7b` instead of `llava:13b`
- Reduce max frames in the UI (try 4-6 instead of 8)
- Close other applications to free RAM

### Slow Performance
- Use GPU if available: Ollama automatically detects CUDA/Metal
- Try smaller models for faster inference
- Reduce image quality in processing

## 📁 File Structure

```
secondBrainAI/
├── app_local.py              # Local Streamlit app
├── vision_engine_local.py    # Ollama-based vision engine
├── requirements_local.txt    # Local dependencies
├── README_local.md          # This file
└── [other original files...] # Reuse existing logic
```</content>
<parameter name="filePath">/Users/sanchitvartak/Desktop/Spring26/AI_BNgan/ai-project/secondBrainAI/README_local.md