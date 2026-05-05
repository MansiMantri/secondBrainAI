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

### 2. Install Recommended Models
Matches defaults in `vision_engine_local.py` and **`bash setup_local.sh`**:

```bash
ollama pull llama3        # text: summaries, page merge, Ask/Q&A
ollama pull llava:13b     # vision: per-page / frame analysis
```

**Low-resource alternative:** `ollama pull llava:7b` and `ollama pull llama2:7b`, then pick those in the sidebar.

**Automated install:** from the `secondBrainAI` folder, run `bash setup_local.sh` (installs the same default models, venv, and `requirements_local.txt`). See **SETUP_LOCAL.md** for details.

### 3. Install Python Dependencies
```bash
pip install -r requirements_local.txt
```

### 4. Run the Application

**Full UI (recommended):** sidebar navigation, batch uploads, saved summary library, Ask/RAG — use:

```bash
streamlit run app.py
```

**Minimal demo** (single-file flow, fewer features):

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

### Text Models (for synthesis / page merge / Q&A)
```bash
ollama pull llama3            # default preference
ollama pull qwen2.5          # alternative instruct model
ollama pull mistral

# Legacy / niche
ollama pull llama2:7b
ollama pull codellama:7b      # Code-focused
```

## 📋 System Requirements

### Minimum (lighter models; see pulls above)
- RAM: 16GB
- Storage: ~25GB free space
- OS: macOS, Linux, or Windows

### Recommended (Llama 3 + LLaVA 13B — default setup)
- RAM: 32GB
- Storage: ~40GB+ free space (models + uploads)
- GPU: Optional; Metal/CUDA speeds things up when available

## 🔄 How It Works

1. **Frame Extraction**: Videos are sampled into evenly-spaced frames
2. **Vision Analysis**: Each frame analyzed by LLaVA for visual content
3. **Audio Transcription**: Whisper extracts speech (if available)
4. **Synthesis**: The chosen text model (e.g. Llama 3) combines visuals + transcripts into narrative summaries

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

# Test vision / text (adjust tags to what you installed)
ollama run llava:13b
ollama run llama3
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