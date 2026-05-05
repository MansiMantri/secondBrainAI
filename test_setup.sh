#!/bin/bash
# test_setup.sh - Verify SecondBrainAI Local setup
# Run with: bash test_setup.sh

echo "🧪 Testing SecondBrainAI Local Setup"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "app_local.py" ]; then
    echo "❌ Please run this script from the secondBrainAI directory"
    exit 1
fi

# Check Python environment
echo "🐍 Checking Python environment..."
if [ -d "secondbrain_env" ]; then
    source secondbrain_env/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found. Using system Python."
fi

# Test Python imports
echo "📦 Testing Python imports..."
python3 -c "
try:
    import ollama
    print('✅ Ollama import successful')
except ImportError as e:
    print('❌ Ollama import failed:', e)
    exit(1)

try:
    import streamlit
    print('✅ Streamlit import successful')
except ImportError as e:
    print('❌ Streamlit import failed:', e)
    exit(1)

try:
    from PIL import Image
    print('✅ PIL import successful')
except ImportError as e:
    print('❌ PIL import failed:', e)
    exit(1)
"

# Test Ollama connection
echo "🦙 Testing Ollama connection..."
if command -v ollama &> /dev/null; then
    if ollama list &> /dev/null; then
        echo "✅ Ollama service running"

        # Check for required models
        vision_count=$(ollama list | grep -c "llava\|bakllava\|moondream" || echo "0")
        text_count=$(ollama list | grep -c "llama2\|codellama\|orca" || echo "0")

        echo "🤖 Available models:"
        echo "   Vision models: $vision_count"
        echo "   Text models: $text_count"

        if [ "$vision_count" -eq 0 ]; then
            echo "⚠️  No vision models found. Run: ollama pull llava:7b"
        fi

        if [ "$text_count" -eq 0 ]; then
            echo "⚠️  No text models found. Run: ollama pull llama2:7b"
        fi
    else
        echo "❌ Ollama service not responding"
        echo "   Try: ollama serve"
    fi
else
    echo "❌ Ollama not installed"
fi

# Test vision engine
echo "🔍 Testing vision engine..."
python3 -c "
try:
    from vision_engine_local import check_ollama_models
    models = check_ollama_models()
    if models.get('error'):
        print('❌ Vision engine error:', models['error'])
    else:
        print('✅ Vision engine working')
        print('   Vision models:', len(models.get('vision', [])))
        print('   Text models:', len(models.get('text', [])))
except Exception as e:
    print('❌ Vision engine test failed:', e)
"

# Final status
echo ""
echo "📋 Setup Status Summary:"
echo "========================"

# Check each component
components=("Python Environment" "Ollama Installation" "AI Models" "Python Dependencies" "Vision Engine")
status_checks=(
    "python3 --version &> /dev/null"
    "command -v ollama &> /dev/null"
    "ollama list | grep -q 'llava\|llama2'"
    "python3 -c 'import ollama, streamlit, PIL' &> /dev/null"
    "python3 -c 'from vision_engine_local import check_ollama_models; check_ollama_models()' &> /dev/null"
)

for i in "${!components[@]}"; do
    component="${components[$i]}"
    check="${status_checks[$i]}"

    if eval "$check"; then
        echo "✅ $component: OK"
    else
        echo "❌ $component: FAILED"
    fi
done

echo ""
echo "🚀 To run SecondBrainAI Local:"
echo "   streamlit run app_local.py"
echo ""
echo "📖 For detailed setup help, see SETUP_LOCAL.md"</content>
<parameter name="filePath">/Users/sanchitvartak/Desktop/Spring26/AI_BNgan/ai-project/secondBrainAI/test_setup.sh