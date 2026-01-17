#!/bin/bash

echo "================================================="
echo "   🚀 STARTING AI VISION SYSTEM (MacOS/Linux)"
echo "================================================="

# 1. CHECK PYTHON
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed."
    echo "Please install Python 3 (brew install python3) and try again."
    exit 1
fi

# 2. CHECK VIRTUAL ENV
if [ ! -d "env" ]; then
    echo "ℹ️ First time setup detected..."
    echo "ℹ️ Creating Virtual Environment 'env'..."
    python3 -m venv env
    
    echo "ℹ️ Activating Environment..."
    source env/bin/activate
    
    echo "ℹ️ Upgrading PIP..."
    pip install --upgrade pip
    
    echo "ℹ️ Installing Dependencies..."
    pip install -r requirements.txt
    
    echo "✅ Setup Complete!"
else
    echo "ℹ️ Virtual Environment found. Activating..."
    source env/bin/activate
fi

# 3. LAUNCH APP
echo ""
echo "📦 Lanching Dashboard..."
streamlit run dashboard.py
