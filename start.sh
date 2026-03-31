#!/bin/bash
# Start AI Vigilance (works on Windows Git Bash, Linux, and Linux VM)

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
fi

# On headless Linux, ensure no GUI attempts
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export DISPLAY=${DISPLAY:-""}
    export OPENCV_VIDEOIO_PRIORITY_MSMF=0
fi

echo "Starting AI Vigilance on http://0.0.0.0:8000"
python -m uvicorn app:app --host 0.0.0.0 --port 8000
