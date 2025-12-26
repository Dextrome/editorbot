#!/bin/bash
# WSL Setup Script for RL Audio Editor
# Run this once to set up the environment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== RL Audio Editor - WSL Setup ===${NC}"

# Check WSL version
if grep -qi microsoft /proc/version; then
    echo -e "${GREEN}Running in WSL${NC}"
else
    echo -e "${RED}Not running in WSL!${NC}"
    exit 1
fi

# Check for Python
if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}Python not found. Installing...${NC}"
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
    sudo ln -sf /usr/bin/python3 /usr/bin/python
fi

echo -e "${YELLOW}Python version:${NC}"
python --version

# Install PyTorch with CUDA (if not already installed)
echo -e "${YELLOW}Checking PyTorch installation...${NC}"
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo -e "${GREEN}PyTorch already installed${NC}"
    python -c "import torch; print(f'  Version: {torch.__version__}')"
fi

# Install other dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install librosa soundfile numpy scipy tensorboard wandb natten

# Install project in editable mode if setup.py exists
cd /mnt/f/editorbot
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo -e "${YELLOW}Installing project...${NC}"
    pip install -e .
fi

# Verify CUDA
echo -e "${YELLOW}Verifying CUDA...${NC}"
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Test torch.compile
echo -e "${YELLOW}Testing torch.compile...${NC}"
python -c "
import torch
try:
    m = torch.nn.Linear(10, 10).cuda()
    m = torch.compile(m, mode='reduce-overhead')
    x = torch.randn(2, 10).cuda()
    _ = m(x)
    print('torch.compile: WORKING')
except Exception as e:
    print(f'torch.compile: FAILED ({e})')
"

echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "To run training:"
echo "  cd /mnt/f/editorbot"
echo "  bash run_wsl.sh"
echo ""
echo "Or with custom settings:"
echo "  SAVE_DIR=./models/my-model N_ENVS=4 bash run_wsl.sh"
