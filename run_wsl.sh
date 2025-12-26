#!/bin/bash
# WSL Training Script for RL Audio Editor
# Enables torch.compile() for faster training (requires Linux/WSL)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== RL Audio Editor - WSL Training ===${NC}"

# Change to project directory (Windows F: drive)
cd /mnt/f/editorbot

# Check CUDA availability
echo -e "${YELLOW}Checking CUDA...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo -e "${RED}CUDA not available! Make sure you have:${NC}"
    echo "  1. WSL2 (not WSL1)"
    echo "  2. NVIDIA GPU drivers installed on Windows"
    echo "  3. PyTorch with CUDA support installed in WSL"
    exit 1
fi

# Test torch.compile
echo -e "${YELLOW}Testing torch.compile()...${NC}"
if python -c "import torch; m = torch.nn.Linear(10,10).cuda(); m = torch.compile(m, mode='reduce-overhead'); x = torch.randn(2,10).cuda(); m(x); print('torch.compile works!')"; then
    echo -e "${GREEN}torch.compile() is working!${NC}"
    COMPILE_FLAG="--compile"
else
    echo -e "${YELLOW}torch.compile() failed, running without it${NC}"
    COMPILE_FLAG=""
fi

# Default parameters (can be overridden via command line)
DATA_DIR="${DATA_DIR:-./training_data}"
SAVE_DIR="${SAVE_DIR:-./models/model-wsl}"
N_ENVS="${N_ENVS:-8}"
STEPS="${STEPS:-2048}"
EPOCHS="${EPOCHS:-5000}"
CHECKPOINT="${CHECKPOINT:-}"
BC_NPZ="${BC_NPZ:-}"
BC_WEIGHT="${BC_WEIGHT:-0.1}"

# Build command
CMD="python -m rl_editor.train \
    --data-dir $DATA_DIR \
    --save-dir $SAVE_DIR \
    --n-envs $N_ENVS \
    --steps-per-epoch $STEPS \
    --n-epochs $EPOCHS \
    --subprocess \
    $COMPILE_FLAG"

# Add checkpoint if specified
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Add BC dataset if specified
if [ -n "$BC_NPZ" ]; then
    CMD="$CMD --bc-mixed $BC_NPZ --bc-weight $BC_WEIGHT"
fi

# Add any extra args passed to script
CMD="$CMD $@"

echo -e "${GREEN}Running training...${NC}"
echo -e "${YELLOW}Command: $CMD${NC}"
echo ""

# Run training
eval $CMD
