#!/bin/bash

# LongCat-Image API Server Launcher Script
# This script helps launch the API server with proper configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
T2I_CHECKPOINT="${T2I_CHECKPOINT:-./weights/LongCat-Image}"
EDIT_CHECKPOINT="${EDIT_CHECKPOINT:-./weights/LongCat-Image-Edit}"
USE_CPU_OFFLOAD="${USE_CPU_OFFLOAD:-true}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "======================================================"
    echo "  LongCat-Image OpenAI-Compatible API Server"
    echo "======================================================"
    echo -e "${NC}"
}

check_python() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}✗ Python not found. Please install Python 3.10+${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION}${NC}"
}

check_dependencies() {
    echo -e "\n${BLUE}Checking dependencies...${NC}"
    
    local deps=("fastapi" "uvicorn" "torch" "transformers" "PIL" "longcat_image")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if python -c "import ${dep}" 2>/dev/null; then
            echo -e "${GREEN}✓ ${dep}${NC}"
        else
            echo -e "${RED}✗ ${dep}${NC}"
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "\n${YELLOW}Missing dependencies: ${missing_deps[*]}${NC}"
        echo -e "Install with: ${YELLOW}pip install -r requirements.txt${NC}"
        echo -e "             ${YELLOW}pip install fastapi uvicorn${NC}"
        return 1
    fi
    
    return 0
}

check_models() {
    echo -e "\n${BLUE}Checking models...${NC}"
    
    if [ -d "$T2I_CHECKPOINT" ]; then
        echo -e "${GREEN}✓ T2I model found at ${T2I_CHECKPOINT}${NC}"
    else
        echo -e "${RED}✗ T2I model not found at ${T2I_CHECKPOINT}${NC}"
        echo -e "${YELLOW}  Download with: huggingface-cli download meituan-longcat/LongCat-Image --local-dir ${T2I_CHECKPOINT}${NC}"
        return 1
    fi
    
    if [ -d "$EDIT_CHECKPOINT" ]; then
        echo -e "${GREEN}✓ Edit model found at ${EDIT_CHECKPOINT}${NC}"
    else
        echo -e "${RED}✗ Edit model not found at ${EDIT_CHECKPOINT}${NC}"
        echo -e "${YELLOW}  Download with: huggingface-cli download meituan-longcat/LongCat-Image-Edit --local-dir ${EDIT_CHECKPOINT}${NC}"
        return 1
    fi
    
    return 0
}

check_cuda() {
    echo -e "\n${BLUE}Checking GPU...${NC}"
    
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        echo -e "${GREEN}✓ CUDA available (${GPU_COUNT} GPU(s))${NC}"
        echo -e "  ${GREEN}GPU: ${GPU_NAME}${NC}"
    else
        echo -e "${YELLOW}⚠ CUDA not available - inference will be slow${NC}"
    fi
}

show_config() {
    echo -e "\n${BLUE}Configuration:${NC}"
    echo -e "  API Host: ${GREEN}${API_HOST}${NC}"
    echo -e "  API Port: ${GREEN}${API_PORT}${NC}"
    echo -e "  T2I Checkpoint: ${GREEN}${T2I_CHECKPOINT}${NC}"
    echo -e "  Edit Checkpoint: ${GREEN}${EDIT_CHECKPOINT}${NC}"
    echo -e "  CPU Offload: ${GREEN}${USE_CPU_OFFLOAD}${NC}"
    echo -e "  Max Batch Size: ${GREEN}${MAX_BATCH_SIZE}${NC}"
}

show_usage() {
    echo -e "\n${BLUE}API Usage:${NC}"
    echo -e "  • Interactive Docs: ${GREEN}http://${API_HOST}:${API_PORT}/docs${NC}"
    echo -e "  • API Base: ${GREEN}http://${API_HOST}:${API_PORT}${NC}"
    echo -e "\n${BLUE}Endpoints:${NC}"
    echo -e "  • POST   /v1/images/generations    - Generate images from text"
    echo -e "  • POST   /v1/images/edits          - Edit images"
    echo -e "  • GET    /v1/models                - List models"
    echo -e "  • GET    /v1/health                - Health check"
    echo -e "\n${BLUE}Python Example:${NC}"
    echo -e "  ${GREEN}python examples_api_usage.py${NC}"
}

start_server() {
    echo -e "\n${BLUE}Starting API server...${NC}\n"
    
    export API_HOST="${API_HOST}"
    export API_PORT="${API_PORT}"
    export T2I_CHECKPOINT="${T2I_CHECKPOINT}"
    export EDIT_CHECKPOINT="${EDIT_CHECKPOINT}"
    export USE_CPU_OFFLOAD="${USE_CPU_OFFLOAD}"
    export MAX_BATCH_SIZE="${MAX_BATCH_SIZE}"
    
    python api_server.py
}

# Main execution
main() {
    print_header
    
    # Run checks
    check_python || exit 1
    check_dependencies || { echo -e "\n${RED}Please install missing dependencies${NC}"; exit 1; }
    check_models || { echo -e "\n${RED}Please download missing models${NC}"; exit 1; }
    check_cuda
    
    show_config
    show_usage
    
    # Start server
    trap 'echo -e "\n${YELLOW}Server stopped${NC}"; exit 0' INT TERM
    start_server
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            API_HOST="$2"
            shift 2
            ;;
        --port)
            API_PORT="$2"
            shift 2
            ;;
        --t2i-checkpoint)
            T2I_CHECKPOINT="$2"
            shift 2
            ;;
        --edit-checkpoint)
            EDIT_CHECKPOINT="$2"
            shift 2
            ;;
        --no-cpu-offload)
            USE_CPU_OFFLOAD="false"
            shift
            ;;
        --max-batch-size)
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./launch_api.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host <address>              API host (default: 0.0.0.0)"
            echo "  --port <port>                 API port (default: 8000)"
            echo "  --t2i-checkpoint <path>       T2I model path"
            echo "  --edit-checkpoint <path>      Edit model path"
            echo "  --no-cpu-offload              Disable CPU offload"
            echo "  --max-batch-size <size>       Max batch size (default: 1)"
            echo "  --help                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./launch_api.sh"
            echo "  ./launch_api.sh --port 8080 --host 127.0.0.1"
            echo "  ./launch_api.sh --no-cpu-offload --max-batch-size 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main
main
