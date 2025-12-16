#!/bin/bash
# Automated Build and Setup Script for Parallel Particle Simulation (C++)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Parallel Particle Simulation - C++ Version${NC}"
echo -e "${BLUE}Build and Setup Script${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# ============================================================================
# Platform Detection
# ============================================================================

echo "Detecting platform..."

PLATFORM="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    if [ -f /etc/nv_tegra_release ]; then
        PLATFORM="jetson"
        echo -e "${GREEN}✓ NVIDIA Jetson platform detected${NC}"
        cat /etc/nv_tegra_release
    else
        echo -e "${GREEN}✓ Linux platform detected${NC}"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    echo -e "${GREEN}✓ macOS platform detected${NC}"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="windows"
    echo -e "${GREEN}✓ Windows platform detected${NC}"
fi

echo ""

# ============================================================================
# Compiler Checks
# ============================================================================

echo "Checking for required tools..."

# Check for C++ compiler
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1)
    echo -e "${GREEN}✓ g++ found: $GCC_VERSION${NC}"
else
    echo -e "${RED}✗ g++ not found - please install build tools${NC}"
    exit 1
fi

# Check for make
if command -v make &> /dev/null; then
    echo -e "${GREEN}✓ make found${NC}"
else
    echo -e "${RED}✗ make not found - please install build tools${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Dependency Checks
# ============================================================================

echo "Checking for SDL2 libraries..."

SDL2_FOUND=false
SDL2_TTF_FOUND=false

if pkg-config --exists sdl2 2>/dev/null; then
    SDL2_VERSION=$(pkg-config --modversion sdl2)
    echo -e "${GREEN}✓ SDL2 found: version $SDL2_VERSION${NC}"
    SDL2_FOUND=true
elif [ -f /usr/include/SDL2/SDL.h ] || [ -f /usr/local/include/SDL2/SDL.h ]; then
    echo -e "${GREEN}✓ SDL2 headers found${NC}"
    SDL2_FOUND=true
else
    echo -e "${YELLOW}⚠ SDL2 not found${NC}"
fi

if pkg-config --exists SDL2_ttf 2>/dev/null; then
    SDL2_TTF_VERSION=$(pkg-config --modversion SDL2_ttf)
    echo -e "${GREEN}✓ SDL2_ttf found: version $SDL2_TTF_VERSION${NC}"
    SDL2_TTF_FOUND=true
elif [ -f /usr/include/SDL2/SDL_ttf.h ] || [ -f /usr/local/include/SDL2/SDL_ttf.h ]; then
    echo -e "${GREEN}✓ SDL2_ttf headers found${NC}"
    SDL2_TTF_FOUND=true
else
    echo -e "${YELLOW}⚠ SDL2_ttf not found${NC}"
fi

# Offer to install if missing
if [ "$SDL2_FOUND" = false ] || [ "$SDL2_TTF_FOUND" = false ]; then
    if [ "$PLATFORM" = "linux" ] || [ "$PLATFORM" = "jetson" ]; then
        echo ""
        echo -e "${YELLOW}Required dependencies are missing.${NC}"
        read -p "Would you like to install them now? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Installing SDL2 dependencies...${NC}"
            sudo apt-get update
            sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev
            echo -e "${GREEN}✓ Dependencies installed${NC}"
        else
            echo -e "${RED}Cannot build without required dependencies${NC}"
            exit 1
        fi
    fi
fi

echo ""

# ============================================================================
# Optional Dependencies
# ============================================================================

echo "Checking for optional features..."

# Check for MPI
if command -v mpic++ &> /dev/null; then
    MPI_VERSION=$(mpic++ --version | head -n1)
    echo -e "${GREEN}✓ MPI found: $MPI_VERSION${NC}"
    MPI_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ MPI not found - MPI mode will not be available${NC}"
    echo -e "  Install with: ${BLUE}sudo apt-get install libopenmpi-dev${NC}"
    MPI_AVAILABLE=false
fi

# Check for CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo -e "${GREEN}✓ CUDA found: version $CUDA_VERSION${NC}"
    CUDA_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ CUDA not found - GPU modes will not be available${NC}"
    CUDA_AVAILABLE=false
fi

echo ""

# ============================================================================
# Build Selection
# ============================================================================

echo -e "${BLUE}Select build configuration:${NC}"
echo "  1) Standard (Sequential + OpenMP) - Recommended for first build"
echo "  2) With MPI support"
echo "  3) With CUDA GPU support"
echo "  4) Full build (MPI + CUDA)"
echo "  5) All available features (auto-detect)"
echo ""

# Default to option 5 for automated builds
if [ -z "$BUILD_OPTION" ]; then
    read -p "Enter choice [1-5] (default: 5): " BUILD_OPTION
    BUILD_OPTION=${BUILD_OPTION:-5}
fi

echo ""

# ============================================================================
# Build Process
# ============================================================================

echo -e "${BLUE}Building project...${NC}"

make clean > /dev/null 2>&1 || true

case $BUILD_OPTION in
    1)
        echo -e "${BLUE}Building standard version...${NC}"
        make
        ;;
    2)
        if [ "$MPI_AVAILABLE" = true ]; then
            echo -e "${BLUE}Building with MPI...${NC}"
            make mpi
        else
            echo -e "${RED}MPI not available, building standard version${NC}"
            make
        fi
        ;;
    3)
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo -e "${BLUE}Building with CUDA...${NC}"
            make cuda
        else
            echo -e "${RED}CUDA not available, building standard version${NC}"
            make
        fi
        ;;
    4)
        if [ "$MPI_AVAILABLE" = true ] && [ "$CUDA_AVAILABLE" = true ]; then
            echo -e "${BLUE}Building full version (MPI + CUDA)...${NC}"
            make cuda_mpi
        else
            echo -e "${RED}MPI or CUDA not available, building standard version${NC}"
            make
        fi
        ;;
    5)
        echo -e "${BLUE}Auto-detecting and building all available features...${NC}"
        make
        if [ "$MPI_AVAILABLE" = true ]; then
            make mpi
        fi
        if [ "$CUDA_AVAILABLE" = true ]; then
            make cuda
        fi
        if [ "$MPI_AVAILABLE" = true ] && [ "$CUDA_AVAILABLE" = true ]; then
            make cuda_mpi
        fi
        ;;
esac

echo ""

# ============================================================================
# System Information
# ============================================================================

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}System Information${NC}"
echo -e "${BLUE}================================================${NC}"

if [ -f /proc/cpuinfo ]; then
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -n1 | cut -d: -f2 | xargs)
    if [ -z "$CPU_MODEL" ]; then
        CPU_MODEL=$(grep "^Hardware" /proc/cpuinfo | cut -d: -f2 | xargs)
    fi
    if [ ! -z "$CPU_MODEL" ]; then
        echo "CPU: $CPU_MODEL"
    fi
fi

if [ -f /proc/meminfo ]; then
    TOTAL_RAM=$(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')
    echo "RAM: $TOTAL_RAM"
fi

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

echo ""

# ============================================================================
# Build Summary
# ============================================================================

echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

echo "Available executables:"
[ -f "./particle_sim" ] && echo -e "  ${GREEN}✓${NC} ./particle_sim (Sequential + Multithreaded)"
[ -f "./particle_sim_mpi" ] && echo -e "  ${GREEN}✓${NC} ./particle_sim_mpi (+ MPI)"
[ -f "./particle_sim_cuda" ] && echo -e "  ${GREEN}✓${NC} ./particle_sim_cuda (+ GPU)"
[ -f "./particle_sim_full" ] && echo -e "  ${GREEN}✓${NC} ./particle_sim_full (All modes)"

echo ""
echo "To run:"
echo -e "  ${BLUE}./particle_sim${NC}                    # Standard version"
[ -f "./particle_sim_mpi" ] && echo -e "  ${BLUE}mpirun -np 4 ./particle_sim_mpi${NC}  # MPI version"
[ -f "./particle_sim_cuda" ] && echo -e "  ${BLUE}./particle_sim_cuda${NC}               # GPU version"

echo ""
echo "Controls:"
echo "  [1-5] Switch parallelization modes"
echo "  [M]   Toggle menu"
echo "  [+/-] Add/remove particles"
echo "  [ESC] Exit"

echo ""
echo -e "${GREEN}Setup complete! Ready to run.${NC}"
