#!/bin/bash
# Automated Build and Setup Script for Parallel Particle Simulation (C++)
# Linux and NVIDIA Jetson platforms only

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Parallel Particle Simulation - C++ Version${NC}"
echo -e "${BLUE}Build and Setup Script${NC}"
echo -e "${BLUE}Linux and Jetson Platforms${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# ============================================================================
# Platform Detection
# ============================================================================

echo "Detecting platform..."

PLATFORM="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f /etc/nv_tegra_release ]; then
        PLATFORM="jetson"
        echo -e "${GREEN}✓ NVIDIA Jetson platform detected${NC}"
        cat /etc/nv_tegra_release
    else
        PLATFORM="linux"
        echo -e "${GREEN}✓ Linux platform detected${NC}"
    fi
else
    echo -e "${RED}✗ Unsupported platform: $OSTYPE${NC}"
    echo -e "${RED}This project requires Linux or Jetson${NC}"
    exit 1
fi

echo ""

# ============================================================================
# System Information
# ============================================================================

echo -e "${CYAN}System Information:${NC}"
if [ -f /proc/cpuinfo ]; then
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -n1 | cut -d: -f2 | xargs)
    if [ -z "$CPU_MODEL" ]; then
        CPU_MODEL=$(grep "^Hardware" /proc/cpuinfo | cut -d: -f2 | xargs)
    fi
    if [ ! -z "$CPU_MODEL" ]; then
        echo "  CPU: $CPU_MODEL"
    fi
    
    CPU_CORES=$(grep -c ^processor /proc/cpuinfo)
    echo "  Cores: $CPU_CORES"
fi

if [ -f /proc/meminfo ]; then
    TOTAL_RAM=$(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')
    echo "  RAM: $TOTAL_RAM"
fi

if [ "$PLATFORM" = "jetson" ]; then
    JETSON_MODEL=$(cat /etc/nv_tegra_release | grep -oP '(?<=BOARD: )[^\s]+' || echo "Unknown")
    echo "  Jetson Model: $JETSON_MODEL"
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
    echo -e "  Install with: ${CYAN}sudo apt-get install build-essential${NC}"
    exit 1
fi

# Check for make
if command -v make &> /dev/null; then
    MAKE_VERSION=$(make --version | head -n1)
    echo -e "${GREEN}✓ make found: $MAKE_VERSION${NC}"
else
    echo -e "${RED}✗ make not found - please install build tools${NC}"
    echo -e "  Install with: ${CYAN}sudo apt-get install build-essential${NC}"
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

echo ""

# ============================================================================
# Optional Dependencies
# ============================================================================

echo "Checking for optional features..."

MPI_AVAILABLE=false
CUDA_AVAILABLE=false

# Check for MPI
if command -v mpic++ &> /dev/null; then
    MPI_VERSION=$(mpic++ --version | head -n1)
    echo -e "${GREEN}✓ MPI found: $MPI_VERSION${NC}"
    MPI_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ MPI not found - MPI mode will not be available${NC}"
    echo -e "  Install with: ${CYAN}sudo apt-get install libopenmpi-dev openmpi-bin${NC}"
fi

# Check for CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo -e "${GREEN}✓ CUDA found: version $CUDA_VERSION${NC}"
    CUDA_AVAILABLE=true
    
    # Additional CUDA info for Jetson
    if [ "$PLATFORM" = "jetson" ]; then
        CUDA_PATH=$(which nvcc | sed 's|/bin/nvcc||')
        echo -e "  CUDA Path: $CUDA_PATH"
    fi
else
    echo -e "${YELLOW}⚠ CUDA not found - GPU modes will not be available${NC}"
    if [ "$PLATFORM" = "jetson" ]; then
        echo -e "${RED}  WARNING: Jetson should have CUDA pre-installed with JetPack${NC}"
        echo -e "  Check your JetPack installation"
    fi
fi

# Check OpenMP support
if echo | g++ -fopenmp -x c++ -E - &> /dev/null; then
    echo -e "${GREEN}✓ OpenMP support detected${NC}"
else
    echo -e "${YELLOW}⚠ OpenMP not supported - multithreaded mode may not work${NC}"
fi

echo ""

# ============================================================================
# Jetson Performance Configuration
# ============================================================================

if [ "$PLATFORM" = "jetson" ]; then
    echo -e "${CYAN}Jetson Performance Configuration${NC}"
    
    # Check current power mode
    if command -v nvpmodel &> /dev/null; then
        CURRENT_MODE=$(sudo nvpmodel -q 2>/dev/null | grep "NV Power Mode" | awk '{print $NF}')
        echo "  Current power mode: $CURRENT_MODE"
        
        echo ""
        read -p "Enable maximum performance mode? (recommended for benchmarking) (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Setting maximum performance mode...${NC}"
            sudo nvpmodel -m 0
            sudo jetson_clocks
            echo -e "${GREEN}✓ Performance mode enabled${NC}"
            echo "  Note: This will increase power consumption and heat"
        fi
    else
        echo -e "${YELLOW}⚠ nvpmodel not found - cannot configure power mode${NC}"
    fi
    
    echo ""
fi

# ============================================================================
# Build Selection
# ============================================================================

echo -e "${BLUE}Select build configuration:${NC}"
echo "  1) Standard (Sequential + OpenMP)"
echo "  2) With MPI support"
echo "  3) With CUDA GPU support (recommended for Jetson)"
echo "  4) Full build (MPI + CUDA)"
echo "  5) All available features (auto-detect)"
echo ""

# Default to option 5 for automated builds
if [ -z "$BUILD_OPTION" ]; then
    if [ "$PLATFORM" = "jetson" ] && [ "$CUDA_AVAILABLE" = true ]; then
        echo -e "${CYAN}Auto-selecting option 3 (CUDA) for Jetson platform${NC}"
        BUILD_OPTION=3
    else
        read -p "Enter choice [1-5] (default: 5): " BUILD_OPTION
        BUILD_OPTION=${BUILD_OPTION:-5}
    fi
fi

echo ""

# ============================================================================
# Build Process
# ============================================================================

echo -e "${BLUE}Building project...${NC}"

# Clean previous builds
make clean > /dev/null 2>&1 || true

BUILD_SUCCESS=false

case $BUILD_OPTION in
    1)
        echo -e "${BLUE}Building standard version (Sequential + OpenMP)...${NC}"
        if make; then
            BUILD_SUCCESS=true
        fi
        ;;
    2)
        if [ "$MPI_AVAILABLE" = true ]; then
            echo -e "${BLUE}Building with MPI support...${NC}"
            if make mpi; then
                BUILD_SUCCESS=true
            fi
        else
            echo -e "${RED}MPI not available, building standard version${NC}"
            if make; then
                BUILD_SUCCESS=true
            fi
        fi
        ;;
    3)
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo -e "${BLUE}Building with CUDA GPU support...${NC}"
            if make cuda; then
                BUILD_SUCCESS=true
            fi
        else
            echo -e "${RED}CUDA not available, building standard version${NC}"
            if make; then
                BUILD_SUCCESS=true
            fi
        fi
        ;;
    4)
        if [ "$MPI_AVAILABLE" = true ] && [ "$CUDA_AVAILABLE" = true ]; then
            echo -e "${BLUE}Building full version (MPI + CUDA)...${NC}"
            if make cuda_mpi; then
                BUILD_SUCCESS=true
            fi
        else
            echo -e "${RED}MPI or CUDA not available${NC}"
            echo -e "${YELLOW}Building with available features...${NC}"
            if [ "$CUDA_AVAILABLE" = true ]; then
                make cuda
            elif [ "$MPI_AVAILABLE" = true ]; then
                make mpi
            else
                make
            fi
            BUILD_SUCCESS=true
        fi
        ;;
    5)
        echo -e "${BLUE}Auto-detecting and building all available features...${NC}"
        
        # Build standard version
        if make; then
            BUILD_SUCCESS=true
        fi
        
        # Build MPI version if available
        if [ "$MPI_AVAILABLE" = true ]; then
            echo ""
            echo -e "${BLUE}Building MPI version...${NC}"
            make mpi
        fi
        
        # Build CUDA version if available (recommended for Jetson)
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo ""
            echo -e "${BLUE}Building CUDA version...${NC}"
            make cuda
        fi
        
        # Build full version if both available
        if [ "$MPI_AVAILABLE" = true ] && [ "$CUDA_AVAILABLE" = true ]; then
            echo ""
            echo -e "${BLUE}Building full version (MPI + CUDA)...${NC}"
            make cuda_mpi
        fi
        ;;
    *)
        echo -e "${RED}Invalid option: $BUILD_OPTION${NC}"
        exit 1
        ;;
esac

if [ "$BUILD_SUCCESS" = false ]; then
    echo -e "${RED}Build failed!${NC}"
    echo "Check error messages above for details"
    exit 1
fi

echo ""

# ============================================================================
# GPU Information
# ============================================================================

if [ "$CUDA_AVAILABLE" = true ]; then
    echo -e "${CYAN}GPU Information:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        echo "  nvidia-smi not available"
        echo "  GPU: CUDA-capable device detected"
    fi
    echo ""
fi

# ============================================================================
# Build Summary
# ============================================================================

echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

echo "Available executables:"
EXECUTABLES_FOUND=false

if [ -f "./particle_sim" ]; then
    echo -e "  ${GREEN}✓${NC} ./particle_sim (Sequential + Multithreaded)"
    EXECUTABLES_FOUND=true
fi

if [ -f "./particle_sim_mpi" ]; then
    echo -e "  ${GREEN}✓${NC} ./particle_sim_mpi (+ MPI)"
    EXECUTABLES_FOUND=true
fi

if [ -f "./particle_sim_cuda" ]; then
    echo -e "  ${GREEN}✓${NC} ./particle_sim_cuda (+ GPU) ${CYAN}← Recommended${NC}"
    EXECUTABLES_FOUND=true
fi

if [ -f "./particle_sim_full" ]; then
    echo -e "  ${GREEN}✓${NC} ./particle_sim_full (All modes)"
    EXECUTABLES_FOUND=true
fi

if [ "$EXECUTABLES_FOUND" = false ]; then
    echo -e "  ${RED}No executables found - build may have failed${NC}"
fi

echo ""
echo "To run:"

if [ -f "./particle_sim_cuda" ]; then
    echo -e "  ${CYAN}./particle_sim_cuda${NC}               # GPU version (recommended)"
fi

if [ -f "./particle_sim" ]; then
    echo -e "  ${CYAN}./particle_sim${NC}                    # Standard version"
fi

if [ -f "./particle_sim_mpi" ]; then
    echo -e "  ${CYAN}mpirun -np 4 ./particle_sim_mpi${NC}  # MPI version (4 processes)"
fi

if [ -f "./particle_sim_full" ]; then
    echo -e "  ${CYAN}mpirun -np 4 ./particle_sim_full${NC} # Full version (all modes)"
fi

echo ""
echo -e "${CYAN}Controls:${NC}"
echo "  [1-5] Switch parallelization modes"
echo "  [M]   Toggle menu"
echo "  [+/-] Add/remove particles"
echo "  [R]   Reset simulation"
echo "  [ESC] Exit"

if [ "$PLATFORM" = "jetson" ]; then
    echo ""
    echo -e "${CYAN}Jetson Tips:${NC}"
    echo "  • Press '5' for GPU Complex mode (fastest)"
    echo "  • Monitor temperature with: tegrastats"
    echo "  • Ensure cooling is adequate"
    echo "  • Start with few particles and increase gradually"
fi

echo ""
echo -e "${GREEN}Setup complete! Ready to run.${NC}"

# ============================================================================
# Post-Build Verification
# ============================================================================

echo ""
read -p "Run a quick test? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "./particle_sim_cuda" ]; then
        echo -e "${BLUE}Launching particle_sim_cuda for 5 seconds...${NC}"
        timeout 5 ./particle_sim_cuda || true
    elif [ -f "./particle_sim" ]; then
        echo -e "${BLUE}Launching particle_sim for 5 seconds...${NC}"
        timeout 5 ./particle_sim || true
    fi
    echo -e "${GREEN}✓ Test complete${NC}"
fi

echo ""
echo -e "${BLUE}For more information, see:${NC}"
echo "  README.md - Project overview"
echo "  JETSON_INSTALLATION.md - Complete Jetson setup guide"
echo "  DOCUMENTATION.md - Technical details"