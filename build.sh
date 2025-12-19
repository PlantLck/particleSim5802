#!/bin/bash
# Automated Build and Setup Script for Parallel Particle Simulation
# Linux and NVIDIA Jetson platforms only

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Parallel Particle Simulation - Build Script${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

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

echo ""

echo "Checking for required tools..."

if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1)
    echo -e "${GREEN}✓ g++ found: $GCC_VERSION${NC}"
else
    echo -e "${RED}✗ g++ not found - please install build tools${NC}"
    echo -e "  Install with: ${CYAN}sudo apt-get install build-essential${NC}"
    exit 1
fi

if command -v make &> /dev/null; then
    MAKE_VERSION=$(make --version | head -n1)
    echo -e "${GREEN}✓ make found: $MAKE_VERSION${NC}"
else
    echo -e "${RED}✗ make not found - please install build tools${NC}"
    echo -e "  Install with: ${CYAN}sudo apt-get install build-essential${NC}"
    exit 1
fi

echo ""

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

echo "Checking for optional features..."

MPI_AVAILABLE=false
CUDA_AVAILABLE=false

if command -v mpic++ &> /dev/null; then
    MPI_VERSION=$(mpic++ --version | head -n1)
    echo -e "${GREEN}✓ MPI found: $MPI_VERSION${NC}"
    MPI_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ MPI not found - MPI mode will not be available${NC}"
    echo -e "  Install with: ${CYAN}sudo apt-get install libopenmpi-dev openmpi-bin${NC}"
fi

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo -e "${GREEN}✓ CUDA found: version $CUDA_VERSION${NC}"
    CUDA_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ CUDA not found - GPU modes will not be available${NC}"
fi

echo ""

echo -e "${BLUE}Select build configuration:${NC}"
echo "  1) Standard (Sequential + OpenMP)"
echo "  2) With MPI support"
echo "  3) With CUDA GPU support (recommended for Jetson)"
echo "  4) Full build (MPI + CUDA)"
echo "  5) All available features (auto-detect)"
echo ""

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

echo -e "${BLUE}Building project...${NC}"

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
        
        if make; then
            BUILD_SUCCESS=true
        fi
        
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo ""
            echo -e "${BLUE}Building CUDA version...${NC}"
            make cuda
        fi
        
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
    echo -e "  ${GREEN}✓${NC} ./particle_sim_cuda (+ GPU) ${CYAN}<- Recommended${NC}"
    EXECUTABLES_FOUND=true
fi

if [ -f "./particle_sim_full" ]; then
    echo -e "  ${GREEN}✓${NC} ./particle_sim_full (All modes)"
    EXECUTABLES_FOUND=true
fi

echo ""
echo -e "${CYAN}Controls:${NC}"
echo "  [1-5] Switch parallelization modes"
echo "  [M]   Toggle menu"
echo "  [+/-] Add/remove particles"
echo "  [UP/DOWN] or [Scroll] Adjust mouse force"
echo "  [LMB] Attract (hold and drag)"
echo "  [RMB] Repel (hold and drag)"
echo "  [R]   Reset simulation"
echo "  [ESC] Exit"

echo ""
echo -e "${GREEN}Setup complete! Ready to run.${NC}"
