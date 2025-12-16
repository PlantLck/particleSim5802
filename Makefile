# Makefile for Parallel Particle Simulation (C++ Version)
# Linux and NVIDIA Jetson platforms only
# Supports: Standard build, MPI, CUDA

# ============================================================================
# Compiler Configuration
# ============================================================================

CXX := g++
NVCC := nvcc
MPICC := mpic++

# C++ standard and optimization
CXXFLAGS := -std=c++14 -Wall -O3 -march=native
NVCCFLAGS := -std=c++14 -O3 --expt-relaxed-constexpr

# Platform detection and configuration
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    
    # Check for Jetson
    ifneq (,$(wildcard /etc/nv_tegra_release))
        PLATFORM := jetson
        $(info Detected: NVIDIA Jetson platform)
    else
        $(info Detected: Linux platform)
    endif
    
    # SDL2 configuration using pkg-config
    SDL_CFLAGS := $(shell sdl2-config --cflags)
    SDL_LIBS := $(shell sdl2-config --libs) -lSDL2_ttf
else
    $(error Unsupported platform: $(UNAME_S). This project requires Linux or Jetson.)
endif

# OpenMP support (enabled by default)
OPENMP_FLAGS := -fopenmp

# CUDA architecture detection - IMPROVED
ifeq ($(PLATFORM),jetson)
    # Try to detect specific Jetson model
    JETSON_MODEL := $(shell cat /etc/nv_tegra_release 2>/dev/null | grep -oP '(?<=R)[0-9]+' | head -1)
    
    ifeq ($(JETSON_MODEL),32)
        # Jetson Nano, TX1, TX2 (Maxwell/Pascal)
        CUDA_ARCH := -arch=sm_53
        CXXFLAGS += -DJETSON_NANO
        $(info Detected: Jetson Nano/TX series - Using sm_53)
    else ifeq ($(JETSON_MODEL),35)
        # Jetson Orin
        CUDA_ARCH := -arch=sm_87
        $(info Detected: Jetson Orin series - Using sm_87)
    else
        # Default to Xavier (most common modern Jetson)
        CUDA_ARCH := -arch=sm_72
        $(info Detected: Jetson Xavier series - Using sm_72)
    endif
else
    # Desktop GPU - use common architecture
    CUDA_ARCH := -arch=sm_86
    $(info Using desktop GPU architecture: sm_86 (adjust if needed))
endif

# Allow manual override
ifdef CUDA_ARCH_OVERRIDE
    CUDA_ARCH := $(CUDA_ARCH_OVERRIDE)
    $(info Manual CUDA architecture override: $(CUDA_ARCH))
endif

# ============================================================================
# Source Files
# ============================================================================

SOURCES := Main.cpp Simulation.cpp Physics.cpp Rendering.cpp SystemMonitor.cpp
OBJECTS := $(SOURCES:.cpp=.o)

CUDA_SOURCE := PhysicsGPU.cu
CUDA_OBJECT := PhysicsGPU.o

# ============================================================================
# Build Targets
# ============================================================================

# Default target: Standard build with Sequential + OpenMP
.PHONY: all
all: particle_sim

particle_sim: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -lm
	@echo ""
	@echo "✓ Build complete: ./particle_sim"
	@echo "  Modes available: Sequential (1), Multithreaded (2)"
	@echo ""

# MPI build: Sequential + OpenMP + MPI
.PHONY: mpi
mpi: CXX := $(MPICC)
mpi: CXXFLAGS += -DUSE_MPI
mpi: particle_sim_mpi

particle_sim_mpi: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -lm
	@echo ""
	@echo "✓ MPI build complete: ./particle_sim_mpi"
	@echo "  Run with: mpirun -np 4 ./particle_sim_mpi"
	@echo "  Modes available: Sequential (1), Multithreaded (2), MPI (3)"
	@echo ""

# CUDA build: All modes including GPU
.PHONY: cuda
cuda: CXXFLAGS += -DUSE_CUDA
cuda: particle_sim_cuda

particle_sim_cuda: $(OBJECTS) $(CUDA_OBJECT)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -L/usr/local/cuda/lib64 -lcudart -lm
	@echo ""
	@echo "✓ CUDA build complete: ./particle_sim_cuda"
	@echo "  Modes available: Sequential (1), Multithreaded (2), GPU Simple (4), GPU Complex (5)"
	@echo "  GPU Architecture: $(CUDA_ARCH)"
	@echo ""

# CUDA + MPI build: All modes
.PHONY: cuda_mpi
cuda_mpi: CXX := $(MPICC)
cuda_mpi: CXXFLAGS += -DUSE_CUDA -DUSE_MPI
cuda_mpi: particle_sim_full

particle_sim_full: $(OBJECTS) $(CUDA_OBJECT)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -L/usr/local/cuda/lib64 -lcudart -lm
	@echo ""
	@echo "✓ Full build complete: ./particle_sim_full"
	@echo "  Run with: mpirun -np 4 ./particle_sim_full"
	@echo "  All 5 modes available"
	@echo "  GPU Architecture: $(CUDA_ARCH)"
	@echo ""

# ============================================================================
# Compilation Rules
# ============================================================================

%.o: %.cpp ParticleSimulation.hpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(SDL_CFLAGS) -c $< -o $@

$(CUDA_OBJECT): $(CUDA_SOURCE) ParticleSimulation.hpp
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

# ============================================================================
# Utility Targets
# ============================================================================

.PHONY: clean
clean:
	rm -f *.o particle_sim particle_sim_mpi particle_sim_cuda particle_sim_full
	@echo "✓ Clean complete"

.PHONY: help
help:
	@echo "Parallel Particle Simulation - Build System"
	@echo "============================================"
	@echo ""
	@echo "Standard Targets:"
	@echo "  make          - Build standard version (Sequential + OpenMP)"
	@echo "  make mpi      - Build with MPI support"
	@echo "  make cuda     - Build with CUDA GPU support"
	@echo "  make cuda_mpi - Build with all features (CUDA + MPI)"
	@echo ""
	@echo "Utility Targets:"
	@echo "  make clean    - Remove all build artifacts"
	@echo "  make help     - Show this help message"
	@echo "  make info     - Show build configuration"
	@echo "  make test     - Run quick functionality test"
	@echo ""
	@echo "Platform: $(PLATFORM)"
	@echo "Compiler: $(CXX)"
	@echo ""

.PHONY: test
test: all
	@echo "Running quick functionality test..."
	@echo "  (Launch simulation - press ESC to exit)"
	@timeout 5 ./particle_sim || true
	@echo "✓ Test complete"

# ============================================================================
# Information Target
# ============================================================================

.PHONY: info
info:
	@echo "Build Configuration"
	@echo "==================="
	@echo ""
	@echo "Platform Detection:"
	@echo "  System: $(UNAME_S)"
	@echo "  Platform: $(PLATFORM)"
	@echo ""
	@echo "Compilers:"
	@echo "  CXX: $(CXX)"
	@echo "  NVCC: $(shell which $(NVCC) 2>/dev/null || echo 'not found')"
	@echo "  MPICC: $(shell which $(MPICC) 2>/dev/null || echo 'not found')"
	@echo ""
	@echo "Compiler Flags:"
	@echo "  CXXFLAGS: $(CXXFLAGS)"
	@echo "  OPENMP_FLAGS: $(OPENMP_FLAGS)"
	@echo "  CUDA_ARCH: $(CUDA_ARCH)"
	@echo ""
	@echo "Libraries:"
	@echo "  SDL_CFLAGS: $(SDL_CFLAGS)"
	@echo "  SDL_LIBS: $(SDL_LIBS)"
	@echo ""
	@echo "Feature Detection:"
	@which $(CXX) > /dev/null 2>&1 && echo "  ✓ C++ Compiler" || echo "  ✗ C++ Compiler"
	@which $(NVCC) > /dev/null 2>&1 && echo "  ✓ CUDA Compiler" || echo "  ✗ CUDA Compiler"
	@which $(MPICC) > /dev/null 2>&1 && echo "  ✓ MPI Compiler" || echo "  ✗ MPI Compiler"
	@pkg-config --exists sdl2 2>/dev/null && echo "  ✓ SDL2" || echo "  ✗ SDL2"
	@pkg-config --exists SDL2_ttf 2>/dev/null && echo "  ✓ SDL2_ttf" || echo "  ✗ SDL2_ttf"
	@echo ""
	@echo "CUDA Information:"
	@if which nvcc > /dev/null 2>&1; then \
		nvcc --version | grep "release"; \
	else \
		echo "  CUDA not installed"; \
	fi
	@echo ""
ifeq ($(PLATFORM),jetson)
	@echo "Jetson Information:"
	@cat /etc/nv_tegra_release 2>/dev/null || echo "  Unable to read Jetson version"
	@echo ""
	@echo "Current Power Mode:"
	@sudo nvpmodel -q 2>/dev/null | grep "NV Power Mode" || echo "  Unable to read power mode (requires sudo)"
endif

# ============================================================================
# Jetson-Specific Targets
# ============================================================================

ifeq ($(PLATFORM),jetson)
.PHONY: jetson-max-performance
jetson-max-performance:
	@echo "Setting Jetson to maximum performance mode..."
	@sudo nvpmodel -m 0
	@sudo jetson_clocks
	@echo "✓ Performance mode enabled"
	@echo "  Verify with: sudo nvpmodel -q"

.PHONY: jetson-status
jetson-status:
	@echo "Jetson System Status"
	@echo "===================="
	@echo ""
	@echo "Power Mode:"
	@sudo nvpmodel -q 2>/dev/null || echo "  Unable to read (requires sudo)"
	@echo ""
	@echo "Temperature:"
	@cat /sys/devices/virtual/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf "  %.1f°C\n", $$1/1000}' || echo "  Unable to read"
	@echo ""
	@echo "GPU Stats:"
	@tegrastats --interval 500 --stop || echo "  Unable to read (tegrastats not available)"
endif

# ============================================================================
# Development Targets
# ============================================================================

.PHONY: debug
debug: CXXFLAGS := -std=c++14 -Wall -g -O0
debug: NVCCFLAGS := -std=c++14 -g -G -O0
debug: all
	@echo "✓ Debug build complete with symbols"

.PHONY: profile
profile: CXXFLAGS += -pg
profile: all
	@echo "✓ Profile build complete (use gprof for analysis)"

# ============================================================================
# Installation Target (Optional)
# ============================================================================

PREFIX ?= /usr/local
BINDIR := $(PREFIX)/bin

.PHONY: install
install: cuda
	@echo "Installing to $(BINDIR)..."
	@mkdir -p $(BINDIR)
	@cp particle_sim_cuda $(BINDIR)/particle_sim
	@chmod 755 $(BINDIR)/particle_sim
	@echo "✓ Installation complete"
	@echo "  Run with: particle_sim"

.PHONY: uninstall
uninstall:
	@echo "Removing from $(BINDIR)..."
	@rm -f $(BINDIR)/particle_sim
	@echo "✓ Uninstallation complete"

# ============================================================================
# Documentation
# ============================================================================

.PHONY: docs
docs:
	@echo "Available Documentation:"
	@echo "  README.md           - Project overview and quick start"
	@echo "  JETSON_INSTALLATION.md - Complete Jetson setup guide"
	@echo "  DOCUMENTATION.md    - Technical architecture"
	@echo "  CHANGELOG.md        - Version history"

# ============================================================================
# Notes
# ============================================================================

# Architecture Reference:
# - sm_53: Jetson Nano, TX1, TX2
# - sm_62: Jetson TX2
# - sm_72: Jetson Xavier NX, AGX Xavier
# - sm_75: Turing (GTX 16xx, RTX 20xx)
# - sm_86: Ampere (RTX 30xx, A100)
# - sm_87: Jetson Orin
# - sm_89: Ada Lovelace (RTX 40xx)

# To override CUDA architecture:
#   make cuda CUDA_ARCH_OVERRIDE="-arch=sm_53"

# For verbose compilation:
#   make V=1

# Platform-specific notes:
# - Jetson: Unified memory architecture, optimize for thermal constraints
# - Linux Desktop: Discrete GPU, ensure proper cooling
