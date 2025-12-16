# Makefile for Parallel Particle Simulation (C++ Version)
# Supports: Standard build, MPI, CUDA, and cross-platform compilation

# ============================================================================
# Compiler Configuration
# ============================================================================

CXX := g++
NVCC := nvcc
MPICC := mpic++

# C++ standard and optimization
CXXFLAGS := -std=c++17 -Wall -O3 -march=native
NVCCFLAGS := -std=c++17 -O3

# Platform detection and configuration
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    SDL_CFLAGS := $(shell sdl2-config --cflags)
    SDL_LIBS := $(shell sdl2-config --libs) -lSDL2_ttf
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
    SDL_CFLAGS := $(shell sdl2-config --cflags)
    SDL_LIBS := $(shell sdl2-config --libs) -lSDL2_ttf
else
    PLATFORM := windows
    # Windows paths (adjust based on your SDL2 installation)
    SDL_INCLUDE := -IC:/SDL2/include -IC:/SDL2_ttf/include
    SDL_LIBDIR := -LC:/SDL2/lib/x64 -LC:/SDL2_ttf/lib/x64
    SDL_LIBS := $(SDL_LIBDIR) -lSDL2 -lSDL2main -lSDL2_ttf
    SDL_CFLAGS := $(SDL_INCLUDE)
endif

# OpenMP support (enabled by default)
OPENMP_FLAGS := -fopenmp

# Detect CUDA architecture (for Jetson or Desktop GPU)
CUDA_ARCH := -arch=sm_86  # RTX 3050 / Xavier NX
# For Jetson Nano: -arch=sm_53
# For Jetson Xavier: -arch=sm_72
# For RTX 30xx/40xx: -arch=sm_86

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
	@echo "✓ Build complete: ./particle_sim"
	@echo "  Modes available: Sequential (1), Multithreaded (2)"

# MPI build: Sequential + OpenMP + MPI
.PHONY: mpi
mpi: CXX := $(MPICC)
mpi: CXXFLAGS += -DUSE_MPI
mpi: particle_sim_mpi

particle_sim_mpi: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -lm
	@echo "✓ MPI build complete: ./particle_sim_mpi"
	@echo "  Run with: mpirun -np 4 ./particle_sim_mpi"
	@echo "  Modes available: Sequential (1), Multithreaded (2), MPI (3)"

# CUDA build: All modes including GPU
.PHONY: cuda
cuda: CXXFLAGS += -DUSE_CUDA
cuda: particle_sim_cuda

particle_sim_cuda: $(OBJECTS) $(CUDA_OBJECT)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -L/usr/local/cuda/lib64 -lcudart -lm
	@echo "✓ CUDA build complete: ./particle_sim_cuda"
	@echo "  Modes available: Sequential (1), Multithreaded (2), GPU Simple (4), GPU Complex (5)"

# CUDA + MPI build: All modes
.PHONY: cuda_mpi
cuda_mpi: CXX := $(MPICC)
cuda_mpi: CXXFLAGS += -DUSE_CUDA -DUSE_MPI
cuda_mpi: particle_sim_full

particle_sim_full: $(OBJECTS) $(CUDA_OBJECT)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -L/usr/local/cuda/lib64 -lcudart -lm
	@echo "✓ Full build complete: ./particle_sim_full"
	@echo "  Run with: mpirun -np 4 ./particle_sim_full"
	@echo "  All 5 modes available"

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
	@echo ""
	@echo "Platform: $(PLATFORM)"
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"

.PHONY: test
test: all
	@echo "Running quick functionality test..."
	@echo "  (Launch simulation and test mode switching)"
	@./particle_sim || true

# ============================================================================
# Platform-Specific Notes
# ============================================================================

.PHONY: info
info:
	@echo "Build Configuration"
	@echo "==================="
	@echo "Platform: $(PLATFORM)"
	@echo "CXX: $(CXX)"
	@echo "NVCC: $(NVCC)"
	@echo "MPICC: $(MPICC)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "SDL_CFLAGS: $(SDL_CFLAGS)"
	@echo "SDL_LIBS: $(SDL_LIBS)"
	@echo "CUDA_ARCH: $(CUDA_ARCH)"
	@echo ""
	@echo "Detected Features:"
	@which $(CXX) > /dev/null 2>&1 && echo "  ✓ C++ Compiler" || echo "  ✗ C++ Compiler"
	@which $(NVCC) > /dev/null 2>&1 && echo "  ✓ CUDA Compiler" || echo "  ✗ CUDA Compiler"
	@which $(MPICC) > /dev/null 2>&1 && echo "  ✓ MPI Compiler" || echo "  ✗ MPI Compiler"
	@pkg-config --exists sdl2 && echo "  ✓ SDL2" || echo "  ✗ SDL2"
