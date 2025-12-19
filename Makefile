# Makefile for Parallel Particle Simulation (C++ Version)
# Linux and NVIDIA Jetson platforms only

# ============================================================================
# Compiler Configuration
# ============================================================================

CXX := g++
NVCC := nvcc
MPICC := mpic++

CXXFLAGS := -std=c++14 -Wall -O3 -march=native
NVCCFLAGS := -std=c++14 -O3 --expt-relaxed-constexpr

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    
    ifneq (,$(wildcard /etc/nv_tegra_release))
        PLATFORM := jetson
        $(info Detected: NVIDIA Jetson platform)
        CXXFLAGS += -DPLATFORM_JETSON
    else
        $(info Detected: Linux platform)
        CXXFLAGS += -DPLATFORM_LINUX
    endif
    
    SDL_CFLAGS := $(shell sdl2-config --cflags)
    SDL_LIBS := $(shell sdl2-config --libs) -lSDL2_ttf
else
    $(error Unsupported platform: $(UNAME_S). This project requires Linux or Jetson.)
endif

OPENMP_FLAGS := -fopenmp

# CUDA architecture detection
ifeq ($(PLATFORM),jetson)
    JETSON_MODEL := $(shell cat /etc/nv_tegra_release 2>/dev/null | grep -oP '(?<=R)[0-9]+' | head -1)
    
    ifeq ($(JETSON_MODEL),32)
        CUDA_ARCH := -arch=sm_53
        CXXFLAGS += -DJETSON_NANO
        $(info Detected: Jetson Nano/TX series - Using sm_53)
    else ifeq ($(JETSON_MODEL),35)
        CUDA_ARCH := -arch=sm_87
        $(info Detected: Jetson Orin series - Using sm_87)
    else
        CUDA_ARCH := -arch=sm_72
        $(info Detected: Jetson Xavier series - Using sm_72)
    endif
else
    CUDA_ARCH := -arch=sm_86
    $(info Using desktop GPU architecture: sm_86 (adjust if needed))
endif

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

.PHONY: all
all: particle_sim

particle_sim: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -lm
	@echo ""
	@echo "Build complete: ./particle_sim"
	@echo "  Modes available: Sequential (1), Multithreaded (2)"
	@echo ""

.PHONY: mpi
mpi: CXX := $(MPICC)
mpi: CXXFLAGS += -DUSE_MPI
mpi: particle_sim_mpi

particle_sim_mpi: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -lm
	@echo ""
	@echo "MPI build complete: ./particle_sim_mpi"
	@echo "  Run with: mpirun -np 4 ./particle_sim_mpi"
	@echo "  Modes available: Sequential (1), Multithreaded (2), MPI (3)"
	@echo ""

.PHONY: cuda
cuda: CXXFLAGS += -DUSE_CUDA
cuda: particle_sim_cuda

particle_sim_cuda: $(OBJECTS) $(CUDA_OBJECT)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -L/usr/local/cuda/lib64 -lcudart -lm
	@echo ""
	@echo "CUDA build complete: ./particle_sim_cuda"
	@echo "  Modes available: Sequential (1), Multithreaded (2), GPU Simple (4), GPU Complex (5)"
	@echo "  GPU Architecture: $(CUDA_ARCH)"
	@echo ""

.PHONY: cuda_mpi
cuda_mpi: CXX := $(MPICC)
cuda_mpi: CXXFLAGS += -DUSE_CUDA -DUSE_MPI
cuda_mpi: particle_sim_full

particle_sim_full: $(OBJECTS) $(CUDA_OBJECT)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(SDL_LIBS) -L/usr/local/cuda/lib64 -lcudart -lm
	@echo ""
	@echo "Full build complete: ./particle_sim_full"
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
	@echo "Clean complete"

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
	@echo ""
	@echo "Platform: $(PLATFORM)"
	@echo "Compiler: $(CXX)"
	@echo ""

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

.PHONY: debug
debug: CXXFLAGS := -std=c++14 -Wall -g -O0
debug: NVCCFLAGS := -std=c++14 -g -G -O0
debug: all
	@echo "Debug build complete with symbols"
