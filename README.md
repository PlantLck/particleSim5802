# Parallel Particle Simulation on Edge Devices

**Author**: Max Locke  
**Course**: Introduction to Parallel Programming and Algorithms / Honors Project  
**Platform**: NVIDIA Jetson

## Overview

This project implements a real-time particle physics simulation featuring five distinct parallelization approaches, designed to explore the performance characteristics and practical limitations of different parallel computing paradigms on embedded hardware. The simulation visualizes thousands of particles interacting through elastic collisions, wall constraints, friction, and user-controlled forces, with the unique capability to switch between parallelization modes at runtime for direct performance comparison.

## Key Features

- **Five Parallelization Modes**: Sequential baseline, multithreaded (OpenMP), distributed (MPI), basic GPU (CUDA), and optimized GPU with spatial grid
- **Real-Time Mode Switching**: Compare performance characteristics without restarting the application
- **Comprehensive Performance Metrics**: Live FPS, physics timing, render timing, temperature, and power consumption
- **Interactive Controls**: Dynamic particle spawning, adjustable force magnitude, and intuitive mouse interaction
- **Spatial Partitioning**: O(n) collision detection using optimized spatial grid implementation

## System Requirements

### Minimum Hardware (Jetson)
- NVIDIA Jetson (any model: Nano, TX2, Xavier NX, AGX Xavier, Orin)
- 2GB RAM minimum (4GB+ recommended)
- Ubuntu 18.04 or later (L4T operating system)

### Software Dependencies

**Required**:
- GCC 7.4+ or compatible C++17 compiler
- SDL2 library (graphics and windowing)
- SDL2_ttf library (text rendering)

**Optional** (enables specific modes):
- CUDA Toolkit 10.2+ (enables GPU modes 4 and 5)
- OpenMPI 4.0+ (enables distributed mode 3)
- OpenMP 3.0+ (typically included with GCC, enables mode 2)

### Quick Start (Jetson/Linux)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev build-essential

# Build the project
chmod +x build.sh
./build.sh  # Interactive build script with auto-detection

# Or use Make directly
make cuda   # Recommended: builds with GPU support

# Run the simulation
./particle_sim_cuda
```

## Project Structure

```
ParticleSimulation/
├── ParticleSimulation.hpp    # Central header with all class definitions
├── Main.cpp                   # Application entry point and main loop
├── Simulation.cpp             # Particle management and spatial grid implementation
├── Physics.cpp                # All five physics mode implementations
├── PhysicsGPU.cu             # CUDA kernels for GPU modes
├── Rendering.cpp              # SDL2 graphics and UI overlay
├── SystemMonitor.cpp          # Platform-specific performance monitoring
├── Makefile                   # Build system with multiple targets
├── build.sh                   # Interactive build script
├── README.md                  # This file
```

## Usage

### Controls

**Mode Selection**:
- `1` - Sequential mode
- `2` - Multithreaded mode (OpenMP)
- `3` - MPI mode (requires MPI build)
- `4` - GPU Simple mode (requires CUDA build)
- `5` - GPU Complex mode (requires CUDA build)

**Particle Control**:
- `+`/`=` - Add 50 particles
- `-` - Remove 50 particles
- `Left Mouse Button` - Hold and drag to attract particles
- `Right Mouse Button` - Hold and drag to repel particles
- `UP`/`DOWN` arrows or `Mouse Scroll` - Adjust force magnitude

**Application Control**:
- `SPACE` - Pause/Resume simulation
- `R` - Reset simulation
- `M` - Toggle mode selection menu
- `ESC` - Exit application

### Running with MPI

For distributed mode across multiple processes:

```bash
# Build with MPI support
make cuda_mpi

# Run with 4 processes
mpirun -np 4 ./particle_sim_full
```

## Performance Characteristics

Performance varies significantly based on hardware platform and parallelization mode. The following represents general scaling characteristics observed during testing.

Performance metrics are displayed in real-time during execution, allowing direct observation of parallelization effectiveness.

## Technical Highlights

### Spatial Grid Optimization
The simulation employs a spatial partitioning grid that reduces collision detection complexity from O(n²) to O(n) by checking only particles in nearby grid cells.

### GPU Memory Management
The GPU implementation leverages CUDA unified memory on Jetson platforms to minimize explicit data transfers.

### Interactive Mouse Force
The mouse force system allows click-and-drag interaction with adjustable force magnitude, providing intuitive control over particle behavior.

## Build Options

The Makefile provides multiple build targets:

```bash
make          # Standard: Sequential + OpenMP
make mpi      # Add MPI support
make cuda     # Add GPU support (recommended)
make cuda_mpi # Full build with all features
make clean    # Remove build artifacts
make help     # Display all available targets
make info     # Show build configuration
```
