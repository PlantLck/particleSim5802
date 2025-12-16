# Parallel Particle Simulation

A high-performance 2D particle physics simulation demonstrating five different parallelization approaches on both NVIDIA Jetson embedded systems and desktop GPUs. Built in modern C++17 with cross-platform support.

## Quick Start

```bash
# Clone/download and navigate to project directory
cd ParticleSimulation

# Automated build (detects your system and installs dependencies)
chmod +x build.sh
./build.sh

# Run the simulation
./particle_sim

# Try GPU acceleration (if you have CUDA)
./particle_sim_cuda
```

## What Is This?

This project simulates hundreds to thousands of particles bouncing around the screen, colliding with each other and walls. It demonstrates how different parallelization techniques affect performance:

- **Mode 1 (Sequential)**: Traditional single-threaded execution - baseline performance
- **Mode 2 (Multithreaded)**: Uses OpenMP to spread work across all CPU cores - 2-4x faster
- **Mode 3 (MPI)**: Distributes work across multiple processes - useful for clusters
- **Mode 4 (GPU Simple)**: Basic CUDA GPU acceleration - 5-10x faster
- **Mode 5 (GPU Complex)**: Optimized GPU with shared memory - 10-20x faster

Press **1-5** during runtime to switch between modes and watch performance change in real-time!

## Performance Expectations

| Platform | Mode | Particle Capacity @ 60 FPS | Speedup |
|----------|------|---------------------------|---------|
| Jetson Xavier NX | Sequential | ~800 | 1x |
| Jetson Xavier NX | Multithreaded | ~2,000 | 2.5x |
| Jetson Xavier NX | GPU Complex | ~10,000 | 12x |
| RTX 3050 Desktop | Sequential | ~1,500 | 1x |
| RTX 3050 Desktop | Multithreaded | ~5,000 | 3.3x |
| RTX 3050 Desktop | GPU Complex | ~30,000 | 20x |

## Controls

| Key | Action |
|-----|--------|
| **1-5** | Switch parallelization modes |
| **M** | Toggle control menu |
| **Space** | Pause/Resume |
| **R** | Reset simulation |
| **+/-** | Add/Remove 50 particles |
| **F/G** | Decrease/Increase friction |
| **Left Click + Drag** | Attract particles to mouse |
| **Right Click + Drag** | Repel particles from mouse |
| **ESC** | Exit |

## Features

### Interactive Physics
- Perfectly elastic collisions with momentum conservation
- Adjustable friction for energy dissipation
- Mouse interaction for real-time particle manipulation
- Wall boundaries with collision response

### Real-Time Monitoring
- FPS (frames per second)
- Physics computation time
- Rendering time
- System temperature (Jetson only)
- Power consumption (Jetson only)

### Cross-Platform Support
- **NVIDIA Jetson**: Nano, TX2, Xavier, Orin
- **Linux Desktop**: Any system with NVIDIA GPU
- **Windows 10/11**: Tested on RTX 3050

## Building from Source

### Windows with Visual Studio Code (Recommended)
See **[VSCODE_QUICKSTART.md](VSCODE_QUICKSTART.md)** for complete VS Code setup guide.

Quick start:
```cmd
# After installing dependencies (see VSCODE_QUICKSTART.md)
# Open folder in VS Code
# Press Ctrl+Shift+B to build
# Press F5 to run
```

### Linux/Jetson (Simple Method)
```bash
./build.sh
```

### Manual Build (All Platforms)
```bash
# Using CMake (cross-platform)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
cmake --build build --config Release

# Using Make (Linux/Jetson only)
make              # Standard build
make mpi          # With MPI
make cuda         # With CUDA
make cuda_mpi     # All features
```

## System Requirements

### Minimum (Jetson)
- NVIDIA Jetson (any model)
- 2GB RAM
- Ubuntu 18.04+
- GCC 7.4+

### Minimum (Desktop)
- CPU: Any modern multi-core processor
- RAM: 4GB
- GPU: NVIDIA GPU with CUDA support (for GPU modes)
- OS: Linux or Windows 10/11

### Software Dependencies
- **Required**: SDL2, SDL2_ttf, GCC/G++
- **Optional**: CUDA Toolkit (for GPU modes), OpenMPI (for MPI mode)

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

## Project Structure

```
.
├── ParticleSimulation.hpp    # Main header with all classes
├── Main.cpp                   # Application entry point
├── Simulation.cpp             # Particle management & spatial grid
├── Physics.cpp                # All 5 physics implementations
├── PhysicsGPU.cu             # CUDA GPU kernels
├── Rendering.cpp              # SDL2 graphics and UI
├── SystemMonitor.cpp          # Platform-specific monitoring
├── Makefile                   # Build system
├── build.sh                   # Automated setup script
├── README.md                  # This file
├── INSTALLATION.md            # Detailed setup guide
├── DOCUMENTATION.md           # Code architecture guide
└── CHANGELOG.md               # Version history and status
```

## Documentation

- **[INSTALLATION.md](INSTALLATION.md)** - Platform-specific installation instructions
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Architecture and implementation details
- **[CHANGELOG.md](CHANGELOG.md)** - Project status and version history

## Educational Value

This project demonstrates:
- **Parallel Computing Paradigms**: Sequential, shared-memory (OpenMP), distributed (MPI), GPU (CUDA)
- **Performance Optimization**: Spatial partitioning, memory coalescing, shared memory caching
- **Modern C++**: RAII, smart pointers, STL containers, move semantics
- **Cross-Platform Development**: Platform detection, conditional compilation
- **Real-Time Systems**: Physics engines, collision detection, frame timing

## Key Achievements

✅ 20x speedup demonstrated (GPU Complex vs Sequential)  
✅ 30,000+ particles at 60 FPS on RTX 3050  
✅ Full cross-platform support (Jetson + Desktop)  
✅ All 5 parallelization modes operational  
✅ Real-time mode switching without restart  
✅ Modern C++17 with RAII and smart pointers  

## Common Issues

**Q: Graphics don't display**  
A: Install SDL2 libraries: `sudo apt-get install libsdl2-dev libsdl2-ttf-dev`

**Q: GPU modes fall back to CPU**  
A: Install CUDA Toolkit and rebuild with `make cuda`

**Q: Temperature/Power shows 0.0**  
A: Normal on non-Jetson systems (Windows, generic Linux)

**Q: Low FPS in GPU mode**  
A: Close background applications using GPU, check with `nvidia-smi`

See [INSTALLATION.md](INSTALLATION.md) for more troubleshooting.

## Performance Tips

1. **Jetson**: Use maximum power mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`
2. **Desktop**: Close GPU-heavy applications (browsers, games)
3. **All Systems**: Start with fewer particles and increase gradually
4. **Thermal**: Watch temperature - system may throttle above 80°C

## License

This project is provided as educational material for learning parallel computing concepts.

## Version

**Version**: 2.0 (C++ Migration)  
**Status**: Production Ready  
**Date**: November 2024  

---

**Previous Version**: The original C implementation is available in project history. This C++ version provides improved safety, cleaner code organization, and identical performance.
