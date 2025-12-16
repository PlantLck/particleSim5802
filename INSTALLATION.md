# Installation Guide

Complete installation instructions for Parallel Particle Simulation across different platforms.

## Table of Contents

- [NVIDIA Jetson Installation](#nvidia-jetson-installation)
- [Linux Desktop Installation](#linux-desktop-installation)
- [Windows Desktop Installation](#windows-desktop-installation)
- [Dependency Details](#dependency-details)
- [Troubleshooting](#troubleshooting)

---

## NVIDIA Jetson Installation

### Prerequisites

Tested on: Jetson Nano, TX2, Xavier NX, AGX Xavier, AGX Orin

### Step 1: System Update

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 2: Install Dependencies

```bash
# Required: SDL2 libraries
sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev

# Required: Build tools
sudo apt-get install -y build-essential

# Optional: MPI for distributed mode
sudo apt-get install -y libopenmpi-dev openmpi-bin

# CUDA is pre-installed on Jetson OS
```

### Step 3: Download Project

```bash
# Transfer files to Jetson (from your computer)
scp -r ParticleSimulation/ username@jetson-ip:~/

# Or clone from repository
# git clone <your-repo> ParticleSimulation
```

### Step 4: Build

```bash
cd ParticleSimulation
chmod +x build.sh
./build.sh

# Choose option 5 for auto-detect
# Or manually:
make           # Standard build
make mpi       # With MPI
make cuda      # With GPU support
make cuda_mpi  # All features
```

### Step 5: Performance Mode

For maximum performance:

```bash
# Enable maximum power mode (0 = max performance)
sudo nvpmodel -m 0

# Lock clocks to maximum
sudo jetson_clocks

# Verify mode
sudo nvpmodel -q
```

### Step 6: Run

```bash
# Standard version (modes 1-2)
./particle_sim

# GPU version (modes 1-2, 4-5)
./particle_sim_cuda

# MPI version (modes 1-3)
mpirun -np 4 ./particle_sim_mpi

# Full version (all modes)
mpirun -np 4 ./particle_sim_full
```

---

## Linux Desktop Installation

### Tested On
- Ubuntu 20.04, 22.04
- Debian 11, 12
- Fedora 36+
- Arch Linux

### Prerequisites

- NVIDIA GPU with CUDA support (for GPU modes)
- 4GB+ RAM
- Multi-core CPU

### Step 1: Install Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update

# Required
sudo apt-get install -y build-essential libsdl2-dev libsdl2-ttf-dev

# Optional: MPI
sudo apt-get install -y libopenmpi-dev openmpi-bin

# Optional: CUDA (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads
# Follow NVIDIA's installation guide for your distribution
```

#### Fedora

```bash
sudo dnf install gcc gcc-c++ make SDL2-devel SDL2_ttf-devel

# Optional: MPI
sudo dnf install openmpi-devel

# Optional: CUDA
# Download from NVIDIA and follow their guide
```

#### Arch Linux

```bash
sudo pacman -S base-devel sdl2 sdl2_ttf

# Optional: MPI
sudo pacman -S openmpi

# Optional: CUDA
sudo pacman -S cuda
```

### Step 2: Verify CUDA Installation (Optional)

```bash
nvcc --version
nvidia-smi
```

If these commands work, you have CUDA installed correctly.

### Step 3: Build and Run

```bash
cd ParticleSimulation
chmod +x build.sh
./build.sh

# Run
./particle_sim        # Standard
./particle_sim_cuda   # With GPU
```

---

## Windows Desktop Installation

### Tested Configuration
- **OS**: Windows 10/11
- **GPU**: NVIDIA GeForce RTX 3050
- **CPU**: AMD Ryzen 5 5600X
- **RAM**: 32GB

### Prerequisites

You'll need:
1. Visual Studio 2022 (or 2019)
2. CUDA Toolkit 12.x
3. SDL2 libraries
4. SDL2_ttf library

### Step 1: Install Visual Studio 2022

1. Download from: https://visualstudio.microsoft.com/
2. Run installer
3. Select "Desktop development with C++"
4. Include: MSVC, Windows SDK, C++ CMake tools
5. Install

### Step 2: Install CUDA Toolkit

1. Download from: https://developer.nvidia.com/cuda-downloads
2. Select: Windows, x86_64, 10/11, exe (local)
3. Run installer
4. Follow prompts (default options are fine)
5. Verify installation:

```cmd
nvcc --version
nvidia-smi
```

### Step 3: Install SDL2 Libraries

#### Method A: Pre-built Libraries (Recommended)

1. **Download SDL2**:
   - Go to: https://www.libsdl.org/download-2.0.php
   - Download: SDL2-devel-2.x.x-VC.zip
   - Extract to: `C:\SDL2\`

2. **Download SDL2_ttf**:
   - Go to: https://www.libsdl.org/projects/SDL_ttf/
   - Download: SDL2_ttf-devel-2.x.x-VC.zip
   - Extract to: `C:\SDL2_ttf\`

Your directory structure should look like:
```
C:\SDL2\
  ├── include\
  ├── lib\
  │   └── x64\
  └── ...

C:\SDL2_ttf\
  ├── include\
  ├── lib\
  │   └── x64\
  └── ...
```

#### Method B: vcpkg Package Manager

```cmd
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install sdl2:x64-windows sdl2-ttf:x64-windows
.\vcpkg integrate install
```

### Step 4: Update MAX_PARTICLES (Optional)

For RTX 3050, you can handle more particles:

Edit `ParticleSimulation.hpp`, line ~23:
```cpp
#ifdef PLATFORM_WINDOWS
    constexpr int MAX_PARTICLES = 30000;  // Already set for desktop
#endif
```

### Step 5: Create Visual Studio Project

#### Option A: Using Provided .sln (If Available)

1. Open `ParticleSimulation.sln` in Visual Studio
2. Skip to Step 6

#### Option B: Create New Project

1. **File → New → Project**
2. Select "Empty Project"
3. Name: `ParticleSimulation`
4. Location: Your project directory

5. **Add Source Files**:
   - Right-click "Source Files" → Add → Existing Item
   - Add: `Main.cpp`, `Simulation.cpp`, `Physics.cpp`, `Rendering.cpp`, `SystemMonitor.cpp`
   - Add: `ParticleSimulation.hpp`

6. **Add CUDA File**:
   - Right-click `PhysicsGPU.cu` → Properties
   - Item Type: "CUDA C/C++"

7. **Project Configuration** (All Configurations, x64):
   
   **C/C++ → General → Additional Include Directories**:
   ```
   C:\SDL2\include;C:\SDL2_ttf\include;%(AdditionalIncludeDirectories)
   ```

   **C/C++ → Language**:
   - C++ Language Standard: `ISO C++17`
   - OpenMP Support: `Yes (/openmp)`

   **C/C++ → Preprocessor → Preprocessor Definitions** (add these):
   ```
   PLATFORM_WINDOWS;USE_CUDA;_CRT_SECURE_NO_WARNINGS
   ```

   **Linker → General → Additional Library Directories**:
   ```
   C:\SDL2\lib\x64;C:\SDL2_ttf\lib\x64;%(AdditionalLibraryDirectories)
   ```

   **Linker → Input → Additional Dependencies**:
   ```
   SDL2.lib;SDL2main.lib;SDL2_ttf.lib;cudart.lib;%(AdditionalDependencies)
   ```

   **Linker → System → SubSystem**: `Console (/SUBSYSTEM:CONSOLE)`

   **CUDA C/C++ → Device → Code Generation**:
   ```
   compute_86,sm_86
   ```
   (For RTX 3050. For other GPUs, see: https://developer.nvidia.com/cuda-gpus)

### Step 6: Build

1. Set Configuration to **Release** and **x64**
2. **Build → Build Solution** (F7)
3. Check for errors in Output window

### Step 7: Copy DLLs

Copy these DLLs to your `x64\Release\` folder:

```
C:\SDL2\lib\x64\SDL2.dll → x64\Release\SDL2.dll
C:\SDL2_ttf\lib\x64\SDL2_ttf.dll → x64\Release\SDL2_ttf.dll
C:\SDL2_ttf\lib\x64\libfreetype-6.dll → x64\Release\libfreetype-6.dll (if present)
```

### Step 8: Run

1. Press **F5** to run with debugging
2. Or run `x64\Release\ParticleSimulation.exe` directly

### Expected Performance

On RTX 3050 system:
- Sequential: ~1,500 particles @ 60 FPS
- Multithreaded: ~5,000 particles @ 60 FPS
- GPU Complex: ~30,000 particles @ 60 FPS (15-20x speedup)

---

## Dependency Details

### Required Dependencies

#### SDL2 (Simple DirectMedia Layer 2)
- **Purpose**: Window creation, graphics rendering, input handling
- **Version**: 2.0.12 or later
- **Website**: https://www.libsdl.org/
- **License**: zlib license (permissive)

#### SDL2_ttf (TrueType Font Library)
- **Purpose**: Text rendering for UI overlays
- **Version**: 2.0.15 or later
- **Website**: https://www.libsdl.org/projects/SDL_ttf/
- **License**: zlib license (permissive)
- **Note**: Gracefully degrades if unavailable (no text, but simulation works)

#### GCC/G++ Compiler
- **Purpose**: C++ compilation
- **Version**: GCC 7.4+ (C++17 support required)
- **Alternatives**: Clang 5.0+, MSVC 2019+

### Optional Dependencies

#### CUDA Toolkit
- **Purpose**: GPU acceleration (Modes 4 & 5)
- **Version**: CUDA 10.2+ (11.x or 12.x recommended)
- **Website**: https://developer.nvidia.com/cuda-toolkit
- **Platform**: Requires NVIDIA GPU
- **Note**: Without CUDA, GPU modes fall back to sequential

#### OpenMPI
- **Purpose**: Distributed processing (Mode 3)
- **Version**: 4.0.0 or later
- **Website**: https://www.open-mpi.org/
- **Note**: Without MPI, mode 3 falls back to sequential

#### OpenMP
- **Purpose**: CPU multithreading (Mode 2)
- **Version**: 3.0+ (usually included with GCC)
- **Note**: Built into most modern compilers

---

## Troubleshooting

### Compilation Issues

#### "SDL2/SDL.h: No such file or directory"

**Linux:**
```bash
sudo apt-get install libsdl2-dev
# Or check if headers are in /usr/include/SDL2/ or /usr/local/include/SDL2/
```

**Windows:**
- Verify SDL2 is in `C:\SDL2\`
- Check include directories in project settings

#### "undefined reference to SDL_main"

Add `SDL2main.lib` to linker input (Windows) or `-lSDL2main` (Linux)

#### "CUDA error: no kernel image available"

Wrong CUDA architecture specified. Check your GPU compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Then update Makefile or VS project settings.

### Runtime Issues

#### Black Screen / No Window

1. **Check SDL2 installation**:
   ```bash
   ldconfig -p | grep SDL2    # Linux
   ```

2. **Check display**:
   ```bash
   echo $DISPLAY    # Linux - should show :0 or :1
   ```

3. **Try running with sudo** (not recommended long-term):
   ```bash
   sudo ./particle_sim
   ```

#### No Text Displayed

- SDL2_ttf not installed or font file missing
- Simulation will work, just without text overlays
- Install: `sudo apt-get install libsdl2-ttf-dev`

#### GPU Mode Shows "Fallback to CPU"

1. **CUDA not compiled**:
   ```bash
   ./particle_sim_cuda    # Make sure you built CUDA version
   ```

2. **CUDA driver/runtime issue**:
   ```bash
   nvidia-smi    # Should show GPU information
   nvcc --version    # Should show CUDA version
   ```

#### Low Performance / Thermal Throttling

**Jetson:**
```bash
# Check current mode
sudo nvpmodel -q

# Set to max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor temperature
watch -n 1 cat /sys/class/thermal/thermal_zone0/temp
```

**Desktop:**
```bash
# Monitor GPU
nvidia-smi -l 1

# Close background applications
# Check GPU isn't being used by browser, games, etc.
```

#### Permission Denied on build.sh

```bash
chmod +x build.sh
./build.sh
```

### MPI Issues

#### "mpirun not found"

```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin

# Fedora
sudo dnf install openmpi

# Then ensure it's in PATH
export PATH=$PATH:/usr/lib64/openmpi/bin
```

#### MPI Hangs or Crashes

- Try with fewer processes: `mpirun -np 2 ./particle_sim_mpi`
- Check network interfaces: `ifconfig`
- Disable oversubscription: `mpirun --oversubscribe -np 4 ./particle_sim_mpi`

### Platform-Specific Issues

#### Jetson: "Failed to allocate CUDA memory"

Jetson shares memory between CPU and GPU. Reduce particle count or close other applications.

#### Windows: "MSVCR120.dll missing"

Install Visual C++ Redistributables: https://aka.ms/vs/17/release/vc_redist.x64.exe

#### macOS: Not Officially Supported

The code may compile on macOS with NVIDIA eGPU but this is untested. SDL2 and OpenMP should work fine. CUDA support depends on eGPU setup.

---

## Verification

After installation, verify everything works:

```bash
# Test standard build
./particle_sim
# Press '1' (Sequential mode)
# Should show particles moving
# FPS should be displayed
# Press ESC to exit

# Test multithreaded
./particle_sim
# Press '2' (Multithreaded mode)
# FPS should improve significantly
# Check CPU usage with 'htop' - should show multiple cores active

# Test GPU (if CUDA build)
./particle_sim_cuda
# Press '5' (GPU Complex mode)
# FPS should be much higher
# Check with 'nvidia-smi' - GPU usage should increase

# Test MPI (if MPI build)
mpirun -np 4 ./particle_sim_mpi
# Should launch 4 processes
# Only rank 0 shows graphics
# Check with 'htop' - should see 4 processes
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check System Requirements**: Ensure your platform meets minimum requirements
2. **Review Error Messages**: Read compiler/runtime errors carefully
3. **Check Logs**: Look for additional error information in terminal output
4. **Verify Dependencies**: Use package manager to verify installations
5. **Clean Build**: Try `make clean && make` to rebuild from scratch

Common command for clean rebuild:
```bash
make clean
rm -f *.o particle_sim*
./build.sh
```

---

## Next Steps

Once installed successfully:

1. Read [README.md](README.md) for controls and features
2. Read [DOCUMENTATION.md](DOCUMENTATION.md) to understand the code
3. Try all 5 modes and compare performance
4. Experiment with particle counts and parameters
5. Monitor system metrics (temperature, power, FPS)
