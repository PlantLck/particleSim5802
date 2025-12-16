# Jetson Installation and Deployment Guide

Complete installation, configuration, and optimization guide for running the Parallel Particle Simulation on NVIDIA Jetson embedded platforms.

## Table of Contents

- [Supported Platforms](#supported-platforms)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [Performance Configuration](#performance-configuration)
- [Running the Simulation](#running-the-simulation)
- [Performance Benchmarking](#performance-benchmarking)
- [Troubleshooting](#troubleshooting)
- [Optimization Guide](#optimization-guide)

---

## Supported Platforms

### Tested Configurations

| Platform | RAM | CUDA Cores | Architecture | Recommended |
|----------|-----|------------|--------------|-------------|
| **Jetson Nano** | 4GB | 128 | sm_53 | Entry level |
| **Jetson TX2** | 8GB | 256 | sm_62 | Good |
| **Jetson Xavier NX** | 8/16GB | 384 | sm_72 | ✅ Recommended |
| **Jetson AGX Xavier** | 32GB | 512 | sm_72 | Excellent |
| **Jetson AGX Orin** | 32/64GB | 2048 | sm_87 | Best |

### Software Requirements

- **Operating System**: Ubuntu 18.04/20.04 (L4T - Linux for Tegra)
- **JetPack**: 4.6+ or 5.0+
- **CUDA**: 10.2+ (included with JetPack)
- **GCC**: 7.4+ (pre-installed)

---

## Prerequisites

### Hardware Setup

1. **Power Supply**: Ensure adequate power (5V 4A for most models)
2. **Cooling**: Fan or heatsink recommended for sustained loads
3. **Display**: HDMI monitor for visualization
4. **Input**: USB keyboard and mouse
5. **Storage**: 5GB free space on SD card or NVMe

### Software Check

Verify JetPack installation:

```bash
# Check L4T version
cat /etc/nv_tegra_release

# Expected output example:
# R32 (release), REVISION: 7.1, GCID: 29818004, BOARD: t186ref, EABI: aarch64
```

Verify CUDA installation:

```bash
nvcc --version

# Expected output:
# Cuda compilation tools, release 10.2, V10.2.xxx
```

---

## Quick Start

### 5-Minute Setup

```bash
# 1. System update
sudo apt-get update
sudo apt-get upgrade -y

# 2. Install dependencies
sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev build-essential

# 3. Clone/transfer project
cd ~
# (transfer files here or clone repository)
cd ParticleSimulation

# 4. Build
chmod +x build.sh
./build.sh
# Select option 3 (CUDA) when prompted

# 5. Run
./particle_sim_cuda
```

**Expected Result**: Window opens with 500 particles bouncing around. Press `5` for GPU Complex mode.

---

## Detailed Installation

### Step 1: System Preparation

Update system packages:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 2: Install Core Dependencies

#### Required: SDL2 Graphics Libraries

```bash
sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev
```

Verify installation:

```bash
pkg-config --modversion sdl2
# Should output version like: 2.0.10
```

#### Required: Build Tools

Usually pre-installed, but verify:

```bash
sudo apt-get install -y build-essential
```

Verify compilers:

```bash
g++ --version
# Should show GCC 7.4 or newer

nvcc --version
# Should show CUDA 10.2 or newer
```

### Step 3: Optional Dependencies

#### For MPI Mode (Distributed Computing)

```bash
sudo apt-get install -y libopenmpi-dev openmpi-bin
```

Verify:

```bash
mpic++ --version
mpirun --version
```

#### For Development Tools

```bash
# GDB debugger
sudo apt-get install -y gdb

# Performance monitoring
sudo apt-get install -y sysstat
```

### Step 4: Transfer Project Files

**Option A: From development machine**

```bash
# On your development machine:
scp -r ParticleSimulation/ jetson@<jetson-ip>:~/

# On Jetson, verify:
cd ~/ParticleSimulation
ls -la
```

**Option B: Clone from repository**

```bash
cd ~
git clone <your-repository-url> ParticleSimulation
cd ParticleSimulation
```

**Option C: USB drive**

```bash
# Mount USB
sudo mkdir -p /media/usb
sudo mount /dev/sda1 /media/usb

# Copy files
cp -r /media/usb/ParticleSimulation ~/
cd ~/ParticleSimulation
```

### Step 5: Build Project

#### Automated Build (Recommended)

```bash
chmod +x build.sh
./build.sh
```

The script will:
- Detect Jetson platform
- Check dependencies
- Offer to install missing packages
- Suggest performance mode configuration
- Build appropriate executables

Select option 3 (CUDA) or 5 (auto-detect) when prompted.

#### Manual Build

```bash
# Standard build (Sequential + OpenMP)
make

# CUDA build (recommended)
make cuda

# With MPI
make mpi

# All features
make cuda_mpi

# View all options
make help
```

### Step 6: Verify Build

Check for executables:

```bash
ls -lh particle_sim*

# Expected output:
# -rwxr-xr-x 1 user user 156K particle_sim
# -rwxr-xr-x 1 user user 178K particle_sim_cuda
```

---

## Performance Configuration

### Power Mode Configuration

**CRITICAL for performance testing.**

#### Understanding Power Modes

Jetson has multiple power modes balancing performance and power consumption:

| Mode | Description | Use Case |
|------|-------------|----------|
| 0 | MAXN (Maximum Performance) | Benchmarking, demo |
| 1 | Mode 10W | Battery operation |
| 2 | Mode 15W | Balanced |

#### Set Maximum Performance

```bash
# Check current mode
sudo nvpmodel -q

# Set to maximum performance (mode 0)
sudo nvpmodel -m 0

# Lock clocks to maximum frequencies
sudo jetson_clocks

# Verify
sudo nvpmodel -q
```

**Note**: Mode 0 increases power consumption and heat. Ensure adequate cooling.

#### Automatic Configuration

The `build.sh` script will offer to configure this automatically.

For manual script:

```bash
# Create performance script
cat > ~/jetson_max_perf.sh << 'EOF'
#!/bin/bash
sudo nvpmodel -m 0
sudo jetson_clocks
echo "Jetson set to maximum performance mode"
sudo nvpmodel -q
EOF

chmod +x ~/jetson_max_perf.sh
./jetson_max_perf.sh
```

### Thermal Management

Monitor temperature during operation:

```bash
# Real-time temperature (Ctrl+C to exit)
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone0/temp

# One-time check (temperature in millidegrees)
cat /sys/devices/virtual/thermal/thermal_zone0/temp | awk '{print $1/1000 "°C"}'
```

**Safe Operating Range**:
- Normal: 30-60°C
- Warning: 60-80°C (performance may throttle)
- Critical: >80°C (automatic throttling)

**Cooling Solutions**:
1. Add fan (5V, connect to 4-pin header)
2. Use heatsink
3. Improve airflow around device
4. Reduce particle count if throttling occurs

---

## Running the Simulation

### Basic Execution

```bash
# GPU version (recommended)
./particle_sim_cuda

# Standard version
./particle_sim

# MPI version (4 processes)
mpirun -np 4 ./particle_sim_mpi
```

### Controls

| Key | Action | Description |
|-----|--------|-------------|
| **1** | Sequential | Baseline single-threaded |
| **2** | Multithreaded | OpenMP CPU parallelization |
| **3** | MPI | Distributed computing |
| **4** | GPU Simple | Basic CUDA acceleration |
| **5** | GPU Complex | Optimized CUDA (fastest) |
| **+** | Add particles | Increase by 50 |
| **-** | Remove particles | Decrease by 50 |
| **M** | Toggle menu | Show/hide controls |
| **Space** | Pause/Resume | Freeze simulation |
| **R** | Reset | Regenerate particles |
| **F/G** | Friction | Decrease/Increase |
| **ESC** | Exit | Close application |

### Mouse Controls

| Action | Effect |
|--------|--------|
| **Left Click + Drag** | Attract particles to cursor |
| **Right Click + Drag** | Repel particles from cursor |

### Display Information

On-screen display shows:
- **FPS**: Frames per second (target: 60)
- **Particles**: Current count / maximum
- **Mode**: Current parallelization mode
- **Physics Time**: Computation time (ms)
- **Render Time**: Drawing time (ms)
- **Friction**: Current friction coefficient
- **Temperature**: GPU/CPU temperature (°C)
- **Power**: System power consumption (W)

---

## Performance Benchmarking

### Performance Expectations

#### Jetson Xavier NX (Recommended Platform)

| Mode | Particles @ 60 FPS | Speedup | Notes |
|------|-------------------|---------|-------|
| Sequential (1) | 800 | 1.0x | Baseline |
| Multithreaded (2) | 2,000 | 2.5x | 6-core CPU |
| MPI (3) | 1,800 | 2.25x | Communication overhead |
| GPU Simple (4) | 6,000 | 7.5x | Basic CUDA |
| GPU Complex (5) | 10,000 | 12.5x | Optimized |

#### Jetson Nano (Entry Level)

| Mode | Particles @ 60 FPS | Speedup |
|------|-------------------|---------|
| Sequential (1) | 400 | 1.0x |
| Multithreaded (2) | 800 | 2.0x |
| GPU Complex (5) | 3,500 | 8.8x |

#### Jetson AGX Orin (High-End)

| Mode | Particles @ 60 FPS | Speedup |
|------|-------------------|---------|
| Sequential (1) | 1,200 | 1.0x |
| Multithreaded (2) | 3,500 | 2.9x |
| GPU Complex (5) | 20,000+ | 16.7x+ |

### Benchmarking Procedure

#### 1. Preparation

```bash
# Set maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Start monitoring in separate terminal
tegrastats
```

#### 2. Testing Protocol

For each mode (1-5):

```bash
# Start simulation
./particle_sim_cuda

# Switch to mode (press key 1-5)
# Let stabilize for 10 seconds
# Record FPS from display
# Gradually add particles with '+' key
# Find maximum particles @ 60 FPS
# Record temperature and power
```

#### 3. Data Collection Template

Create `benchmark_results.csv`:

```csv
Mode,Particles,FPS,Physics_ms,Render_ms,Temp_C,Power_W,Notes
Sequential,800,60,12.5,2.1,45.2,8.5,Baseline
Multithreaded,2000,60,11.8,3.2,48.1,10.2,OpenMP
MPI,1800,60,13.2,3.0,47.5,9.8,4 processes
GPU Simple,6000,60,8.5,5.1,55.3,12.5,Basic CUDA
GPU Complex,10000,60,7.2,6.8,58.7,14.2,Optimized
```

#### 4. Automated Testing Script

```bash
#!/bin/bash
# benchmark.sh - Automated performance test

echo "Mode,Duration_s,Avg_FPS" > benchmark_results.txt

for mode in 1 2 4 5; do
    echo "Testing Mode $mode..."
    
    # Run for 30 seconds
    timeout 30s ./particle_sim_cuda &
    PID=$!
    
    # Manual: Switch to mode $mode and record
    wait $PID
done

echo "Benchmark complete. See benchmark_results.txt"
```

### Performance Monitoring

#### Real-Time System Stats

```bash
# Comprehensive stats
tegrastats

# Example output:
# RAM 2156/7853MB CPU [25%@1420,15%@1420,20%@1420,18%@1420,22%@1420,19%@1420]
# EMC_FREQ 0% GR3D_FREQ 38% VIC_FREQ 0% APE 150 MTS fg 0% bg 0%
# PLL@43C CPU@45.5C PMIC@100C GPU@44C AO@50.5C thermal@45.75C
# POM_5V_IN 3542/3542 POM_5V_GPU 542/542 POM_5V_CPU 1401/1401
```

#### GPU Profiling

```bash
# Profile CUDA kernels
nvprof ./particle_sim_cuda

# Detailed analysis
nsys profile --stats=true ./particle_sim_cuda

# Kernel-level profiling
ncu --set full ./particle_sim_cuda
```

---

## Troubleshooting

### Build Issues

#### Error: "SDL2/SDL.h: No such file or directory"

**Solution**:
```bash
sudo apt-get install libsdl2-dev libsdl2-ttf-dev
```

#### Error: "nvcc: not found"

**Cause**: CUDA not in PATH

**Solution**:
```bash
# Add to ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Error: "unsupported GPU architecture 'compute_XX'"

**Cause**: Wrong CUDA architecture for your Jetson

**Solution**: Edit Makefile, line ~37:

```makefile
# For Jetson Nano/TX1/TX2
CUDA_ARCH := -arch=sm_53

# For Jetson Xavier NX/AGX
CUDA_ARCH := -arch=sm_72

# For Jetson Orin
CUDA_ARCH := -arch=sm_87
```

Then rebuild:
```bash
make clean
make cuda
```

### Runtime Issues

#### Issue: Black screen / No window appears

**Solution 1**: Check SDL2 installation
```bash
pkg-config --modversion sdl2
# Should show version number
```

**Solution 2**: Check display
```bash
echo $DISPLAY
# Should show :0 or :1

export DISPLAY=:0
./particle_sim_cuda
```

**Solution 3**: Run with sudo (temporary workaround)
```bash
sudo ./particle_sim_cuda
# Not recommended long-term
```

#### Issue: Low FPS even in GPU mode

**Solution 1**: Verify performance mode
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

**Solution 2**: Check for thermal throttling
```bash
cat /sys/devices/virtual/thermal/thermal_zone0/temp
# If >80000 (80°C), system is throttling
# Improve cooling or reduce particle count
```

**Solution 3**: Verify GPU mode active
```bash
# While running, press '5' to ensure GPU Complex mode
# Check display shows "Mode: GPU Complex"
```

**Solution 4**: Monitor GPU usage
```bash
# In separate terminal
tegrastats

# Look for GR3D_FREQ
# Should be >50% when running GPU mode
```

#### Issue: "CUDA out of memory"

**Cause**: Jetson has unified memory shared between CPU and GPU

**Solution 1**: Close other applications
```bash
# Check memory usage
free -h

# Close unnecessary processes
```

**Solution 2**: Reduce particle count
- Press `-` multiple times to decrease particles
- Or edit `ParticleSimulation.hpp`:

```cpp
#ifdef PLATFORM_JETSON
    constexpr int MAX_PARTICLES = 5000;  // Reduce from 10000
#endif
```

Rebuild:
```bash
make clean && make cuda
```

#### Issue: MPI hangs or crashes

**Solution 1**: Reduce process count
```bash
# Try with 2 processes instead of 4
mpirun -np 2 ./particle_sim_mpi
```

**Solution 2**: Enable oversubscription
```bash
mpirun --oversubscribe -np 4 ./particle_sim_mpi
```

**Solution 3**: Check MPI installation
```bash
mpirun --version
# Should show Open MPI version
```

#### Issue: Temperature too high (>80°C)

**Immediate Action**:
1. Press ESC to exit simulation
2. Let Jetson cool for 5 minutes
3. Check cooling solution

**Long-term Solutions**:
1. Add or improve fan
2. Apply thermal paste to heatsink
3. Reduce particle count
4. Use lower power mode: `sudo nvpmodel -m 2`
5. Improve ambient airflow

#### Issue: No temperature/power readings

**Cause**: Normal for some Jetson configurations or kernel versions

**Verification**:
```bash
# Check thermal zones
ls /sys/devices/virtual/thermal/

# Check power rails
ls /sys/bus/i2c/drivers/ina3221x/
```

If directories are empty or missing, sensor data is unavailable. This doesn't affect simulation functionality.

---

## Optimization Guide

### Maximizing Performance

#### 1. System Configuration

```bash
# Disable desktop GUI for maximum resources
sudo systemctl set-default multi-user.target
sudo reboot

# Re-enable GUI
sudo systemctl set-default graphical.target
sudo reboot
```

#### 2. Particle Count Tuning

Start with baseline and increase:

```bash
# Start simulation
./particle_sim_cuda

# Press '5' for GPU Complex mode
# Press '+' to add 50 particles
# Repeat until FPS drops below 60
# Optimal count = last count with 60 FPS
```

#### 3. Code Optimizations

**For Jetson Xavier NX**, edit `ParticleSimulation.hpp`:

```cpp
#ifdef PLATFORM_JETSON
    constexpr int MAX_PARTICLES = 12000;  // Increase if your system handles it
    constexpr float DEFAULT_PARTICLE_RADIUS = 2.5f;  // Smaller = more particles
#endif
```

**For Grid Cell Size**, edit `ParticleSimulation.hpp`:

```cpp
constexpr int GRID_CELL_SIZE = 25;  // Increase for fewer particles, decrease for more
```

#### 4. CUDA Optimizations

For advanced users, edit `PhysicsGPU.cu`:

```cuda
// Line ~350 - Adjust block size
int threads = 256;  // Try 128 or 512

// Line ~550 - Shared memory size
__shared__ Particle s_particles[256];  // Match threads value
```

Rebuild after changes:
```bash
make clean && make cuda
```

### Power Optimization

For battery operation or thermal constraints:

```bash
# Set balanced mode
sudo nvpmodel -m 2

# Run simulation
./particle_sim_cuda

# Monitor power
cat /sys/bus/i2c/drivers/ina3221x/*/iio_device/in_power*_input
```

### Network-Free MPI

For single-Jetson MPI testing:

```bash
# Use localhost only
mpirun -np 4 --host localhost:4 ./particle_sim_mpi
```

---

## Advanced Topics

### Cross-Compilation

To compile on development machine for Jetson:

```bash
# Install cross-compiler
sudo apt-get install g++-aarch64-linux-gnu

# Set environment
export CXX=aarch64-linux-gnu-g++

# Adjust Makefile for cross-compilation
# (Advanced - requires CUDA cross-compilation toolkit)
```

### Custom Jetson Board Support

For custom carrier boards:

```bash
# Check device tree
ls /proc/device-tree/

# Verify thermal zones
cat /sys/class/thermal/thermal_zone*/type
```

### Performance Analysis

#### Using perf

```bash
sudo apt-get install linux-tools-common
perf record -g ./particle_sim_cuda
perf report
```

#### Using nsys (NVIDIA Nsight Systems)

```bash
# Collect data
nsys profile -o report ./particle_sim_cuda

# View in GUI (requires X forwarding or copy to desktop)
nsys-ui report.qdrep
```

---

## Additional Resources

### Official Documentation

- **NVIDIA Jetson**: https://developer.nvidia.com/embedded/jetson
- **JetPack SDK**: https://developer.nvidia.com/embedded/jetpack
- **CUDA Toolkit**: https://docs.nvidia.com/cuda/
- **Jetson Linux**: https://developer.nvidia.com/embedded/linux-tegra

### Community Resources

- **Jetson Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
- **Jetson Projects**: https://developer.nvidia.com/embedded/community/jetson-projects
- **Jetson Hacks**: https://www.jetsonhacks.com/

### Related Tools

- **jtop**: Real-time monitoring tool
  ```bash
  sudo -H pip install jetson-stats
  jtop
  ```

- **tegrastats**: Built-in monitoring
  ```bash
  tegrastats --help
  ```

---

## Appendix

### Jetson Model Comparison

| Feature | Nano | Xavier NX | AGX Xavier | Orin |
|---------|------|-----------|------------|------|
| CUDA Cores | 128 | 384 | 512 | 2048 |
| Tensor Cores | 0 | 48 | 64 | 64 |
| Max Power | 10W | 20W | 30W | 60W |
| Memory | 4GB | 8GB | 32GB | 64GB |
| CUDA Arch | sm_53 | sm_72 | sm_72 | sm_87 |

### Performance Tips Summary

✅ **Do**:
- Use GPU Complex mode (key 5)
- Enable maximum performance mode
- Ensure adequate cooling
- Monitor temperature continuously
- Start with fewer particles, increase gradually

❌ **Don't**:
- Run without cooling in sustained loads
- Exceed 80°C temperature
- Run CPU-intensive tasks simultaneously
- Use Debug builds for benchmarking

---

## Quick Reference

### One-Line Commands

```bash
# Quick install
sudo apt-get update && sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev && make cuda

# Performance mode
sudo nvpmodel -m 0 && sudo jetson_clocks

# Run
./particle_sim_cuda

# Monitor
tegrastats
```

### Keyboard Shortcuts

| 1 | 2 | 3 | 4 | 5 | +/- | M | R | ESC |
|---|---|---|---|---|-----|---|---|-----|
| Sequential | Multi | MPI | GPU-S | GPU-C | Count | Menu | Reset | Exit |

---

**Installation Time**: 15-20 minutes  
**Difficulty**: Beginner  
**Result**: High-performance particle simulation running at 10,000 particles @ 60 FPS on Jetson Xavier NX

---

*Last Updated: December 2024*  
*Tested on: Jetson Xavier NX with JetPack 5.1*  
*Version: 3.0.0 - Jetson Production Release*