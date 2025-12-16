# Visual Studio Code Setup Guide for Windows

Complete step-by-step guide to build and run the Parallel Particle Simulation using VS Code on Windows.

## System Requirements

- **OS**: Windows 10 or Windows 11
- **GPU**: NVIDIA GPU with CUDA support (RTX 3050 recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 5GB free space for tools and SDKs

## Installation Overview

Total time: ~45-60 minutes

1. Install Visual Studio Code (5 min)
2. Install C++ Build Tools (15 min)
3. Install CMake (5 min)
4. Install CUDA Toolkit (15 min)
5. Install SDL2 Libraries (5 min)
6. Configure Project (5 min)
7. Build and Run (5 min)

---

## Step 1: Install Visual Studio Code (5 minutes)

### Download and Install

1. Go to: https://code.visualstudio.com/
2. Download "Download for Windows" (Stable build)
3. Run the installer
4. **Important**: Check these options during installation:
   - ✅ Add "Open with Code" to context menu
   - ✅ Add to PATH
   - ✅ Register Code as an editor for supported file types

### Install Required Extensions

1. Open VS Code
2. Click Extensions icon (Ctrl+Shift+X) or left sidebar
3. Search and install these extensions:
   - **C/C++** (by Microsoft) - Essential
   - **CMake Tools** (by Microsoft) - Essential
   - **CMake** (by twxs) - Helpful for syntax highlighting

**Screenshot locations**: Extensions → Search → Install

---

## Step 2: Install C++ Build Tools (15 minutes)

You need a C++ compiler. You have two options:

### Option A: Visual Studio Build Tools (Recommended)

**Why**: Better Windows integration, CUDA works out of box

1. Go to: https://visualstudio.microsoft.com/downloads/
2. Scroll to "All Downloads" → "Tools for Visual Studio"
3. Download **Build Tools for Visual Studio 2022**
4. Run installer
5. Select **Desktop development with C++**
6. On the right panel, ensure these are checked:
   - ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools
   - ✅ Windows 10 SDK (or 11 SDK)
   - ✅ C++ CMake tools for Windows
7. Click Install (3-5 GB download)
8. Restart computer when done

**Verify Installation**:
```cmd
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cl
# Should show Microsoft C/C++ Compiler version
```

### Option B: MinGW-w64 (Alternative)

**Why**: Lighter weight, GCC on Windows

1. Go to: https://www.mingw-w64.org/downloads/
2. Download MSYS2 installer: https://www.msys2.org/
3. Install to `C:\msys64`
4. Open MSYS2 terminal
5. Run:
```bash
pacman -Syu
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake
```
6. Add to PATH: `C:\msys64\mingw64\bin`

**Verify**:
```cmd
g++ --version
# Should show GCC version
```

---

## Step 3: Install CMake (5 minutes)

CMake is the build system we'll use.

### Download and Install

1. Go to: https://cmake.org/download/
2. Download **Windows x64 Installer** (cmake-3.xx.x-windows-x86_64.msi)
3. Run installer
4. **Important**: Select "Add CMake to system PATH for all users"
5. Install

### Verify Installation

Open Command Prompt (Win+R, type `cmd`, Enter):
```cmd
cmake --version
```

Should show: `cmake version 3.27.x` or newer

---

## Step 4: Install CUDA Toolkit (15 minutes)

Required for GPU acceleration modes.

### Download CUDA

1. Go to: https://developer.nvidia.com/cuda-downloads
2. Select:
   - Operating System: **Windows**
   - Architecture: **x86_64**
   - Version: **10** (or 11)
   - Installer Type: **exe (local)**
3. Download (~3 GB)

### Install CUDA

1. Run installer
2. Choose **Custom** installation
3. Ensure these are selected:
   - ✅ CUDA Toolkit
   - ✅ Visual Studio Integration (if using VS Build Tools)
   - ✅ NSight Tools (optional, for profiling)
4. Install location: Default (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`)
5. Wait for installation (10-15 minutes)

### Verify CUDA Installation

```cmd
nvcc --version
# Should show: Cuda compilation tools, release 12.x

nvidia-smi
# Should show your GPU (RTX 3050)
```

**If nvidia-smi doesn't work**: Update GPU drivers from https://www.nvidia.com/drivers

---

## Step 5: Install SDL2 Libraries (5 minutes)

SDL2 is for graphics and window management.

### Download SDL2

1. **SDL2 Main Library**:
   - Go to: https://www.libsdl.org/download-2.0.php
   - Download: **SDL2-devel-2.x.x-VC.zip** (Visual C++ development libraries)
   - Extract to: `C:\SDL2`

2. **SDL2_ttf (Font Library)**:
   - Go to: https://www.libsdl.org/projects/SDL_ttf/
   - Download: **SDL2_ttf-devel-2.x.x-VC.zip**
   - Extract to: `C:\SDL2_ttf`

### Verify Directory Structure

Your directories should look like:
```
C:\SDL2\
├── include\
│   └── SDL.h (and other headers)
├── lib\
│   └── x64\
│       ├── SDL2.lib
│       └── SDL2main.lib
└── ...

C:\SDL2_ttf\
├── include\
│   └── SDL_ttf.h
├── lib\
│   └── x64\
│       └── SDL2_ttf.lib
└── ...
```

**Important**: The paths must be exactly `C:\SDL2` and `C:\SDL2_ttf` (our CMake file expects these).

---

## Step 6: Setup Project in VS Code (5 minutes)

### Open Project

1. Copy all project files to a directory (e.g., `C:\Projects\ParticleSimulation`)
2. Open VS Code
3. File → Open Folder → Select your project folder
4. VS Code will ask to configure CMake → Click **Yes** (or do it later)

### Configure CMake Kit

1. Press **Ctrl+Shift+P** (Command Palette)
2. Type: `CMake: Select a Kit`
3. Choose one:
   - **Visual Studio Build Tools 2022 Release - amd64** (if using VS Build Tools)
   - **GCC x.x.x** (if using MinGW)

### Configure Build

1. Press **Ctrl+Shift+P**
2. Type: `CMake: Configure`
3. Watch the output panel - should see:
   ```
   -- Building for Windows Desktop
   -- CUDA found - GPU modes available
   -- OpenMP found - Multithreaded mode enabled
   -- SDL2: SDL2;SDL2main;SDL2_ttf
   ```

**If you see errors**: Check the troubleshooting section below.

---

## Step 7: Build and Run (5 minutes)

### Method 1: Using VS Code Tasks (Recommended)

1. Press **Ctrl+Shift+B** (Build)
2. Select: **CMake: Build Release**
3. Wait for compilation (1-2 minutes first time)
4. Look for: `Build finished with exit code 0`

### Method 2: Using CMake Tools Extension

1. Click CMake icon in left sidebar
2. Click **Build** button (or **Build All**)
3. Wait for completion

### Run the Simulation

**Option A: Using VS Code Task**
1. Press **Ctrl+Shift+P**
2. Type: `Tasks: Run Task`
3. Select: **Run Simulation**

**Option B: Using Terminal**
```cmd
cd build\Release
particle_sim.exe
```

**Option C: Using Debug (F5)**
1. Press **F5**
2. Select configuration: **Run Simulation (Windows)**

### Expected Result

- Window opens showing particle simulation
- Particles bouncing around
- Stats displayed (FPS, particle count, etc.)
- Controls:
  - **1-5**: Switch modes
  - **M**: Toggle menu
  - **+/-**: Add/remove particles
  - **ESC**: Exit

---

## Troubleshooting

### CMake Configuration Fails

**Error**: `Could not find SDL2`

**Solution**:
1. Verify SDL2 is in `C:\SDL2` and `C:\SDL2_ttf`
2. Check folder structure has `include` and `lib\x64`
3. Edit `CMakeLists.txt` if you used different paths:
   ```cmake
   set(SDL2_INCLUDE_DIR "C:/YourPath/SDL2/include")
   ```

**Error**: `CUDA not found`

**Solution**:
1. Verify CUDA installed: `nvcc --version`
2. Add CUDA to PATH:
   - System Properties → Environment Variables
   - PATH → Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
3. Restart VS Code

**Error**: `No C++ compiler found`

**Solution**:
1. Verify compiler installed
2. For VS Build Tools, run from "Developer Command Prompt for VS 2022"
3. For MinGW, ensure `C:\msys64\mingw64\bin` is in PATH

### Build Fails

**Error**: `fatal error C1083: Cannot open include file: 'SDL2/SDL.h'`

**Solution**: CMake didn't find SDL2 correctly. See CMake errors above.

**Error**: `LINK : fatal error LNK1181: cannot open input file 'SDL2.lib'`

**Solution**:
1. Check `C:\SDL2\lib\x64\SDL2.lib` exists
2. Verify you're building for x64, not x86
3. Re-run CMake configuration

**Error**: CUDA compilation errors

**Solution**:
1. Check CUDA architecture matches your GPU:
   ```cmd
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```
2. Update CMakeLists.txt line:
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 86)  # For RTX 3050
   # Use 75 for RTX 20xx, 53 for older GPUs
   ```

### Runtime Fails

**Error**: `SDL2.dll not found`

**Solution**: Copy DLLs to executable directory:
```cmd
copy C:\SDL2\lib\x64\SDL2.dll build\Release\
copy C:\SDL2_ttf\lib\x64\SDL2_ttf.dll build\Release\
```

Better solution - add to PATH:
- System Properties → Environment Variables
- PATH → Add: `C:\SDL2\lib\x64;C:\SDL2_ttf\lib\x64`

**Error**: `The code execution cannot proceed because cudart64_XX.dll was not found`

**Solution**: CUDA DLLs not in PATH:
- PATH → Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`

**Error**: Window opens but crashes immediately

**Solution**:
1. Run from terminal to see error messages:
   ```cmd
   cd build\Release
   particle_sim.exe
   ```
2. Check temperature/power monitoring (Windows returns 0.0 - this is normal)
3. Try starting with fewer particles (edit code or use -/+ keys)

### Performance Issues

**Problem**: Low FPS even in GPU mode

**Solutions**:
1. Verify you're in Release build (not Debug)
2. Close GPU-heavy applications (Chrome, Discord)
3. Check GPU usage: `nvidia-smi -l 1`
4. Verify running GPU Complex mode (press 5)
5. Check thermal throttling: `nvidia-smi --query-gpu=temperature.gpu --format=csv`

**Problem**: Can't see performance difference between modes

**Solution**:
1. Increase particle count with +++ key
2. GPU shines with 2000+ particles
3. Check mode actually switched (displayed on screen)

---

## Building Different Configurations

### Release Build (Fast, Default)
```cmd
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
cmake --build build --config Release
```

### Debug Build (Slow, but debuggable)
```cmd
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON
cmake --build build-debug --config Debug
```

### Without CUDA (CPU only)
```cmd
cmake -S . -B build -DUSE_CUDA=OFF
cmake --build build --config Release
```

### With MPI (Advanced)
```cmd
# First install Microsoft MPI or MPICH for Windows
cmake -S . -B build -DUSE_MPI=ON -DUSE_CUDA=ON
cmake --build build --config Release
```

---

## VS Code Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl+Shift+B** | Build project |
| **F5** | Start debugging/running |
| **Ctrl+Shift+P** | Command palette |
| **Ctrl+`** | Toggle terminal |
| **Ctrl+Shift+E** | Explorer |
| **Ctrl+Shift+X** | Extensions |
| **F7** | Build (if configured) |

---

## Recommended VS Code Settings

Add to `.vscode/settings.json`:

```json
{
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "cmake.configureOnOpen": false,
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools"
}
```

---

## Next Steps After Successful Build

1. **Test All Modes**:
   - Press 1: Sequential (baseline)
   - Press 2: Multithreaded (should see ~3x speedup)
   - Press 5: GPU Complex (should see ~20x speedup!)

2. **Experiment**:
   - Use +/- to adjust particle count
   - Try 30,000 particles in GPU mode
   - Use mouse to attract/repel particles

3. **Monitor Performance**:
   - Watch FPS counter
   - Open Task Manager → Performance → GPU
   - Run `nvidia-smi -l 1` in another terminal

4. **Profile and Optimize**:
   - Use NSight Systems for GPU profiling
   - Check CMake output for optimization flags
   - Try different CUDA architectures

---

## Directory Structure After Build

```
ParticleSimulation/
├── build/
│   └── Release/
│       ├── particle_sim.exe  ← Your executable
│       ├── SDL2.dll          ← Copy here
│       └── SDL2_ttf.dll      ← Copy here
├── build-debug/
│   └── Debug/
│       └── particle_sim.exe  ← Debug version
├── .vscode/
│   ├── tasks.json
│   ├── launch.json
│   └── settings.json
├── CMakeLists.txt
├── *.cpp, *.hpp, *.cu
└── *.md (documentation)
```

---

## Common Windows-Specific Tips

### Performance Mode

For maximum performance:
1. NVIDIA Control Panel → Manage 3D Settings
2. Power Management Mode: **Prefer Maximum Performance**
3. Windows → Settings → System → Power → **Best Performance**

### Multiple GPU Systems

If you have integrated + NVIDIA GPU:
1. Right-click particle_sim.exe
2. "Run with graphics processor" → **High-performance NVIDIA processor**

### Firewall Issues

If MPI fails:
1. Windows Defender Firewall → Allow an app
2. Add `particle_sim.exe`
3. Allow on Private networks

---

## Advanced: Using Developer Command Prompt

Alternative to VS Code:

1. Start Menu → Search "Developer Command Prompt for VS 2022"
2. Navigate to project:
   ```cmd
   cd C:\Projects\ParticleSimulation
   ```
3. Build:
   ```cmd
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```
4. Run:
   ```cmd
   build\Release\particle_sim.exe
   ```

---

## Summary Checklist

Before you start, ensure you have:

- [ ] Visual Studio Code installed
- [ ] C++ Build Tools or MinGW installed
- [ ] CMake installed and in PATH
- [ ] CUDA Toolkit installed
- [ ] SDL2 in `C:\SDL2`
- [ ] SDL2_ttf in `C:\SDL2_ttf`
- [ ] Project files in a folder
- [ ] VS Code extensions installed (C/C++, CMake Tools)

After setup:

- [ ] CMake configures without errors
- [ ] Project builds successfully (Ctrl+Shift+B)
- [ ] Executable runs (F5 or from terminal)
- [ ] Window opens with particles
- [ ] GPU mode works (press 5)
- [ ] Can achieve 30,000 particles @ 60 FPS in GPU Complex mode

---

## Getting Help

If stuck:
1. Check error messages carefully
2. Review troubleshooting section
3. Verify each installation step
4. Check CMake output for clues
5. Ensure all paths are correct

---

**Estimated Time**: 45-60 minutes for fresh install
**Difficulty**: Intermediate
**Result**: High-performance particle simulation running in VS Code on Windows!
