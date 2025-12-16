# VS Code Support Added! ✅

Your particle simulation now supports Visual Studio Code on Windows (and all platforms)!

## What Changed

### ✅ New: CMake Build System
- **CMakeLists.txt** - Cross-platform build configuration
- Works on Windows, Linux, Jetson, macOS
- Automatically detects CUDA, OpenMP, MPI
- Simpler than Visual Studio .sln files

### ✅ New: VS Code Configuration
**Complete .vscode folder** with:
- **tasks.json** - Build tasks (Ctrl+Shift+B)
- **launch.json** - Debug/Run configurations (F5)
- **c_cpp_properties.json** - IntelliSense settings
- **settings.json** - Workspace settings

### ✅ New: Documentation
- **VSCODE_QUICKSTART.md** - Fast 5-step setup (start here!)
- **VSCODE_SETUP_WINDOWS.md** - Complete detailed guide

### ✅ Advantages Over Visual Studio 2022

| Feature | VS 2022 | VS Code |
|---------|---------|---------|
| **Size** | ~7 GB | ~300 MB |
| **Speed** | Slower | Much faster |
| **Flexibility** | Windows only | Cross-platform |
| **Configuration** | GUI-based | Text-based (easier to share) |
| **Build System** | .sln/.vcxproj | CMake (industry standard) |
| **Learning Curve** | Steeper | Gentler |

---

## Quick Start (60 minutes)

### 1. Install Tools (~40 min)
- Visual Studio Code + extensions
- Visual Studio Build Tools 2022
- CMake
- CUDA Toolkit
- SDL2 libraries

See [**VSCODE_QUICKSTART.md**](VSCODE_QUICKSTART.md) for step-by-step.

### 2. Open Project (2 min)
```
1. Extract files to C:\Projects\ParticleSimulation
2. Open VS Code
3. File → Open Folder → Select project folder
4. Ctrl+Shift+P → "CMake: Configure"
```

### 3. Build (5 min)
```
Press Ctrl+Shift+B
Select "CMake: Build Release"
```

### 4. Copy DLLs (1 min)
```cmd
copy C:\SDL2\lib\x64\SDL2.dll build\Release\
copy C:\SDL2_ttf\lib\x64\SDL2_ttf.dll build\Release\
```

### 5. Run! (1 min)
```
Press F5
```

---

## File Overview

### New Files
```
.vscode/
├── tasks.json              ← Build commands
├── launch.json             ← Debug/Run configs
├── c_cpp_properties.json   ← IntelliSense
└── settings.json           ← Workspace settings

CMakeLists.txt              ← Build configuration

VSCODE_QUICKSTART.md        ← Quick setup guide
VSCODE_SETUP_WINDOWS.md     ← Detailed guide
```

### Unchanged Files
```
All C++ source files (.cpp, .hpp, .cu)
All documentation (.md files)
Makefile (still works for Linux/Jetson)
build.sh (still works for Linux/Jetson)
```

---

## Build Methods Comparison

### VS Code + CMake (New, Recommended for Windows)
✅ Lightweight (~300 MB vs 7 GB)  
✅ Cross-platform (works on Linux/Jetson too)  
✅ Industry standard (CMake)  
✅ Faster startup  
✅ Easier to configure  
✅ Better terminal integration  

**Build**: Ctrl+Shift+B  
**Run**: F5  

### Visual Studio 2022 (Old Method)
❌ Very large install (~7 GB)  
❌ Windows only  
❌ Requires manual .sln setup  
❌ Slower  
❌ More complex  
✅ Better visual debugging tools  

### Makefile (Linux/Jetson)
✅ Simple  
✅ Fast  
❌ Doesn't work on Windows  

**Build**: `make`

---

## What You Need to Do

### If You Want VS Code on Windows:

1. **Read the Quick Start**: [VSCODE_QUICKSTART.md](VSCODE_QUICKSTART.md)
2. **Install prerequisites** (60 min total):
   - VS Code
   - Build Tools
   - CMake
   - CUDA
   - SDL2
3. **Open project in VS Code**
4. **Press Ctrl+Shift+B** to build
5. **Copy DLLs** to build folder
6. **Press F5** to run

### If You're on Linux/Jetson:

**Nothing changes!** You can still use:
```bash
./build.sh
# or
make
```

But you *can* also use VS Code + CMake if you want:
```bash
# Install VS Code
# Open folder in VS Code
# Ctrl+Shift+P → CMake: Configure
# Ctrl+Shift+B to build
```

---

## Keyboard Shortcuts (VS Code)

| Key | Action |
|-----|--------|
| **Ctrl+Shift+B** | Build project |
| **F5** | Run/Debug |
| **Ctrl+Shift+P** | Command palette |
| **Ctrl+`** | Toggle terminal |
| **Ctrl+Shift+E** | File explorer |

---

## Troubleshooting

### "CMake not found"
```cmd
# Install from: https://cmake.org/download/
# Make sure "Add to PATH" was checked
cmake --version
```

### "Could not find SDL2"
```
Verify:
C:\SDL2\include\SDL.h exists
C:\SDL2\lib\x64\SDL2.lib exists
```

### "CUDA not found"
```cmd
nvcc --version
# If fails, install CUDA Toolkit
# Add to PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
```

### Build succeeds but "DLL not found"
```cmd
# Copy to executable directory:
copy C:\SDL2\lib\x64\SDL2.dll build\Release\
copy C:\SDL2_ttf\lib\x64\SDL2_ttf.dll build\Release\
```

---

## Why CMake?

**CMake is the industry standard build system:**
- Used by Google, Microsoft, NVIDIA, etc.
- Works everywhere (Windows, Linux, Mac, embedded)
- Can generate Visual Studio projects, Makefiles, Ninja files, etc.
- One CMakeLists.txt file works on all platforms
- Better than maintaining separate build files for each platform

**Our CMakeLists.txt:**
- Automatically detects your compiler
- Finds CUDA, OpenMP, MPI
- Configures SDL2 paths
- Sets optimization flags
- Generates appropriate build files for your system

---

## Performance

**No difference!** CMake just generates build files. The actual compiler and flags are the same:
- VS Code + CMake = Same performance as Visual Studio
- Same executable produced
- Same optimizations applied
- Same CUDA compilation

---

## Development Workflow

### With VS Code:
```
1. Edit code in VS Code
2. Ctrl+Shift+B (build)
3. F5 (run)
4. Debug with breakpoints
5. Git integration built-in
```

### Terminal-based:
```bash
# Still works!
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build/Release
./particle_sim.exe
```

---

## Summary

### What You Get:
✅ Simpler Windows setup (VS Code vs full Visual Studio)  
✅ Faster, lighter IDE  
✅ Industry-standard build system (CMake)  
✅ Cross-platform compatibility  
✅ All original features preserved  
✅ Same performance  

### What Changed:
- Added CMakeLists.txt
- Added .vscode configuration
- Added VS Code documentation
- **All C++ code unchanged**
- **All features still work**

### What Didn't Change:
- Source code (.cpp, .hpp, .cu files)
- Functionality (all 5 modes still work)
- Performance (identical)
- Linux/Jetson support (Makefile still works)

---

## Ready to Start?

👉 **[VSCODE_QUICKSTART.md](VSCODE_QUICKSTART.md)** ← Start here!

Or for detailed instructions:
👉 **[VSCODE_SETUP_WINDOWS.md](VSCODE_SETUP_WINDOWS.md)**

---

**Total time**: ~60 minutes from zero to running simulation  
**Result**: 30,000 particles @ 60 FPS on your RTX 3050! 🚀
