# Technical Documentation

Comprehensive guide to the architecture, implementation, and design decisions of the Parallel Particle Simulation for NVIDIA Jetson and Linux platforms.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [File Structure](#file-structure)
- [Class Hierarchy](#class-hierarchy)
- [Physics Engine](#physics-engine)
- [Parallelization Modes](#parallelization-modes)
- [Memory Management](#memory-management)
- [Performance Optimizations](#performance-optimizations)
- [Build System](#build-system)
- [Extending the Project](#extending-the-project)

---

## Architecture Overview

### Design Philosophy

The project follows these principles:

1. **Modern C++ (C++17)**: RAII, smart pointers, STL containers
2. **Zero-Cost Abstractions**: Performance equivalent to C
3. **Embedded-Optimized**: Designed for NVIDIA Jetson platform
4. **Extensible**: Easy to add new parallelization modes
5. **Educational**: Clear code demonstrating parallel concepts

### High-Level Flow

```
Main Loop:
1. Handle Input (SDL events) → InputHandler
2. Update Physics (selected mode) → PhysicsEngine
3. Update Spatial Grid → SpatialGrid
4. Render Frame → Graphics
5. Update System Metrics → SystemMonitor
6. Calculate FPS
```

### Data Flow

```
Simulation (owns everything)
    ├── std::vector<Particle> (CPU memory)
    ├── SpatialGrid (spatial partitioning)
    └── PerformanceMetrics (timing data)

PhysicsEngine (static methods)
    ├── update_sequential()
    ├── update_multithreaded() [OpenMP]
    ├── update_mpi() [MPI]
    ├── update_gpu_simple() [CUDA]
    └── update_gpu_complex() [CUDA]

Graphics (SDL rendering)
    ├── render_particles()
    ├── render_stats()
    └── render_menu()
```

---

## File Structure

### Core Files

#### ParticleSimulation.hpp (309 lines)
**Purpose**: Central header defining all interfaces and data structures.

**Key Components**:
```cpp
// Platform detection
#ifdef PLATFORM_JETSON
    // Jetson-specific optimizations
#else
    // Generic Linux
#endif

// Particle structure (POD for CUDA compatibility)
struct Particle {
    float x, y;              // Position
    float vx, vy;            // Velocity
    float radius, mass;      // Physical properties
    uint8_t r, g, b;         // Color
    bool active;             // Enabled flag
};

// Spatial grid for O(n) collision detection
class SpatialGrid {
    std::vector<int> particle_indices;
    std::vector<int> cell_starts;
    std::vector<int> cell_counts;
    // Methods for updating grid and querying neighbors
};

// Main simulation state
class Simulation {
    std::vector<Particle> particles;
    std::unique_ptr<SpatialGrid> grid;
    // Physics parameters, mouse state, performance metrics
};

// Physics engine interface
class PhysicsEngine {
    static void update_sequential(...);
    static void update_multithreaded(...);
    static void update_mpi(...);
    static void update_gpu_simple(...);
    static void update_gpu_complex(...);
};
```

**Design Decisions**:
- `Particle` is POD to allow direct CUDA memcpy
- `SpatialGrid` uses std::vector for automatic memory management
- `Simulation` owns all state, passed by reference to engines
- `PhysicsEngine` is stateless (static methods only)

#### Simulation.cpp (220 lines)
**Purpose**: Implements Simulation class and SpatialGrid.

**Key Algorithms**:

1. **Spatial Grid Update** (O(n)):
```cpp
void SpatialGrid::update(particles) {
    // 1. Count particles per cell
    for each particle:
        cell = get_cell(particle.position)
        cell_counts[cell]++
    
    // 2. Compute prefix sum for cell_starts
    cell_starts[0] = 0
    for i in 1..num_cells:
        cell_starts[i] = cell_starts[i-1] + cell_counts[i-1]
    
    // 3. Fill particle indices
    for each particle:
        cell = get_cell(particle.position)
        insert at cell_starts[cell] + current_count
}
```

2. **Nearby Particle Query** (O(1) amortized):
```cpp
void SpatialGrid::get_nearby_particles(x, y, radius, nearby) {
    // Calculate grid cell range
    min_x = (x - radius) / CELL_SIZE
    max_x = (x + radius) / CELL_SIZE
    
    // Clamp to grid bounds
    // Collect all particles in nearby cells
    for each cell in range:
        for each particle in cell:
            nearby.push_back(particle_index)
}
```

**Performance**: Grid update is O(n), collision detection becomes O(n) instead of O(n²).

#### Physics.cpp (430 lines)
**Purpose**: Implements all 5 physics update modes.

**Collision Physics**:

1. **Particle-Particle Collision**:
```cpp
// Separate overlapping particles
overlap = (r1 + r2) - distance
separation = overlap * (mass_ratio)

// Calculate impulse (perfectly elastic)
relative_velocity = v2 - v1
impulse = -(1 + restitution) * dot(relative_velocity, normal)
         / (1/m1 + 1/m2)

// Apply impulse
v1 -= impulse * normal / m1
v2 += impulse * normal / m2
```

2. **Wall Collision**:
```cpp
if (x - radius < 0):
    x = radius
    vx = -vx
    apply_friction()
```

**Modes**:
- Sequential: Simple loop, baseline performance
- Multithreaded: `#pragma omp parallel for` with dynamic scheduling
- MPI: Domain decomposition with gather/broadcast synchronization
- GPU Simple/Complex: Calls CUDA functions in PhysicsGPU.cu

#### PhysicsGPU.cu (550 lines)
**Purpose**: CUDA GPU kernels for modes 4 and 5.

**GPU Memory Management**:
```cpp
// Static device pointers
static Particle* d_particles = nullptr;
static int* d_grid_indices = nullptr;
static int* d_grid_starts = nullptr;
static int* d_grid_counts = nullptr;

// RAII cleanup in Simulation destructor
Simulation::~Simulation() {
    cleanup_gpu_memory();  // Frees all device memory
}
```

**Kernel Launch Pattern**:
```cpp
// 1. Copy parameters to device constant memory
cudaMemcpyToSymbol(d_friction, &friction, sizeof(float));

// 2. Copy particles to device
cudaMemcpy(d_particles, particles, size, cudaMemcpyHostToDevice);

// 3. Launch kernels
int blocks = (count + 256 - 1) / 256;
update_particles_kernel<<<blocks, 256>>>(d_particles, count, dt);
detect_collisions_kernel<<<blocks, 256>>>(d_particles, ...);

// 4. Copy results back
cudaMemcpy(particles, d_particles, size, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
```

**Optimization (Mode 5)**:
- Shared memory caching for particle data
- Spatial grid for O(n) collision detection on GPU
- Coalesced memory access patterns
- Atomic operations for grid construction

#### Rendering.cpp (260 lines)
**Purpose**: SDL2-based graphics and UI.

**Graphics Class**:
```cpp
class Graphics {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;
    
public:
    Graphics() { /* Initialize SDL */ }
    ~Graphics() { /* Cleanup SDL - RAII */ }
    
    void render_particles(Simulation& sim);
    void render_stats(Simulation& sim);
    void render_menu(Simulation& sim);
    void present();
};
```

**Rendering Pipeline**:
```cpp
1. Clear screen (black)
2. For each particle:
     Set color (r, g, b)
     Draw filled circle at (x, y)
3. Render stats overlay (FPS, particle count, mode, etc.)
4. Render menu (if enabled)
5. SDL_RenderPresent()
```

**Text Rendering**:
- Uses TTF_RenderText_Solid for fast rendering
- Creates texture from surface, renders, destroys texture
- Gracefully handles missing font file

#### SystemMonitor.cpp (120 lines)
**Purpose**: Platform-specific temperature and power monitoring.

**Jetson Platform Implementation**:
```cpp
#ifdef PLATFORM_JETSON
    // Read from /sys/devices/virtual/thermal/...
    float read_temperature() {
        // Try multiple thermal zones
        // Return highest temperature
    }
    
    // Read from /sys/bus/i2c/.../in_power*_input
    float read_power() {
        // Sum across multiple power rails
    }
#elif defined(PLATFORM_LINUX)
    // Generic Linux implementation
    float read_temperature() {
        // Read from thermal_zone0
    }
    float read_power() {
        return 0.0f;  // Not available on generic Linux
    }
#endif
```

**Update Strategy**:
- Called every 30 frames to reduce overhead
- Reads from sysfs on Linux/Jetson
- Returns actual hardware metrics on Jetson

#### Main.cpp (270 lines)
**Purpose**: Application entry point and main loop.

**Application Class**:
```cpp
class Application {
private:
    Simulation sim;
    bool quit;
    time_point last_time, fps_timer;
    
public:
    int run() {
        while (!quit) {
            // 1. Handle events
            // 2. Calculate dt
            // 3. Update physics: sim.update(dt)
            // 4. Render frame
            // 5. Calculate FPS
            // 6. Update metrics
        }
    }
};
```

**Input Handling**:
```cpp
class InputHandler {
    static void handle_keyboard(event, sim, quit) {
        switch (key) {
            case SDLK_1: sim.set_mode(SEQUENTIAL); break;
            case SDLK_2: sim.set_mode(MULTITHREADED); break;
            // etc.
        }
    }
    
    static void handle_mouse(event, sim) {
        // Update mouse state in simulation
    }
};
```

**MPI Integration**:
```cpp
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    
    if (rank == 0) {
        // Only rank 0 does graphics
        init_graphics();
        run_simulation();
        cleanup_graphics();
    } else {
        // Other ranks do physics only
        while (running) {
            PhysicsEngine::update_mpi(sim, dt);
        }
    }
    
    MPI_Finalize();
#endif
```

---

## Class Hierarchy

### Ownership Model

```
Application (stack)
  └─> Simulation (stack)
        ├─> std::vector<Particle> (heap)
        └─> std::unique_ptr<SpatialGrid> (heap)
              └─> std::vector<int> × 3 (heap)

Graphics (static unique_ptr)
  ├─> SDL_Window* (SDL-managed)
  ├─> SDL_Renderer* (SDL-managed)
  └─> TTF_Font* (SDL-managed)

GPU Memory (static, manual management)
  ├─> d_particles (CUDA-managed)
  ├─> d_grid_indices (CUDA-managed)
  ├─> d_grid_starts (CUDA-managed)
  └─> d_grid_counts (CUDA-managed)
```

### Lifetime Management

- **RAII Everywhere**: Destructors handle cleanup automatically
- **No Manual Memory Management**: std::vector, unique_ptr for heap allocations
- **Exception Safety**: Even if exception thrown, destructors run
- **GPU Memory**: Cleaned up in Simulation destructor via cleanup_gpu_memory()

---

## Physics Engine

### Time Step Integration

**Semi-Implicit Euler**:
```cpp
// Update velocity
v += acceleration * dt
// Update position using new velocity
x += v * dt
```

**Stability**:
- `dt` capped at 0.05s (20 FPS minimum)
- Velocity magnitude limited to MAX_VELOCITY (1000.0)

### Collision Detection

**Broad Phase**: Spatial Grid
```cpp
Grid cells: 64×36 (based on window size / cell size)
Each particle assigned to cell: (x / CELL_SIZE, y / CELL_SIZE)
Query: Check only particles in nearby 9 cells
Complexity: O(n) amortized instead of O(n²)
```

**Narrow Phase**: Circle-Circle
```cpp
if distance(p1, p2) < (r1 + r2):
    // Collision detected
    resolve_collision(p1, p2)
```

### Collision Response

**Conservation Laws**:
- Momentum conserved: `m1*v1 + m2*v2 = m1*v1' + m2*v2'`
- Energy conserved (perfectly elastic): `restitution = 1.0`
- Can adjust restitution for inelastic collisions

**Implementation**:
1. Separate overlapping particles (position correction)
2. Calculate relative velocity in collision normal direction
3. Compute impulse magnitude
4. Apply impulse to both particles

---

## Parallelization Modes

### Mode 1: Sequential (Baseline)

**Implementation**: Single-threaded, simple loop.

**Characteristics**:
- Predictable, deterministic
- Easiest to debug
- Baseline for speedup calculations

**Code**:
```cpp
for (int i = 0; i < count; i++) {
    update_particle(particles[i], dt);
}
for (int i = 0; i < count; i++) {
    detect_collisions(particles[i], grid);
}
```

**Expected Performance (Jetson Xavier NX)**:
- ~800 particles @ 60 FPS

### Mode 2: Multithreaded (OpenMP)

**Implementation**: `#pragma omp parallel for` with dynamic scheduling.

**Characteristics**:
- Automatic work distribution across cores
- Dynamic scheduling handles load imbalance
- Shared memory model

**Code**:
```cpp
#pragma omp parallel for schedule(dynamic, 64)
for (int i = 0; i < count; i++) {
    update_particle(particles[i], dt);
}

#pragma omp parallel
{
    #pragma omp for schedule(dynamic, 32) nowait
    for (int i = 0; i < count; i++) {
        detect_collisions(particles[i], grid);
    }
}
```

**Synchronization**:
- Implicit barrier after parallel region
- Collision resolution has rare race conditions (acceptable for visualization)
- No explicit locks for performance

**Expected Performance (Jetson Xavier NX)**:
- ~2,000 particles @ 60 FPS
- 2.5x speedup over sequential

### Mode 3: MPI (Distributed)

**Implementation**: Domain decomposition with explicit communication.

**Characteristics**:
- Processes own particle subsets
- Explicit message passing
- Rank 0 handles graphics

**Communication Pattern**:
```cpp
// 1. Broadcast mouse state to all ranks
MPI_Bcast(&mouse_state, ...)

// 2. Each rank updates its particles
for (int i = start_idx; i < end_idx; i++) {
    update_particle(particles[i], dt);
}

// 3. Gather updated particles to rank 0
if (rank == 0) {
    for (int r = 1; r < size; r++) {
        MPI_Recv(&particles[r_start], ...)
    }
} else {
    MPI_Send(&particles[start_idx], ...)
}

// 4. Broadcast complete state to all ranks
MPI_Bcast(particles, ...)

// 5. Collision detection (all ranks have full data)
for (int i = start_idx; i < end_idx; i++) {
    detect_collisions(particles[i], grid);
}

// 6. Gather and broadcast collision results
```

**Overhead**: Communication is expensive, especially gather/broadcast of full particle array twice per frame.

**Expected Performance (Jetson Xavier NX, 4 processes)**:
- ~1,800 particles @ 60 FPS
- 2.25x speedup (lower than OpenMP due to overhead)

### Mode 4: GPU Simple (CUDA)

**Implementation**: Basic CUDA kernels with global memory.

**Characteristics**:
- Straightforward port to GPU
- Brute-force collision detection on GPU
- No advanced optimizations

**Kernel Structure**:
```cuda
__global__ void update_particles(Particle* particles, int count, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p = &particles[idx];
    // Update position, apply forces, wall collisions
}

__global__ void detect_collisions(Particle* particles, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p1 = &particles[idx];
    
    // Check against ALL other particles (O(n²) on GPU)
    for (int j = idx + 1; j < count; j++) {
        Particle* p2 = &particles[j];
        resolve_collision_gpu(p1, p2);
    }
}
```

**Memory Transfers**:
- Host → Device: Particles, constants (every frame)
- Device → Host: Updated particles (every frame)
- **Bottleneck**: PCIe transfer overhead (less severe on Jetson due to unified memory)

**Expected Performance (Jetson Xavier NX)**:
- ~6,000 particles @ 60 FPS
- 7.5x speedup over sequential

### Mode 5: GPU Complex (Optimized CUDA)

**Implementation**: Advanced CUDA with shared memory and spatial grid.

**Optimizations**:

1. **Shared Memory Caching**:
```cuda
__global__ void update_particles_optimized(...) {
    __shared__ Particle s_particles[256];
    
    // Load particle into shared memory (fast)
    s_particles[threadIdx.x] = particles[idx];
    __syncthreads();
    
    // Work on shared memory copy
    Particle* p = &s_particles[threadIdx.x];
    // ... update particle ...
    
    // Write back to global memory
    __syncthreads();
    particles[idx] = *p;
}
```

2. **GPU Spatial Grid**:
```cuda
__global__ void build_spatial_grid(Particle* particles, ...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    Particle* p = &particles[idx];
    int cell = get_cell(p->x, p->y);
    
    // Atomic increment for thread-safe grid construction
    int pos = atomicAdd(&grid_counts[cell], 1);
    grid_indices[cell * MAX_PARTICLES + pos] = idx;
}

__global__ void detect_collisions_complex(...) {
    // Query only nearby cells (O(n) on GPU)
    for (int cell in nearby_cells) {
        for (int idx in cell_particles) {
            resolve_collision(p1, particles[idx]);
        }
    }
}
```

3. **Coalesced Memory Access**:
- Sequential threads access sequential memory
- Maximizes memory bandwidth utilization

**Expected Performance (Jetson Xavier NX)**:
- ~10,000 particles @ 60 FPS
- 12.5x speedup over sequential

**Jetson Advantage**: Unified memory architecture reduces CPU↔GPU transfer overhead compared to discrete GPUs.

---

## Memory Management

### C++ RAII Pattern

**Before (C)**:
```c
Particle* particles = malloc(MAX_PARTICLES * sizeof(Particle));
// ... use particles ...
free(particles);  // Easy to forget!
```

**After (C++)**:
```cpp
std::vector<Particle> particles;
particles.reserve(MAX_PARTICLES);
// ... use particles ...
// Automatic cleanup when vector goes out of scope
```

### Smart Pointers

**unique_ptr for Single Ownership**:
```cpp
class Simulation {
    std::unique_ptr<SpatialGrid> grid;
    
    Simulation() 
        : grid(std::make_unique<SpatialGrid>(MAX_PARTICLES)) {
        // grid allocated
    }
    
    ~Simulation() {
        // grid automatically deleted
    }
};
```

### GPU Memory

**Manual Management Required** (CUDA doesn't support RAII directly):
```cpp
static Particle* d_particles = nullptr;

void init_gpu_memory(int max_particles) {
    cudaMalloc(&d_particles, max_particles * sizeof(Particle));
}

void cleanup_gpu_memory() {
    if (d_particles) {
        cudaFree(d_particles);
        d_particles = nullptr;
    }
}

// Called automatically in Simulation destructor
Simulation::~Simulation() {
#ifdef USE_CUDA
    cleanup_gpu_memory();
#endif
}
```

### Memory Layout

**Particle Structure** (40 bytes, aligned):
```
Offset  Size  Field
0       4     x
4       4     y
8       4     vx
12      4     vy
16      4     radius
20      4     mass
24      1     r
25      1     g
26      1     b
27      1     active
28-39   12    padding (for alignment)
```

**Spatial Grid**:
```
particle_indices: int[MAX_PARTICLES]     // 40KB for 10K particles
cell_starts:      int[GRID_WIDTH * GRID_HEIGHT]  // ~10KB
cell_counts:      int[GRID_WIDTH * GRID_HEIGHT]  // ~10KB
Total: ~60KB overhead
```

**Jetson Memory Considerations**:
- Unified memory architecture (CPU and GPU share RAM)
- No PCIe transfer overhead
- Memory pressure affects both CPU and GPU performance
- Important to monitor total system memory usage

---

## Performance Optimizations

### Algorithmic Optimizations

1. **Spatial Partitioning**: O(n²) → O(n) for collision detection
2. **Early Exit**: Skip inactive particles, particles moving apart
3. **Velocity Limiting**: Prevents tunneling, maintains stability

### CPU Optimizations

1. **Cache-Friendly Access**: Sequential memory access patterns
2. **Dynamic Scheduling**: OpenMP balances load automatically
3. **Minimal Synchronization**: Rare race conditions acceptable

### GPU Optimizations

1. **Shared Memory**: 100x faster than global memory
2. **Coalesced Access**: Full memory bandwidth utilization
3. **Occupancy**: 256 threads/block for good SM utilization
4. **Constant Memory**: Read-only parameters cached per SM

### Jetson-Specific Optimizations

1. **Unified Memory**: Leverage zero-copy access patterns where possible
2. **Power Modes**: Use `nvpmodel` to balance performance/power
3. **Clock Management**: Lock clocks with `jetson_clocks` for consistent performance
4. **Thermal Awareness**: Monitor temperature to prevent throttling

### Profiling Results

**Sequential (1000 particles on Jetson Xavier NX)**:
- Physics: 15ms
- Render: 2ms
- FPS: 60 (VSync limited)

**GPU Complex (10000 particles on Jetson Xavier NX)**:
- Physics: 5ms (includes memcpy)
- Render: 8ms (more particles to draw)
- FPS: 60 (VSync limited)

---

## Build System

### Makefile Targets

```makefile
make           # Standard: Sequential + OpenMP
make mpi       # Add MPI support
make cuda      # Add GPU support
make cuda_mpi  # All features
make clean     # Remove build artifacts
make help      # Show available targets
make info      # Show build configuration
```

### Conditional Compilation

```cpp
#ifdef USE_MPI
    // MPI-specific code
#endif

#ifdef USE_CUDA
    // CUDA-specific code
#endif

#ifdef _OPENMP
    #pragma omp parallel
#endif
```

### Platform Detection

```makefile
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    SDL_LIBS = $(shell sdl2-config --libs)
    # Check for Jetson
    ifneq (,$(wildcard /etc/nv_tegra_release))
        PLATFORM := jetson
    endif
endif
```

### CUDA Architecture Selection

```makefile
# Jetson Nano
CUDA_ARCH := -arch=sm_53

# Jetson TX2
CUDA_ARCH := -arch=sm_62

# Jetson Xavier NX/AGX
CUDA_ARCH := -arch=sm_72

# Jetson Orin
CUDA_ARCH := -arch=sm_87
```

---

## Extending the Project

### Adding a New Parallelization Mode

1. **Add enum value** in ParticleSimulation.hpp:
```cpp
enum class ParallelMode {
    // ...
    GPU_ADVANCED = 6
};
```

2. **Add update function** in Physics.cpp:
```cpp
void PhysicsEngine::update_gpu_advanced(Simulation& sim, float dt) {
    // Your implementation
}
```

3. **Add case** in Simulation::update():
```cpp
case ParallelMode::GPU_ADVANCED:
    PhysicsEngine::update_gpu_advanced(*this, dt);
    break;
```

4. **Add keyboard binding** in Main.cpp:
```cpp
case SDLK_6:
    sim.set_mode(ParallelMode::GPU_ADVANCED);
    break;
```

### Adding New Physics

1. **Modify Particle** structure to include new properties
2. **Update initialization** in spawn_particle()
3. **Add forces** in update functions:
```cpp
// Example: Add gravity
p.vy += GRAVITY * dt;
```

4. **Handle** in collision resolution if needed

### Adding New Visualizations

1. **Create new render function** in Rendering.cpp:
```cpp
void Graphics::render_heatmap(Simulation& sim) {
    // Render particle density as heatmap
}
```

2. **Call from render_frame()**
3. **Add toggle key** in input handler

---

## Performance Profiling

### Using CUDA Profiler

```bash
# Legacy profiler
nvprof ./particle_sim_cuda

# Nsight Systems (recommended)
nsys profile --stats=true ./particle_sim_cuda

# Nsight Compute (kernel-level)
ncu --set full ./particle_sim_cuda
```

### Using tegrastats (Jetson)

```bash
# Real-time system monitoring
tegrastats

# Log to file
tegrastats --interval 1000 --logfile stats.log
```

### Using Valgrind (Memory Leaks)

```bash
valgrind --leak-check=full --show-leak-kinds=all ./particle_sim
```

### Using GDB (Debugging)

```bash
gdb ./particle_sim
(gdb) run
(gdb) break Physics.cpp:42
(gdb) continue
```

### Performance Metrics to Track

1. **FPS**: Frames per second (target: 60)
2. **Physics Time**: Time spent in physics update (ms)
3. **Render Time**: Time spent rendering (ms)
4. **Particle Count**: Number of active particles
5. **Temperature**: System temperature (°C) - Jetson specific
6. **Power**: Power consumption (W) - Jetson specific
7. **GPU Utilization**: Percentage (use tegrastats)
8. **Memory Usage**: MB used (use tegrastats)

---

## Common Pitfalls

### Race Conditions in OpenMP

**Problem**: Multiple threads modify same particle pair.
**Solution**: Accept rare collisions being missed (visual application) or use fine-grained locking.

### GPU Memory Leaks

**Problem**: Forgetting to call cudaFree.
**Solution**: Use RAII-style wrappers or ensure cleanup in destructor.

### MPI Deadlocks

**Problem**: Ranks waiting for each other.
**Solution**: Ensure all ranks participate in collective operations, use non-blocking communication.

### Cache Thrashing

**Problem**: False sharing in OpenMP.
**Solution**: Use `schedule(dynamic)` with reasonable chunk sizes.

### Thermal Throttling (Jetson)

**Problem**: Performance drops when temperature exceeds threshold.
**Solution**: Add cooling, reduce particle count, or use lower power mode.

---

## Additional Resources

### C++ and Parallel Programming
- **C++ Reference**: https://en.cppreference.com/
- **OpenMP Tutorial**: https://www.openmp.org/resources/tutorials-articles/
- **MPI Tutorial**: https://mpitutorial.com/

### CUDA and GPU Programming
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **Nsight Profiler**: https://developer.nvidia.com/nsight-systems

### Jetson Development
- **Jetson Developer Zone**: https://developer.nvidia.com/embedded/jetson
- **JetPack Documentation**: https://docs.nvidia.com/jetson/
- **Jetson Linux**: https://developer.nvidia.com/embedded/linux-tegra
- **Jetson Projects**: https://developer.nvidia.com/embedded/community/jetson-projects

### SDL2 Graphics
- **SDL2 Documentation**: https://wiki.libsdl.org/
- **SDL2 Tutorials**: https://lazyfoo.net/tutorials/SDL/

---

## Jetson Performance Tuning

### Power Modes

```bash
# List available modes
sudo nvpmodel -q

# Set to maximum performance (mode 0)
sudo nvpmodel -m 0

# For battery/thermal constraints (mode 2)
sudo nvpmodel -m 2
```

### Clock Management

```bash
# Lock clocks to maximum
sudo jetson_clocks

# Show current clocks
sudo jetson_clocks --show

# Restore default clocks
sudo jetson_clocks --restore
```

### Monitoring

```bash
# Real-time stats
tegrastats

# Temperature
cat /sys/devices/virtual/thermal/thermal_zone0/temp

# Power consumption
cat /sys/bus/i2c/drivers/ina3221x/*/iio_device/in_power*_input
```

---

This documentation provides a complete technical reference for understanding, extending, and optimizing the Parallel Particle Simulation on NVIDIA Jetson and Linux platforms.