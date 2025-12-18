/*
 * OPTIMIZED Particle Simulation Header
 * High-Performance Parallel Computing - Comprehensive Optimization Framework
 * 
 * This header defines optimized data structures and interfaces for maximum
 * performance across all parallelization paradigms (Sequential, OpenMP, MPI, CUDA).
 * 
 * KEY OPTIMIZATIONS IN THIS FILE:
 * ================================
 * 
 * DATA STRUCTURE OPTIMIZATIONS:
 * - Cache-line aligned Particle structure (32-byte alignment)
 * - Hot data grouped together for cache efficiency
 * - POD types for CUDA compatibility
 * - Optimized memory layout for SIMD vectorization
 * 
 * SPATIAL GRID OPTIMIZATIONS:
 * - Parallel construction support
 * - Cache-friendly blocked iteration
 * - Exposed internals for advanced parallel algorithms
 * - Incremental update capability (future optimization)
 * 
 * INTERFACE DESIGN:
 * - Zero-cost abstractions
 * - Minimal virtual function overhead
 * - Static polymorphism where possible
 * - RAII for automatic resource management
 */

#ifndef PARTICLE_SIMULATION_HPP
#define PARTICLE_SIMULATION_HPP

#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

class SpatialGrid;
class SystemMonitor;

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

constexpr int WINDOW_WIDTH = 1920;
constexpr int WINDOW_HEIGHT = 1080;
constexpr int GRID_CELL_SIZE = 30;
constexpr int GRID_WIDTH = WINDOW_WIDTH / GRID_CELL_SIZE;
constexpr int GRID_HEIGHT = WINDOW_HEIGHT / GRID_CELL_SIZE;
constexpr int MAX_PARTICLES = 20000;
constexpr int DEFAULT_PARTICLE_COUNT = 800;

// ============================================================================
// OPTIMIZED PARTICLE STRUCTURE
// ============================================================================

/**
 * Particle structure optimized for cache performance and CUDA compatibility
 * 
 * OPTIMIZATION RATIONALE:
 * - 32-byte alignment matches ARM cache line size on Jetson Xavier NX
 * - Hot data (position, velocity) grouped at beginning
 * - Physics properties (radius, mass) follow for collision detection
 * - Cold data (color, active flag) at end
 * - Total size: 28 bytes + 4 padding = 32 bytes (one cache line)
 * 
 * MEMORY LAYOUT:
 * [x, y, vx, vy] - 16 bytes - Accessed every frame for physics
 * [radius, mass] - 8 bytes - Accessed during collision detection
 * [r, g, b, active] - 4 bytes - Accessed only during rendering
 * [padding] - 4 bytes - Align to 32-byte boundary
 * 
 * CACHE EFFICIENCY:
 * - Physics updates: 100% cache line utilization (all hot data in one line)
 * - Collision detection: One cache line load per particle
 * - Rendering: Cache line already loaded from physics pass
 */
struct alignas(32) Particle {
    // HOT DATA: Accessed every frame (16 bytes)
    float x, y;              // Position (most frequently accessed)
    float vx, vy;            // Velocity
    
    // WARM DATA: Accessed during collision detection (8 bytes)
    float radius;            // Collision radius
    float mass;              // For momentum calculations
    
    // COLD DATA: Accessed only for rendering (4 bytes)
    uint8_t r, g, b;         // Color components
    bool active;             // Enable/disable flag
    
    // PADDING: Align to 32 bytes (4 bytes)
    // Compiler automatically adds padding due to alignas(32)
    
    // Default constructor for array initialization
    Particle() : x(0.0f), y(0.0f), vx(0.0f), vy(0.0f),
                 radius(5.0f), mass(1.0f),
                 r(255), g(255), b(255), active(true) {}
} __attribute__((packed));

// Compile-time verification
static_assert(sizeof(Particle) == 32, "Particle must be exactly 32 bytes");
static_assert(alignof(Particle) == 32, "Particle must be 32-byte aligned");

// ============================================================================
// PARALLELIZATION MODE ENUMERATION
// ============================================================================

enum class ParallelMode {
    SEQUENTIAL = 0,      // Single-threaded baseline
    MULTITHREADED = 1,   // OpenMP shared-memory parallelization
    MPI = 2,             // Distributed-memory parallelization
    GPU_SIMPLE = 3,      // Basic GPU acceleration
    GPU_COMPLEX = 4      // Optimized GPU with spatial grid
};

// ============================================================================
// OPTIMIZED SPATIAL GRID
// ============================================================================

/**
 * High-performance spatial grid for O(n) collision detection
 * 
 * OPTIMIZATION FEATURES:
 * - Parallel construction support (OpenMP)
 * - Cache-friendly blocked iteration
 * - Exposed internals for advanced algorithms
 * - Vectorized operations where possible
 * 
 * ALGORITHM:
 * 1. Count particles per cell (parallelizable)
 * 2. Prefix sum to compute cell starts (parallel scan)
 * 3. Fill particle indices (parallelizable)
 * 
 * MEMORY LAYOUT:
 * - particle_indices: [p0, p1, p2, ...] sorted by grid cell
 * - cell_starts: [0, 3, 7, ...] start index for each cell
 * - cell_counts: [3, 4, 2, ...] number of particles in each cell
 */
class SpatialGrid {
private:
    int grid_width;
    int grid_height;
    float cell_size;
    int num_cells;
    
    std::vector<int> particle_indices;  // Particle IDs sorted by cell
    std::vector<int> cell_starts;       // Start index for each cell
    std::vector<int> cell_counts;       // Particle count per cell
    
public:
    /**
     * Constructor: Initialize grid structure
     */
    SpatialGrid(int width, int height, float size)
        : grid_width(width), grid_height(height),
          cell_size(size), num_cells(width * height) {
        
        cell_starts.resize(num_cells, 0);
        cell_counts.resize(num_cells, 0);
        particle_indices.resize(MAX_PARTICLES, -1);
    }
    
    /**
     * Get grid cell index for world coordinates
     * OPTIMIZATION: Inline for hot path
     */
    inline int get_cell_index(float x, float y) const {
        int grid_x = static_cast<int>(x / cell_size);
        int grid_y = static_cast<int>(y / cell_size);
        
        if (grid_x < 0 || grid_x >= grid_width ||
            grid_y < 0 || grid_y >= grid_height) {
            return -1;
        }
        
        return grid_y * grid_width + grid_x;
    }
    
    /**
     * Sequential grid update (baseline implementation)
     */
    void update(std::vector<Particle>& particles);
    
    /**
     * Get nearby particles within radius
     * OPTIMIZATION: Inline for collision detection hot path
     */
    inline void get_nearby_particles(float x, float y, float radius,
                                     std::vector<int>& nearby) const {
        nearby.clear();
        
        // Calculate grid cell range
        int min_grid_x = static_cast<int>((x - radius) / cell_size);
        int max_grid_x = static_cast<int>((x + radius) / cell_size);
        int min_grid_y = static_cast<int>((y - radius) / cell_size);
        int max_grid_y = static_cast<int>((y + radius) / cell_size);
        
        // Clamp to grid bounds
        if (min_grid_x < 0) min_grid_x = 0;
        if (max_grid_x >= grid_width) max_grid_x = grid_width - 1;
        if (min_grid_y < 0) min_grid_y = 0;
        if (max_grid_y >= grid_height) max_grid_y = grid_height - 1;
        
        // Collect particles from nearby cells
        for (int gy = min_grid_y; gy <= max_grid_y; gy++) {
            for (int gx = min_grid_x; gx <= max_grid_x; gx++) {
                int cell_idx = gy * grid_width + gx;
                int start = cell_starts[cell_idx];
                int count = cell_counts[cell_idx];
                
                for (int i = 0; i < count; i++) {
                    int particle_idx = particle_indices[start + i];
                    if (particle_idx >= 0) {
                        nearby.push_back(particle_idx);
                    }
                }
            }
        }
    }
    
    /**
     * OPTIMIZATION: Expose internals for parallel algorithms
     * These allow advanced parallel implementations to directly access
     * and modify grid data structures without going through methods
     */
    inline int get_num_cells() const { return num_cells; }
    inline std::vector<int>& get_cell_counts() { return cell_counts; }
    inline std::vector<int>& get_cell_starts() { return cell_starts; }
    inline std::vector<int>& get_particle_indices() { return particle_indices; }
    inline const std::vector<int>& get_cell_counts() const { return cell_counts; }
    inline const std::vector<int>& get_cell_starts() const { return cell_starts; }
    inline const std::vector<int>& get_particle_indices() const { return particle_indices; }
};

// ============================================================================
// PERFORMANCE METRICS STRUCTURE
// ============================================================================

struct PerformanceMetrics {
    double fps;
    double physics_time_ms;
    double render_time_ms;
    double frame_time_ms;
    float temperature_c;
    float power_watts;
    
    PerformanceMetrics() : fps(0.0), physics_time_ms(0.0), render_time_ms(0.0),
                          frame_time_ms(0.0), temperature_c(0.0f), power_watts(0.0f) {}
};

// ============================================================================
// MAIN SIMULATION CLASS
// ============================================================================

/**
 * Central simulation state and coordination
 * 
 * DESIGN PRINCIPLES:
 * - RAII: Automatic resource management
 * - Value semantics for particles (std::vector)
 * - Unique ownership for grid (std::unique_ptr)
 * - No manual memory management
 * - Exception-safe by construction
 */
class Simulation {
private:
    // Particle state
    std::vector<Particle> particles;
    std::unique_ptr<SpatialGrid> grid;
    
    // Simulation parameters
    int max_particles;
    int window_width;
    int window_height;
    float friction;
    float restitution;
    float mouse_force;
    
    // Mouse state
    int mouse_x, mouse_y;
    bool mouse_pressed;
    bool mouse_attract;
    
    // Application state
    bool running;
    bool reset_requested;
    ParallelMode mode;
    bool verbose_logging;
    
    // Performance metrics
    PerformanceMetrics metrics;
    int frame_counter;
    
public:
    /**
     * Constructor: Initialize simulation with default parameters
     */
    Simulation(int particle_count, int max_count = MAX_PARTICLES);
    
    /**
     * Destructor: Cleanup (CUDA memory freed here)
     */
    ~Simulation();
    
    /**
     * Main update function: Delegates to appropriate physics engine
     */
    void update(float dt);
    
    /**
     * Reset simulation state
     */
    void reset();
    
    /**
     * Particle management
     */
    void spawn_particle(float x, float y, float vx, float vy);
    void spawn_random_particles(int count);
    void add_particles(int count);
    void remove_particles(int count);
    
    // ========================================================================
    // GETTERS AND SETTERS
    // ========================================================================
    
    // Particle access
    inline std::vector<Particle>& get_particles() { return particles; }
    inline const std::vector<Particle>& get_particles() const { return particles; }
    inline Particle* get_particle_data() { return particles.data(); }
    inline int get_particle_count() const { return static_cast<int>(particles.size()); }
    
    // Grid access
    inline SpatialGrid& get_grid() { return *grid; }
    inline const SpatialGrid& get_grid() const { return *grid; }
    
    // Window dimensions
    inline int get_window_width() const { return window_width; }
    inline int get_window_height() const { return window_height; }
    
    // Physics parameters
    inline float get_friction() const { return friction; }
    inline void set_friction(float f) { friction = f; }
    inline void adjust_friction(float delta) { 
        friction += delta;
        if (friction < 0.0f) friction = 0.0f;
        if (friction > 1.0f) friction = 1.0f;
    }
    inline float get_restitution() const { return restitution; }
    inline void set_restitution(float r) { restitution = r; }
    inline float get_mouse_force() const { return mouse_force; }
    inline void set_mouse_force(float f) { mouse_force = f; }
    
    // Mouse state
    inline int get_mouse_x() const { return mouse_x; }
    inline int get_mouse_y() const { return mouse_y; }
    inline bool is_mouse_pressed() const { return mouse_pressed; }
    inline bool is_mouse_attract() const { return mouse_attract; }
    inline void set_mouse_state(int x, int y, bool pressed, bool attract) {
        mouse_x = x; mouse_y = y;
        mouse_pressed = pressed; mouse_attract = attract;
    }
    
    // Application control
    inline bool is_running() const { return running; }
    inline void set_running(bool r) { running = r; }
    inline void request_reset() { reset_requested = true; }
    
    // Parallelization mode
    inline ParallelMode get_mode() const { return mode; }
    inline void set_mode(ParallelMode m) { mode = m; }
    
    // Performance metrics
    inline const PerformanceMetrics& get_metrics() const { return metrics; }
    inline void set_fps(double fps) { metrics.fps = fps; }
    inline void set_physics_time(double ms) { metrics.physics_time_ms = ms; }
    inline void set_render_time(double ms) { metrics.render_time_ms = ms; }
    inline void set_frame_time(double ms) { metrics.frame_time_ms = ms; }
    inline void set_temperature(float temp) { metrics.temperature_c = temp; }
    inline void set_power(float watts) { metrics.power_watts = watts; }
    
    // Logging
    inline bool is_verbose() const { return verbose_logging; }
    inline bool is_verbose_logging() const { return verbose_logging; }
    inline void set_verbose(bool v) { verbose_logging = v; }
    inline void set_verbose_logging(bool v) { verbose_logging = v; }
    
    // Frame counter
    inline int get_frame_counter() const { return frame_counter; }
};

// ============================================================================
// PHYSICS ENGINE INTERFACE
// ============================================================================

/**
 * Static physics engine with multiple implementation modes
 * 
 * DESIGN: Static methods allow compiler to optimize aggressively
 * No virtual function overhead, direct function calls
 */
class PhysicsEngine {
public:
    /**
     * MODE 1: Sequential (Optimized Baseline)
     * OPTIMIZATIONS: Cache-friendly iteration, early exits, compiler hints
     * TARGET: 1.5× improvement over naive implementation
     */
    static void update_sequential(Simulation& sim, float dt);
    
    /**
     * MODE 2: Multithreaded (OpenMP)
     * OPTIMIZATIONS: Parallel grid, persistent regions, optimized scheduling
     * TARGET: 4.5-5.0× speedup on 6-core system
     */
    static void update_multithreaded(Simulation& sim, float dt);
    
    /**
     * MODE 3: Distributed (MPI)
     * OPTIMIZATIONS: Allgather, minimal data, combined broadcasts
     * TARGET: 3.5-4.0× speedup on 4 processes
     */
    static void update_mpi(Simulation& sim, float dt);
    
    /**
     * MODE 4: GPU Simple (CUDA Basic)
     * Basic GPU implementation for comparison
     */
    static void update_gpu_simple(Simulation& sim, float dt);
    
    /**
     * MODE 5: GPU Complex (CUDA Optimized)
     * Full GPU optimization with spatial grid
     */
    static void update_gpu_complex(Simulation& sim, float dt);
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

class Utils {
public:
    /**
     * High-resolution timing
     */
    static double get_time_ms();
    
    /**
     * Random number generation
     */
    static float random_float(float min, float max);
    static int random_int(int min, int max);
    
    /**
     * Performance summary printing
     */
    static void print_performance_summary(const PerformanceMetrics& metrics, ParallelMode mode);
};

// ============================================================================
// SYSTEM MONITORING
// ============================================================================

class SystemMonitor {
public:
    /**
     * Update system metrics (temperature, power)
     */
    static void update_metrics(Simulation& sim);
    
    /**
     * Platform-specific temperature reading
     */
    static float read_temperature();
    
    /**
     * Platform-specific power reading
     */
    static float read_power();
};

// ============================================================================
// CUDA INTERFACE (if available)
// ============================================================================

#ifdef USE_CUDA
extern "C" {
    void update_physics_gpu_simple_cuda(Simulation* sim, float dt);
    void update_physics_gpu_complex_cuda(Simulation* sim, float dt);
    void cleanup_gpu_memory();
}
#endif

#endif // PARTICLE_SIMULATION_HPP
