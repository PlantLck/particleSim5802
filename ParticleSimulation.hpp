#ifndef PARTICLE_SIMULATION_HPP
#define PARTICLE_SIMULATION_HPP

#include <vector>
#include <memory>
#include <cstdint>
#include <string>
#include <chrono>

// ============================================================================
// Platform Detection - Linux and Jetson Only
// ============================================================================

#if defined(__linux__)
    #define PLATFORM_LINUX
    #ifdef __arm__
        #define PLATFORM_JETSON
    #endif
#else
    #error "This project only supports Linux and Jetson platforms"
#endif

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr int WINDOW_WIDTH = 1280;
constexpr int WINDOW_HEIGHT = 720;

// Particle limits based on platform
#ifdef PLATFORM_JETSON
    constexpr int MAX_PARTICLES = 10000;  // Jetson embedded system
#else
    constexpr int MAX_PARTICLES = 15000;  // Linux desktop with GPU
#endif

constexpr int DEFAULT_PARTICLE_COUNT = 500;
constexpr float DEFAULT_PARTICLE_RADIUS = 3.0f;
constexpr int GRID_CELL_SIZE = 20;
constexpr int GRID_WIDTH = (WINDOW_WIDTH / GRID_CELL_SIZE + 1);
constexpr int GRID_HEIGHT = (WINDOW_HEIGHT / GRID_CELL_SIZE + 1);
constexpr float MAX_VELOCITY = 1000.0f;
constexpr int NEARBY_BUFFER_SIZE = 100;

// ============================================================================
// Parallelization Modes
// ============================================================================

enum class ParallelMode {
    SEQUENTIAL = 0,
    MULTITHREADED = 1,
    MPI = 2,
    GPU_SIMPLE = 3,
    GPU_COMPLEX = 4
};

// ============================================================================
// Particle Structure (POD for CUDA compatibility)
// ============================================================================

struct Particle {
    float x, y;              // Position
    float vx, vy;            // Velocity
    float radius;            // Particle radius
    float mass;              // Particle mass
    uint8_t r, g, b;         // Color
    bool active;             // Whether particle is in use
    
    Particle() : x(0), y(0), vx(0), vy(0), radius(DEFAULT_PARTICLE_RADIUS),
                 mass(0), r(0), g(0), b(0), active(false) {}
};

// ============================================================================
// Detailed Performance Metrics
// ============================================================================

struct DetailedMetrics {
    // Overall timing
    double total_physics_time_ms;
    double total_render_time_ms;
    double fps;
    
    // CPU breakdown
    double cpu_update_time_ms;
    double cpu_grid_build_time_ms;
    double cpu_collision_time_ms;
    
    // GPU breakdown (Mode 4 & 5)
    double gpu_h2d_transfer_ms;      // Host to Device
    double gpu_d2h_transfer_ms;      // Device to Host
    double gpu_update_kernel_ms;
    double gpu_collision_kernel_ms;
    double gpu_grid_count_kernel_ms;
    double gpu_grid_fill_kernel_ms;
    double gpu_prefix_sum_ms;
    double gpu_constant_copy_ms;
    double gpu_sync_overhead_ms;
    
    // Memory info
    size_t gpu_memory_used_bytes;
    size_t particle_data_size_bytes;
    
    // System metrics
    float temperature_c;
    float power_watts;
    
    // Frame statistics
    int particle_count;
    int collision_checks;
    int actual_collisions;
    
    DetailedMetrics() : 
        total_physics_time_ms(0), total_render_time_ms(0), fps(0),
        cpu_update_time_ms(0), cpu_grid_build_time_ms(0), cpu_collision_time_ms(0),
        gpu_h2d_transfer_ms(0), gpu_d2h_transfer_ms(0), 
        gpu_update_kernel_ms(0), gpu_collision_kernel_ms(0),
        gpu_grid_count_kernel_ms(0), gpu_grid_fill_kernel_ms(0),
        gpu_prefix_sum_ms(0), gpu_constant_copy_ms(0), gpu_sync_overhead_ms(0),
        gpu_memory_used_bytes(0), particle_data_size_bytes(0),
        temperature_c(0), power_watts(0),
        particle_count(0), collision_checks(0), actual_collisions(0) {}
};

// ============================================================================
// Spatial Grid for Collision Optimization
// ============================================================================

class SpatialGrid {
private:
    std::vector<int> particle_indices;
    std::vector<int> cell_starts;
    std::vector<int> cell_counts;
    int capacity;
    
public:
    SpatialGrid(int max_particles);
    ~SpatialGrid() = default;
    
    void update(const std::vector<Particle>& particles);
    void get_nearby_particles(float x, float y, float radius,
                              std::vector<int>& nearby) const;
    
    // Accessors for C-style access (needed for MPI/CUDA)
    int* get_particle_indices() { return particle_indices.data(); }
    int* get_cell_starts() { return cell_starts.data(); }
    int* get_cell_counts() { return cell_counts.data(); }
    const int* get_particle_indices() const { return particle_indices.data(); }
    const int* get_cell_starts() const { return cell_starts.data(); }
    const int* get_cell_counts() const { return cell_counts.data(); }
};

// ============================================================================
// Main Simulation Class
// ============================================================================

class Simulation {
private:
    std::vector<Particle> particles;
    std::unique_ptr<SpatialGrid> grid;
    int max_particles;
    
    // Physics parameters
    float friction;
    float restitution;
    float mouse_force;
    
    // Mouse state
    int mouse_x, mouse_y;
    bool mouse_pressed;
    bool mouse_attract;
    
    // Simulation control
    bool running;
    bool reset_requested;
    ParallelMode mode;
    
    // Performance metrics
    DetailedMetrics metrics;
    
    // Logging control
    bool verbose_logging;
    int frame_counter;
    
public:
    Simulation(int particle_count = DEFAULT_PARTICLE_COUNT);
    ~Simulation();
    
    // Main update and control
    void update(float dt);
    void reset();
    
    // Particle management
    void spawn_particle(float x, float y, float vx, float vy);
    void spawn_random_particles(int count);
    void add_particles(int count);
    void remove_particles(int count);
    
    // Setters
    void set_mode(ParallelMode new_mode) { mode = new_mode; }
    void set_running(bool state) { running = state; }
    void set_mouse_state(int x, int y, bool pressed, bool attract);
    void adjust_friction(float delta);
    void request_reset() { reset_requested = true; }
    void set_verbose_logging(bool enabled) { verbose_logging = enabled; }
    
    // Getters
    const std::vector<Particle>& get_particles() const { return particles; }
    std::vector<Particle>& get_particles() { return particles; }
    Particle* get_particle_data() { return particles.data(); }
    int get_particle_count() const { return static_cast<int>(particles.size()); }
    int get_max_particles() const { return max_particles; }
    ParallelMode get_mode() const { return mode; }
    bool is_running() const { return running; }
    float get_friction() const { return friction; }
    float get_restitution() const { return restitution; }
    float get_mouse_force() const { return mouse_force; }
    int get_mouse_x() const { return mouse_x; }
    int get_mouse_y() const { return mouse_y; }
    bool is_mouse_pressed() const { return mouse_pressed; }
    bool is_mouse_attract() const { return mouse_attract; }
    const DetailedMetrics& get_metrics() const { return metrics; }
    DetailedMetrics& get_metrics_mutable() { return metrics; }
    SpatialGrid& get_grid() { return *grid; }
    bool is_verbose_logging() const { return verbose_logging; }
    
    // Physics time tracking
    void set_physics_time(double time_ms) { metrics.total_physics_time_ms = time_ms; }
    void set_render_time(double time_ms) { metrics.total_render_time_ms = time_ms; }
    void set_fps(double fps_value) { metrics.fps = fps_value; }
    void set_temperature(float temp) { metrics.temperature_c = temp; }
    void set_power(float power) { metrics.power_watts = power; }
    void increment_frame_counter() { frame_counter++; }
    int get_frame_counter() const { return frame_counter; }
};

// ============================================================================
// Physics Engine Interface
// ============================================================================

class PhysicsEngine {
public:
    static void update_sequential(Simulation& sim, float dt);
    static void update_multithreaded(Simulation& sim, float dt);
    static void update_mpi(Simulation& sim, float dt);
    static void update_gpu_simple(Simulation& sim, float dt);
    static void update_gpu_complex(Simulation& sim, float dt);
    
private:
    static void resolve_particle_collision(Particle& p1, Particle& p2, float restitution);
    static void resolve_wall_collision(Particle& p, float friction);
    static void apply_mouse_force(Particle& p, int mouse_x, int mouse_y, 
                                  float force, bool attract);
    static void limit_velocity(Particle& p, float max_vel);
};

// ============================================================================
// System Monitoring (Platform-Specific)
// ============================================================================

class SystemMonitor {
public:
    static void update_metrics(Simulation& sim);
    static float read_temperature();
    static float read_power();
    static void log_detailed_metrics(const Simulation& sim);
};

// ============================================================================
// Utility Functions
// ============================================================================

class Utils {
public:
    static double get_time_ms();
    static double get_time_us(); // Microsecond precision
    static std::string get_mode_name(ParallelMode mode);
    static void print_performance_summary(const DetailedMetrics& metrics, ParallelMode mode);
};

// ============================================================================
// CUDA Interface (defined in PhysicsGPU.cu)
// ============================================================================

#ifdef USE_CUDA
extern "C" {
    void init_gpu_memory(int max_particles);
    void cleanup_gpu_memory();
    void update_physics_gpu_simple_cuda(Simulation* sim, float dt);
    void update_physics_gpu_complex_cuda(Simulation* sim, float dt);
    size_t get_gpu_memory_usage();
}
#endif

#endif // PARTICLE_SIMULATION_HPP
