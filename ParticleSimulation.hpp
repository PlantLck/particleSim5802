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
// Performance Metrics
// ============================================================================

struct PerformanceMetrics {
    double physics_time_ms;
    double render_time_ms;
    double fps;
    float temperature_c;
    float power_watts;
    
    PerformanceMetrics() : physics_time_ms(0), render_time_ms(0), fps(0),
                          temperature_c(0), power_watts(0) {}
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
    PerformanceMetrics metrics;
    
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
    const PerformanceMetrics& get_metrics() const { return metrics; }
    SpatialGrid& get_grid() { return *grid; }
    
    // Physics time tracking
    void set_physics_time(double time_ms) { metrics.physics_time_ms = time_ms; }
    void set_render_time(double time_ms) { metrics.render_time_ms = time_ms; }
    void set_fps(double fps_value) { metrics.fps = fps_value; }
    void set_temperature(float temp) { metrics.temperature_c = temp; }
    void set_power(float power) { metrics.power_watts = power; }
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
};

// ============================================================================
// Utility Functions
// ============================================================================

class Utils {
public:
    static double get_time_ms();
    static std::string get_mode_name(ParallelMode mode);
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
}
#endif

#endif // PARTICLE_SIMULATION_HPP