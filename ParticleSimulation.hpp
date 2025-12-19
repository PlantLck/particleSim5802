/*
 * Particle Simulation Header
 * High-performance parallel computing framework for particle physics
 */

#ifndef PARTICLE_SIMULATION_HPP
#define PARTICLE_SIMULATION_HPP

#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>

// ============================================================================
// Forward Declarations
// ============================================================================

class SpatialGrid;
class SystemMonitor;

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr int WINDOW_WIDTH = 1920;
constexpr int WINDOW_HEIGHT = 1080;
constexpr int GRID_CELL_SIZE = 30;
constexpr int GRID_WIDTH = WINDOW_WIDTH / GRID_CELL_SIZE;
constexpr int GRID_HEIGHT = WINDOW_HEIGHT / GRID_CELL_SIZE;
constexpr int MAX_PARTICLES = 20000;
constexpr int DEFAULT_PARTICLE_COUNT = 800;
constexpr float DEFAULT_PARTICLE_RADIUS = 5.0f;
constexpr float DEFAULT_MOUSE_FORCE = 800.0f;
constexpr float MOUSE_FORCE_MIN = 100.0f;
constexpr float MOUSE_FORCE_MAX = 3000.0f;
constexpr float MOUSE_FORCE_STEP = 100.0f;
constexpr float MOUSE_FORCE_RADIUS = 400.0f;

// ============================================================================
// Optimized Particle Structure
// ============================================================================

struct alignas(32) Particle {
    float x, y;
    float vx, vy;
    float radius;
    float mass;
    uint8_t r, g, b;
    bool active;
    
    Particle() : x(0.0f), y(0.0f), vx(0.0f), vy(0.0f),
                 radius(DEFAULT_PARTICLE_RADIUS), mass(1.0f),
                 r(255), g(255), b(255), active(true) {}
} __attribute__((packed));

static_assert(sizeof(Particle) == 32, "Particle must be exactly 32 bytes");
static_assert(alignof(Particle) == 32, "Particle must be 32-byte aligned");

// ============================================================================
// Parallelization Mode Enumeration
// ============================================================================

enum class ParallelMode {
    SEQUENTIAL = 0,
    MULTITHREADED = 1,
    MPI = 2,
    GPU_SIMPLE = 3,
    GPU_COMPLEX = 4
};

// ============================================================================
// Optimized Spatial Grid
// ============================================================================

class SpatialGrid {
private:
    int grid_width;
    int grid_height;
    float cell_size;
    int num_cells;
    
    std::vector<int> particle_indices;
    std::vector<int> cell_starts;
    std::vector<int> cell_counts;
    
public:
    SpatialGrid(int width, int height, float size)
        : grid_width(width), grid_height(height),
          cell_size(size), num_cells(width * height) {
        
        cell_starts.resize(num_cells, 0);
        cell_counts.resize(num_cells, 0);
        particle_indices.resize(MAX_PARTICLES, -1);
    }
    
    inline int get_cell_index(float x, float y) const {
        int grid_x = static_cast<int>(x / cell_size);
        int grid_y = static_cast<int>(y / cell_size);
        
        if (grid_x < 0 || grid_x >= grid_width ||
            grid_y < 0 || grid_y >= grid_height) {
            return -1;
        }
        
        return grid_y * grid_width + grid_x;
    }
    
    void update(std::vector<Particle>& particles);
    
    inline void get_nearby_particles(float x, float y, float radius,
                                     std::vector<int>& nearby) const {
        nearby.clear();
        
        int min_grid_x = static_cast<int>((x - radius) / cell_size);
        int max_grid_x = static_cast<int>((x + radius) / cell_size);
        int min_grid_y = static_cast<int>((y - radius) / cell_size);
        int max_grid_y = static_cast<int>((y + radius) / cell_size);
        
        if (min_grid_x < 0) min_grid_x = 0;
        if (max_grid_x >= grid_width) max_grid_x = grid_width - 1;
        if (min_grid_y < 0) min_grid_y = 0;
        if (max_grid_y >= grid_height) max_grid_y = grid_height - 1;
        
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
    
    inline int get_num_cells() const { return num_cells; }
    inline int get_cell_count() const { return num_cells; }
    inline std::vector<int>& get_cell_counts() { return cell_counts; }
    inline std::vector<int>& get_cell_starts() { return cell_starts; }
    inline std::vector<int>& get_particle_indices() { return particle_indices; }
    inline const std::vector<int>& get_cell_counts() const { return cell_counts; }
    inline const std::vector<int>& get_cell_starts() const { return cell_starts; }
    inline const std::vector<int>& get_particle_indices() const { return particle_indices; }
};

// ============================================================================
// Performance Metrics Structure
// ============================================================================

struct PerformanceMetrics {
    double fps;
    double physics_time_ms;
    double render_time_ms;
    double frame_time_ms;
    double total_physics_time_ms;
    double total_render_time_ms;
    float temperature_c;
    float power_watts;
    
    PerformanceMetrics() : fps(0.0), physics_time_ms(0.0), render_time_ms(0.0),
                          frame_time_ms(0.0), total_physics_time_ms(0.0), 
                          total_render_time_ms(0.0), temperature_c(0.0f), 
                          power_watts(0.0f) {}
};

// ============================================================================
// Main Simulation Class
// ============================================================================

class Simulation {
private:
    std::vector<Particle> particles;
    std::unique_ptr<SpatialGrid> grid;
    
    int max_particles;
    int window_width;
    int window_height;
    float friction;
    float restitution;
    float mouse_force;
    float mouse_force_radius;
    
    int mouse_x, mouse_y;
    bool mouse_pressed;
    bool mouse_attract;
    
    bool running;
    bool reset_requested;
    ParallelMode mode;
    bool verbose_logging;
    
    PerformanceMetrics metrics;
    int frame_counter;
    
    double fps_timer;
    int fps_frame_count;
    
public:
    Simulation(int particle_count, int max_count = MAX_PARTICLES);
    ~Simulation();
    
    void update(float dt);
    void reset();
    
    void spawn_particle(float x, float y, float vx, float vy);
    void spawn_random_particles(int count);
    void add_particles(int count);
    void remove_particles(int count);
    
    inline std::vector<Particle>& get_particles() { return particles; }
    inline const std::vector<Particle>& get_particles() const { return particles; }
    inline Particle* get_particle_data() { return particles.data(); }
    inline int get_particle_count() const { return static_cast<int>(particles.size()); }
    inline int get_max_particles() const { return max_particles; }
    
    inline SpatialGrid& get_grid() { return *grid; }
    inline const SpatialGrid& get_grid() const { return *grid; }
    
    inline int get_window_width() const { return window_width; }
    inline int get_window_height() const { return window_height; }
    
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
    inline void adjust_mouse_force(float delta) {
        mouse_force += delta;
        if (mouse_force < MOUSE_FORCE_MIN) mouse_force = MOUSE_FORCE_MIN;
        if (mouse_force > MOUSE_FORCE_MAX) mouse_force = MOUSE_FORCE_MAX;
    }
    inline float get_mouse_force_radius() const { return mouse_force_radius; }
    inline void set_mouse_force_radius(float r) { mouse_force_radius = r; }
    
    inline int get_mouse_x() const { return mouse_x; }
    inline int get_mouse_y() const { return mouse_y; }
    inline bool is_mouse_pressed() const { return mouse_pressed; }
    inline bool is_mouse_attract() const { return mouse_attract; }
    inline void set_mouse_state(int x, int y, bool pressed, bool attract) {
        mouse_x = x; mouse_y = y;
        mouse_pressed = pressed; mouse_attract = attract;
    }
    inline void update_mouse_position(int x, int y) {
        mouse_x = x; mouse_y = y;
    }
    
    inline bool is_running() const { return running; }
    inline void set_running(bool r) { running = r; }
    inline void request_reset() { reset_requested = true; }
    
    inline ParallelMode get_mode() const { return mode; }
    inline void set_mode(ParallelMode m) { mode = m; }
    
    inline const PerformanceMetrics& get_metrics() const { return metrics; }
    inline PerformanceMetrics& get_metrics_mutable() { return metrics; }
    inline void set_fps(double fps) { metrics.fps = fps; }
    inline void set_physics_time(double ms) { 
        metrics.physics_time_ms = ms; 
        metrics.total_physics_time_ms = ms;
    }
    inline void set_render_time(double ms) { 
        metrics.render_time_ms = ms; 
        metrics.total_render_time_ms = ms;
    }
    inline void set_frame_time(double ms) { metrics.frame_time_ms = ms; }
    inline void set_temperature(float temp) { metrics.temperature_c = temp; }
    inline void set_power(float watts) { metrics.power_watts = watts; }
    
    inline bool is_verbose() const { return verbose_logging; }
    inline bool is_verbose_logging() const { return verbose_logging; }
    inline void set_verbose(bool v) { verbose_logging = v; }
    inline void set_verbose_logging(bool v) { verbose_logging = v; }
    
    inline int get_frame_counter() const { return frame_counter; }
    
    void update_fps(double dt);
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
};

// ============================================================================
// Utility Functions
// ============================================================================

class Utils {
public:
    static double get_time_ms();
    static float random_float(float min, float max);
    static int random_int(int min, int max);
    static void print_performance_summary(const PerformanceMetrics& metrics, ParallelMode mode);
    static const char* get_mode_name(ParallelMode mode);
};

// ============================================================================
// System Monitoring
// ============================================================================

class SystemMonitor {
public:
    static void update_metrics(Simulation& sim);
    static float read_temperature();
    static float read_power();
};

// ============================================================================
// CUDA Interface
// ============================================================================

#ifdef USE_CUDA
extern "C" {
    void update_physics_gpu_simple_cuda(Simulation* sim, float dt);
    void update_physics_gpu_complex_cuda(Simulation* sim, float dt);
    void cleanup_gpu_memory();
}
#endif

#endif // PARTICLE_SIMULATION_HPP
