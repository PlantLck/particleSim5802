#include "ParticleSimulation.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>

// ============================================================================
// SpatialGrid Implementation
// ============================================================================

SpatialGrid::SpatialGrid(int max_particles) : capacity(max_particles) {
    int total_cells = GRID_WIDTH * GRID_HEIGHT;
    
    particle_indices.resize(max_particles);
    cell_starts.resize(total_cells);
    cell_counts.resize(total_cells);
    
    std::fill(cell_starts.begin(), cell_starts.end(), 0);
    std::fill(cell_counts.begin(), cell_counts.end(), 0);
}

void SpatialGrid::update(const std::vector<Particle>& particles) {
    int total_cells = GRID_WIDTH * GRID_HEIGHT;
    int count = static_cast<int>(particles.size());
    
    // Clear counts
    std::fill(cell_counts.begin(), cell_counts.end(), 0);
    
    // Count particles in each cell
    for (int i = 0; i < count; i++) {
        if (!particles[i].active) continue;
        
        int grid_x = static_cast<int>(particles[i].x / GRID_CELL_SIZE);
        int grid_y = static_cast<int>(particles[i].y / GRID_CELL_SIZE);
        
        if (grid_x >= 0 && grid_x < GRID_WIDTH && grid_y >= 0 && grid_y < GRID_HEIGHT) {
            int cell_idx = grid_y * GRID_WIDTH + grid_x;
            cell_counts[cell_idx]++;
        }
    }
    
    // Compute cell starts (prefix sum)
    cell_starts[0] = 0;
    for (int i = 1; i < total_cells; i++) {
        cell_starts[i] = cell_starts[i-1] + cell_counts[i-1];
    }
    
    // Reset counts for filling
    std::fill(cell_counts.begin(), cell_counts.end(), 0);
    
    // Fill particle indices
    for (int i = 0; i < count; i++) {
        if (!particles[i].active) continue;
        
        int grid_x = static_cast<int>(particles[i].x / GRID_CELL_SIZE);
        int grid_y = static_cast<int>(particles[i].y / GRID_CELL_SIZE);
        
        if (grid_x >= 0 && grid_x < GRID_WIDTH && grid_y >= 0 && grid_y < GRID_HEIGHT) {
            int cell_idx = grid_y * GRID_WIDTH + grid_x;
            int insert_idx = cell_starts[cell_idx] + cell_counts[cell_idx];
            particle_indices[insert_idx] = i;
            cell_counts[cell_idx]++;
        }
    }
}

void SpatialGrid::get_nearby_particles(float x, float y, float radius,
                                       std::vector<int>& nearby) const {
    nearby.clear();
    
    // Determine grid range to check
    int min_grid_x = static_cast<int>((x - radius) / GRID_CELL_SIZE);
    int max_grid_x = static_cast<int>((x + radius) / GRID_CELL_SIZE);
    int min_grid_y = static_cast<int>((y - radius) / GRID_CELL_SIZE);
    int max_grid_y = static_cast<int>((y + radius) / GRID_CELL_SIZE);
    
    // Clamp to grid bounds
    min_grid_x = std::max(0, min_grid_x);
    max_grid_x = std::min(GRID_WIDTH - 1, max_grid_x);
    min_grid_y = std::max(0, min_grid_y);
    max_grid_y = std::min(GRID_HEIGHT - 1, max_grid_y);
    
    // Collect particles from nearby cells
    for (int gy = min_grid_y; gy <= max_grid_y; gy++) {
        for (int gx = min_grid_x; gx <= max_grid_x; gx++) {
            int cell_idx = gy * GRID_WIDTH + gx;
            int start = cell_starts[cell_idx];
            int count = cell_counts[cell_idx];
            
            for (int i = 0; i < count; i++) {
                nearby.push_back(particle_indices[start + i]);
            }
        }
    }
}

// ============================================================================
// Simulation Implementation
// ============================================================================

Simulation::Simulation(int particle_count)
    : max_particles(MAX_PARTICLES),
      grid(std::make_unique<SpatialGrid>(MAX_PARTICLES)),
      friction(0.001f),
      restitution(1.0f),
      mouse_force(500.0f),
      mouse_x(0),
      mouse_y(0),
      mouse_pressed(false),
      mouse_attract(true),
      running(true),
      reset_requested(false),
      mode(ParallelMode::SEQUENTIAL),
      verbose_logging(false),
      frame_counter(0) {
    
    particles.reserve(MAX_PARTICLES);
    
    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    // Spawn initial particles
    spawn_random_particles(particle_count);
}

Simulation::~Simulation() {
#ifdef USE_CUDA
    cleanup_gpu_memory();
#endif
}

void Simulation::update(float dt) {
    if (reset_requested) {
        reset();
    }
    
    if (running) {
        frame_counter++;
        
        // Call appropriate physics update based on mode
        switch (mode) {
            case ParallelMode::SEQUENTIAL:
                PhysicsEngine::update_sequential(*this, dt);
                break;
            case ParallelMode::MULTITHREADED:
                PhysicsEngine::update_multithreaded(*this, dt);
                break;
            case ParallelMode::MPI:
                PhysicsEngine::update_mpi(*this, dt);
                break;
            case ParallelMode::GPU_SIMPLE:
                PhysicsEngine::update_gpu_simple(*this, dt);
                break;
            case ParallelMode::GPU_COMPLEX:
                PhysicsEngine::update_gpu_complex(*this, dt);
                break;
        }
    }
}

void Simulation::reset() {
    int current_count = static_cast<int>(particles.size());
    particles.clear();
    spawn_random_particles(current_count);
    reset_requested = false;
    frame_counter = 0;
}

void Simulation::spawn_particle(float x, float y, float vx, float vy) {
    if (static_cast<int>(particles.size()) >= max_particles) {
        return;
    }
    
    Particle p;
    p.x = x;
    p.y = y;
    p.vx = vx;
    p.vy = vy;
    p.radius = DEFAULT_PARTICLE_RADIUS;
    p.mass = static_cast<float>(M_PI) * p.radius * p.radius;
    
    // Random color
    p.r = static_cast<uint8_t>(std::rand() % 156 + 100);
    p.g = static_cast<uint8_t>(std::rand() % 156 + 100);
    p.b = static_cast<uint8_t>(std::rand() % 156 + 100);
    
    p.active = true;
    
    particles.push_back(p);
}

void Simulation::spawn_random_particles(int count) {
    for (int i = 0; i < count; i++) {
        float x = static_cast<float>(std::rand() % (WINDOW_WIDTH - 40) + 20);
        float y = static_cast<float>(std::rand() % (WINDOW_HEIGHT - 40) + 20);
        float angle = static_cast<float>(std::rand() % 360) * static_cast<float>(M_PI) / 180.0f;
        float speed = static_cast<float>(std::rand() % 200 + 100);
        float vx = std::cos(angle) * speed;
        float vy = std::sin(angle) * speed;
        
        spawn_particle(x, y, vx, vy);
    }
}

void Simulation::add_particles(int count) {
    spawn_random_particles(count);
}

void Simulation::remove_particles(int count) {
    int to_remove = std::min(count, static_cast<int>(particles.size()));
    if (to_remove > 0) {
        particles.erase(particles.end() - to_remove, particles.end());
    }
}

void Simulation::set_mouse_state(int x, int y, bool pressed, bool attract) {
    mouse_x = x;
    mouse_y = y;
    mouse_pressed = pressed;
    mouse_attract = attract;
}

void Simulation::adjust_friction(float delta) {
    friction += delta;
    friction = std::max(0.0f, std::min(0.1f, friction));
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

double Utils::get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

double Utils::get_time_us() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::micro>(duration).count();
}

std::string Utils::get_mode_name(ParallelMode mode) {
    switch (mode) {
        case ParallelMode::SEQUENTIAL:    return "Sequential";
        case ParallelMode::MULTITHREADED: return "Multithreaded (OpenMP)";
        case ParallelMode::MPI:           return "MPI (Distributed)";
        case ParallelMode::GPU_SIMPLE:    return "GPU Simple";
        case ParallelMode::GPU_COMPLEX:   return "GPU Complex";
        default:                          return "Unknown";
    }
}

void Utils::print_performance_summary(const DetailedMetrics& metrics, ParallelMode mode) {
    printf("\n=== Performance Summary ===\n");
    printf("Mode: %s\n", get_mode_name(mode).c_str());
    printf("FPS: %.1f\n", metrics.fps);
    printf("Total Physics: %.2f ms\n", metrics.total_physics_time_ms);
    printf("Total Render: %.2f ms\n", metrics.total_render_time_ms);
    
    if (mode == ParallelMode::GPU_SIMPLE || mode == ParallelMode::GPU_COMPLEX) {
        printf("\nGPU Breakdown:\n");
        printf("  H2D Transfer: %.3f ms\n", metrics.gpu_h2d_transfer_ms);
        printf("  D2H Transfer: %.3f ms\n", metrics.gpu_d2h_transfer_ms);
        printf("  Update Kernel: %.3f ms\n", metrics.gpu_update_kernel_ms);
        printf("  Collision Kernel: %.3f ms\n", metrics.gpu_collision_kernel_ms);
        
        if (mode == ParallelMode::GPU_COMPLEX) {
            printf("  Grid Count: %.3f ms\n", metrics.gpu_grid_count_kernel_ms);
            printf("  Grid Fill: %.3f ms\n", metrics.gpu_grid_fill_kernel_ms);
            printf("  Prefix Sum: %.3f ms\n", metrics.gpu_prefix_sum_ms);
        }
    }
    printf("\n");
}
