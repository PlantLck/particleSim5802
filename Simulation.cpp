/*
 * Simulation and Spatial Grid Implementation
 * Core particle system management and optimized spatial partitioning
 */

#include "ParticleSimulation.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <chrono>

#ifdef USE_CUDA
extern "C" void cleanup_gpu_memory();
#endif

void SpatialGrid::update(std::vector<Particle>& particles) {
    int particle_count = particles.size();
    
    std::fill(cell_counts.begin(), cell_counts.end(), 0);
    std::fill(particle_indices.begin(), particle_indices.end(), -1);
    
    for (int i = 0; i < particle_count; i++) {
        const Particle& p = particles[i];
        if (!p.active) continue;
        
        int cell = get_cell_index(p.x, p.y);
        if (cell >= 0 && cell < num_cells) {
            cell_counts[cell]++;
        }
    }
    
    cell_starts[0] = 0;
    for (int i = 1; i < num_cells; i++) {
        cell_starts[i] = cell_starts[i-1] + cell_counts[i-1];
    }
    
    std::vector<int> current_positions(num_cells);
    std::copy(cell_starts.begin(), cell_starts.end(), current_positions.begin());
    
    for (int i = 0; i < particle_count; i++) {
        const Particle& p = particles[i];
        if (!p.active) continue;
        
        int cell = get_cell_index(p.x, p.y);
        if (cell >= 0 && cell < num_cells) {
            int pos = current_positions[cell]++;
            if (pos < static_cast<int>(particle_indices.size())) {
                particle_indices[pos] = i;
            }
        }
    }
}

Simulation::Simulation(int particle_count, int max_count)
    : max_particles(max_count),
      window_width(WINDOW_WIDTH),
      window_height(WINDOW_HEIGHT),
      friction(0.001f),
      restitution(0.95f),
      mouse_force(DEFAULT_MOUSE_FORCE),
      mouse_force_radius(MOUSE_FORCE_RADIUS),
      mouse_x(0),
      mouse_y(0),
      mouse_pressed(false),
      mouse_attract(true),
      running(true),
      reset_requested(false),
      mode(ParallelMode::SEQUENTIAL),
      verbose_logging(false),
      frame_counter(0),
      fps_timer(0.0),
      fps_frame_count(0) {
    
    grid = std::make_unique<SpatialGrid>(GRID_WIDTH, GRID_HEIGHT, GRID_CELL_SIZE);
    
    particles.reserve(max_particles);
    
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    spawn_random_particles(particle_count);
    
    if (verbose_logging) {
        printf("Simulation initialized: %d particles, %d max\n",
               particle_count, max_particles);
        printf("Grid: %dx%d cells, %d pixels per cell\n",
               GRID_WIDTH, GRID_HEIGHT, GRID_CELL_SIZE);
    }
}

Simulation::~Simulation() {
#ifdef USE_CUDA
    cleanup_gpu_memory();
#endif
    
    if (verbose_logging) {
        printf("Simulation destroyed after %d frames\n", frame_counter);
    }
}

void Simulation::update(float dt) {
    if (reset_requested) {
        reset();
    }
    
    if (running) {
        frame_counter++;
        
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

void Simulation::update_fps(double dt) {
    fps_timer += dt;
    fps_frame_count++;
    
    if (fps_timer >= 0.5) {
        metrics.fps = fps_frame_count / fps_timer;
        fps_timer = 0.0;
        fps_frame_count = 0;
    }
}

void Simulation::reset() {
    int current_count = static_cast<int>(particles.size());
    particles.clear();
    spawn_random_particles(current_count);
    reset_requested = false;
    frame_counter = 0;
    
    if (verbose_logging) {
        printf("Simulation reset: %d particles\n", current_count);
    }
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
    p.mass = p.radius * p.radius * 0.1f;
    
    p.r = Utils::random_int(100, 255);
    p.g = Utils::random_int(100, 255);
    p.b = Utils::random_int(100, 255);
    p.active = true;
    
    particles.push_back(p);
}

void Simulation::spawn_random_particles(int count) {
    for (int i = 0; i < count && static_cast<int>(particles.size()) < max_particles; i++) {
        float x = Utils::random_float(50.0f, window_width - 50.0f);
        float y = Utils::random_float(50.0f, window_height - 50.0f);
        float vx = Utils::random_float(-100.0f, 100.0f);
        float vy = Utils::random_float(-100.0f, 100.0f);
        
        spawn_particle(x, y, vx, vy);
    }
    
    if (verbose_logging) {
        printf("Spawned %d particles, total: %zu\n", count, particles.size());
    }
}

void Simulation::add_particles(int count) {
    int spawned = 0;
    for (int i = 0; i < count && static_cast<int>(particles.size()) < max_particles; i++) {
        float x = Utils::random_float(50.0f, window_width - 50.0f);
        float y = Utils::random_float(50.0f, window_height - 50.0f);
        float vx = Utils::random_float(-100.0f, 100.0f);
        float vy = Utils::random_float(-100.0f, 100.0f);
        
        spawn_particle(x, y, vx, vy);
        spawned++;
    }
    
    if (verbose_logging) {
        printf("Added %d particles, total: %zu\n", spawned, particles.size());
    }
}

void Simulation::remove_particles(int count) {
    int to_remove = std::min(count, static_cast<int>(particles.size()));
    
    if (to_remove > 0) {
        particles.erase(particles.end() - to_remove, particles.end());
        
        if (verbose_logging) {
            printf("Removed %d particles, total: %zu\n", to_remove, particles.size());
        }
    }
}

double Utils::get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

float Utils::random_float(float min, float max) {
    return min + static_cast<float>(std::rand()) / 
           (static_cast<float>(RAND_MAX / (max - min)));
}

int Utils::random_int(int min, int max) {
    return min + std::rand() % (max - min + 1);
}

const char* Utils::get_mode_name(ParallelMode mode) {
    switch (mode) {
        case ParallelMode::SEQUENTIAL: return "Sequential";
        case ParallelMode::MULTITHREADED: return "Multithreaded";
        case ParallelMode::MPI: return "MPI";
        case ParallelMode::GPU_SIMPLE: return "GPU Simple";
        case ParallelMode::GPU_COMPLEX: return "GPU Complex";
        default: return "Unknown";
    }
}
