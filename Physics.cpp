/*
 * Physics Engine Implementation
 * All five parallelization modes: Sequential, OpenMP, MPI, GPU Simple, GPU Complex
 */

#include "ParticleSimulation.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

// Compiler hints for branch prediction
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

// Constants
constexpr float EPSILON = 1e-6f;
constexpr float MAX_VELOCITY = 500.0f;
constexpr float MOUSE_FORCE_RADIUS = 150.0f;

// Force inlining for hot path functions
#ifdef __GNUC__
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

#ifdef USE_CUDA
extern "C" {
    void update_physics_gpu_simple_cuda(void* sim, float dt);
    void update_physics_gpu_complex_cuda(void* sim, float dt);
}
#endif

// ============================================================================
// Helper Functions
// ============================================================================

FORCE_INLINE void apply_mouse_force_optimized(Particle& p, int mx, int my, 
                                               float force_magnitude, bool attract) {
    float dx = mx - p.x;
    float dy = my - p.y;
    float dist_sq = dx * dx + dy * dy;
    float max_dist_sq = MOUSE_FORCE_RADIUS * MOUSE_FORCE_RADIUS;
    
    if (dist_sq > max_dist_sq || dist_sq < EPSILON) return;
    
    float dist = sqrtf(dist_sq);
    float force = force_magnitude * (1.0f - dist / MOUSE_FORCE_RADIUS);
    
    if (!attract) force = -force;
    
    float fx = (dx / dist) * force;
    float fy = (dy / dist) * force;
    
    p.vx += fx / p.mass;
    p.vy += fy / p.mass;
}

FORCE_INLINE void resolve_wall_collision_optimized(Particle& p, float restitution,
                                                    int window_width, int window_height) {
    bool collided = false;
    
    if (p.x - p.radius < 0.0f) {
        p.x = p.radius;
        p.vx = fabsf(p.vx) * restitution;
        collided = true;
    } else if (p.x + p.radius > window_width) {
        p.x = window_width - p.radius;
        p.vx = -fabsf(p.vx) * restitution;
        collided = true;
    }
    
    if (p.y - p.radius < 0.0f) {
        p.y = p.radius;
        p.vy = fabsf(p.vy) * restitution;
        collided = true;
    } else if (p.y + p.radius > window_height) {
        p.y = window_height - p.radius;
        p.vy = -fabsf(p.vy) * restitution;
        collided = true;
    }
}

FORCE_INLINE void apply_friction_optimized(Particle& p, float friction) {
    if (friction > 0.0f) {
        float friction_factor = 1.0f - friction;
        p.vx *= friction_factor;
        p.vy *= friction_factor;
    }
}

FORCE_INLINE void limit_velocity_optimized(Particle& p, float max_vel) {
    float speed_sq = p.vx * p.vx + p.vy * p.vy;
    float max_sq = max_vel * max_vel;
    
    if (LIKELY(speed_sq <= max_sq)) return;
    
    float speed = sqrtf(speed_sq);
    float scale = max_vel / speed;
    p.vx *= scale;
    p.vy *= scale;
}

FORCE_INLINE bool resolve_particle_collision_optimized(Particle& p1, Particle& p2,
                                                        float restitution) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dist_sq = dx * dx + dy * dy;
    
    float min_dist = p1.radius + p2.radius;
    float min_dist_sq = min_dist * min_dist;
    
    if (LIKELY(dist_sq >= min_dist_sq)) return false;
    if (UNLIKELY(dist_sq < EPSILON)) return false;
    
    // Check if particles are already separating
    float dvx = p2.vx - p1.vx;
    float dvy = p2.vy - p1.vy;
    float relative_velocity = dvx * dx + dvy * dy;
    if (relative_velocity > 0.0f) return false;
    
    float dist = sqrtf(dist_sq);
    
    // Separate overlapping particles
    float overlap = min_dist - dist;
    float total_mass = p1.mass + p2.mass;
    float ratio1 = p2.mass / total_mass;
    float ratio2 = p1.mass / total_mass;
    
    float sep_x = (dx / dist) * overlap;
    float sep_y = (dy / dist) * overlap;
    
    p1.x -= sep_x * ratio1;
    p1.y -= sep_y * ratio1;
    p2.x += sep_x * ratio2;
    p2.y += sep_y * ratio2;
    
    // Calculate collision impulse
    float nx = dx / dist;
    float ny = dy / dist;
    
    float rel_vel_normal = dvx * nx + dvy * ny;
    float impulse_mag = -(1.0f + restitution) * rel_vel_normal / 
                        (1.0f / p1.mass + 1.0f / p2.mass);
    
    float impulse_x = impulse_mag * nx;
    float impulse_y = impulse_mag * ny;
    
    p1.vx -= impulse_x / p1.mass;
    p1.vy -= impulse_y / p1.mass;
    p2.vx += impulse_x / p2.mass;
    p2.vy += impulse_y / p2.mass;
    
    return true;
}

// ============================================================================
// Sequential Implementation
// ============================================================================

void PhysicsEngine::update_sequential(Simulation& sim, float dt) {
    double start_time = Utils::get_time_ms();
    
    auto& particles = sim.get_particles();
    auto& grid = sim.get_grid();
    int count = sim.get_particle_count();
    int window_width = sim.get_window_width();
    int window_height = sim.get_window_height();
    float friction = sim.get_friction();
    float restitution = sim.get_restitution();
    
    bool mouse_pressed = sim.is_mouse_pressed();
    int mouse_x = sim.get_mouse_x();
    int mouse_y = sim.get_mouse_y();
    bool mouse_attract = sim.is_mouse_attract();
    float mouse_force = sim.get_mouse_force();
    
    // Phase 1: Position updates
    #pragma GCC ivdep
    for (int i = 0; i < count; i++) {
        Particle& p = particles[i];
        
        if (UNLIKELY(!p.active)) continue;
        
        if (friction > 0.0f) {
            apply_friction_optimized(p, friction);
        }
        
        if (UNLIKELY(mouse_pressed)) {
            apply_mouse_force_optimized(p, mouse_x, mouse_y, mouse_force, mouse_attract);
        }
        
        // Euler integration
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        
        resolve_wall_collision_optimized(p, restitution, window_width, window_height);
        limit_velocity_optimized(p, MAX_VELOCITY);
    }
    
    // Update spatial grid
    grid.update(particles);
    
    // Phase 2: Collision detection using spatial grid
    for (int i = 0; i < count; i++) {
        Particle& p1 = particles[i];
        if (!p1.active) continue;
        
        auto nearby = grid.get_nearby_particles(p1.x, p1.y, particles);
        
        for (int j : nearby) {
            if (j <= i) continue;
            Particle& p2 = particles[j];
            if (!p2.active) continue;
            
            resolve_particle_collision_optimized(p1, p2, restitution);
        }
    }
    
    double end_time = Utils::get_time_ms();
    sim.set_physics_time(end_time - start_time);
}

// ============================================================================
// OpenMP Implementation
// ============================================================================

#ifdef _OPENMP

// Parallel prefix sum for grid construction
void parallel_prefix_sum(const int* input, int* output, int size) {
    output[0] = 0;
    for (int i = 1; i < size; i++) {
        output[i] = output[i-1] + input[i-1];
    }
}

// Parallel grid update using thread-local counting
void update_grid_parallel(SpatialGrid& grid, std::vector<Particle>& particles) {
    int particle_count = particles.size();
    int num_cells = grid.get_cell_count();
    
    auto& cell_counts = grid.get_cell_counts();
    std::fill(cell_counts.begin(), cell_counts.end(), 0);
    
    int num_threads;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    
    // Thread-local cell counts
    std::vector<std::vector<int>> thread_cell_counts(num_threads);
    for (auto& counts : thread_cell_counts) {
        counts.resize(num_cells, 0);
    }
    
    // Phase 1: Parallel counting
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_counts = thread_cell_counts[tid];
        
        #pragma omp for schedule(static)
        for (int i = 0; i < particle_count; i++) {
            if (!particles[i].active) continue;
            
            int cell = grid.get_cell_index(particles[i].x, particles[i].y);
            if (cell >= 0 && cell < num_cells) {
                local_counts[cell]++;
            }
        }
    }
    
    // Phase 2: Reduce thread-local counts
    for (int cell = 0; cell < num_cells; cell++) {
        int total = 0;
        for (int tid = 0; tid < num_threads; tid++) {
            total += thread_cell_counts[tid][cell];
        }
        cell_counts[cell] = total;
    }
    
    // Phase 3: Prefix sum
    auto& cell_starts = grid.get_cell_starts();
    parallel_prefix_sum(cell_counts.data(), cell_starts.data(), num_cells);
    
    // Phase 4: Fill particle indices
    std::vector<std::vector<int>> thread_positions(num_threads);
    for (auto& pos : thread_positions) {
        pos.resize(num_cells, 0);
    }
    
    #pragma omp parallel for schedule(static)
    for (int cell = 0; cell < num_cells; cell++) {
        int offset = cell_starts[cell];
        for (int tid = 0; tid < num_threads; tid++) {
            thread_positions[tid][cell] = offset;
            offset += thread_cell_counts[tid][cell];
        }
    }
    
    auto& particle_indices = grid.get_particle_indices();
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& positions = thread_positions[tid];
        
        #pragma omp for schedule(static)
        for (int i = 0; i < particle_count; i++) {
            if (!particles[i].active) continue;
            
            int cell = grid.get_cell_index(particles[i].x, particles[i].y);
            if (cell >= 0 && cell < num_cells) {
                int pos = positions[cell]++;
                particle_indices[pos] = i;
            }
        }
    }
}

// Collision pair structure for deferred resolution
struct CollisionPair {
    int p1_idx;
    int p2_idx;
    float impulse_x;
    float impulse_y;
};

void PhysicsEngine::update_multithreaded(Simulation& sim, float dt) {
    double start_time = Utils::get_time_ms();
    
    auto& particles = sim.get_particles();
    auto& grid = sim.get_grid();
    int count = sim.get_particle_count();
    int window_width = sim.get_window_width();
    int window_height = sim.get_window_height();
    float friction = sim.get_friction();
    float restitution = sim.get_restitution();
    
    bool mouse_pressed = sim.is_mouse_pressed();
    int mouse_x = sim.get_mouse_x();
    int mouse_y = sim.get_mouse_y();
    bool mouse_attract = sim.is_mouse_attract();
    float mouse_force = sim.get_mouse_force();
    
    // Persistent parallel region to eliminate fork-join overhead
    #pragma omp parallel
    {
        // Thread-local collision storage
        std::vector<CollisionPair> local_collisions;
        local_collisions.reserve(count / omp_get_num_threads());
        
        // Phase 1: Position updates
        #pragma omp for schedule(static)
        for (int i = 0; i < count; i++) {
            Particle& p = particles[i];
            
            if (!p.active) continue;
            
            if (friction > 0.0f) {
                apply_friction_optimized(p, friction);
            }
            
            if (mouse_pressed) {
                apply_mouse_force_optimized(p, mouse_x, mouse_y, mouse_force, mouse_attract);
            }
            
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            
            resolve_wall_collision_optimized(p, restitution, window_width, window_height);
            limit_velocity_optimized(p, MAX_VELOCITY);
        }
        
        #pragma omp barrier
        #pragma omp single
        {
            update_grid_parallel(grid, particles);
        }
        #pragma omp barrier
        
        // Phase 2: Collision detection
        std::vector<int> nearby;
        nearby.reserve(64);
        
        #pragma omp for schedule(dynamic, 32)
        for (int i = 0; i < count; i++) {
            Particle& p1 = particles[i];
            if (!p1.active) continue;
            
            nearby.clear();
            nearby = grid.get_nearby_particles(p1.x, p1.y, particles);
            
            for (int j : nearby) {
                if (j <= i) continue;
                Particle& p2 = particles[j];
                if (!p2.active) continue;
                
                float dx = p2.x - p1.x;
                float dy = p2.y - p1.y;
                float dist_sq = dx * dx + dy * dy;
                float min_dist = p1.radius + p2.radius;
                float min_dist_sq = min_dist * min_dist;
                
                if (dist_sq >= min_dist_sq || dist_sq < EPSILON) continue;
                
                float dvx = p2.vx - p1.vx;
                float dvy = p2.vy - p1.vy;
                if (dvx * dx + dvy * dy > 0.0f) continue;
                
                float dist = sqrtf(dist_sq);
                float overlap = min_dist - dist;
                float total_mass = p1.mass + p2.mass;
                
                float sep_x = (dx / dist) * overlap * (p2.mass / total_mass);
                float sep_y = (dy / dist) * overlap * (p2.mass / total_mass);
                
                #pragma omp atomic
                p1.x -= sep_x;
                #pragma omp atomic
                p1.y -= sep_y;
                #pragma omp atomic
                p2.x += sep_x;
                #pragma omp atomic
                p2.y += sep_y;
                
                float nx = dx / dist;
                float ny = dy / dist;
                float rel_vel_normal = dvx * nx + dvy * ny;
                float impulse_mag = -(1.0f + restitution) * rel_vel_normal / 
                                    (1.0f / p1.mass + 1.0f / p2.mass);
                
                CollisionPair cp;
                cp.p1_idx = i;
                cp.p2_idx = j;
                cp.impulse_x = impulse_mag * nx;
                cp.impulse_y = impulse_mag * ny;
                local_collisions.push_back(cp);
            }
        }
        
        #pragma omp barrier
        
        // Phase 3: Apply collision impulses
        #pragma omp for schedule(static)
        for (int i = 0; i < count; i++) {
            for (int tid = 0; tid < omp_get_num_threads(); tid++) {
                for (const auto& cp : local_collisions) {
                    if (cp.p1_idx == i) {
                        particles[i].vx -= cp.impulse_x / particles[i].mass;
                        particles[i].vy -= cp.impulse_y / particles[i].mass;
                    }
                    if (cp.p2_idx == i) {
                        particles[i].vx += cp.impulse_x / particles[i].mass;
                        particles[i].vy += cp.impulse_y / particles[i].mass;
                    }
                }
            }
        }
    }
    
    double end_time = Utils::get_time_ms();
    sim.set_physics_time(end_time - start_time);
}

#else
void PhysicsEngine::update_multithreaded(Simulation& sim, float dt) {
    update_sequential(sim, dt);
}
#endif

// ============================================================================
// MPI Implementation
// ============================================================================

#ifdef USE_MPI

// Minimal structure for MPI communication
struct ParticlePhysics {
    float x, y;
    float vx, vy;
};

struct MouseState {
    int x, y;
    int pressed;
    int attract;
};

void PhysicsEngine::update_mpi(Simulation& sim, float dt) {
    double start_time = Utils::get_time_ms();
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    auto& particles = sim.get_particles();
    auto& grid = sim.get_grid();
    int count = sim.get_particle_count();
    int window_width = sim.get_window_width();
    int window_height = sim.get_window_height();
    float friction = sim.get_friction();
    float restitution = sim.get_restitution();
    
    int local_count = count / size;
    int start_idx = rank * local_count;
    int end_idx = (rank == size - 1) ? count : (rank + 1) * local_count;
    
    // Register MPI datatype
    static MPI_Datatype MPI_PARTICLE_PHYSICS = MPI_DATATYPE_NULL;
    if (MPI_PARTICLE_PHYSICS == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(4, MPI_FLOAT, &MPI_PARTICLE_PHYSICS);
        MPI_Type_commit(&MPI_PARTICLE_PHYSICS);
    }
    
    // Broadcast mouse state
    MouseState mouse;
    if (rank == 0) {
        mouse.x = sim.get_mouse_x();
        mouse.y = sim.get_mouse_y();
        mouse.pressed = sim.is_mouse_pressed() ? 1 : 0;
        mouse.attract = sim.is_mouse_attract() ? 1 : 0;
    }
    MPI_Bcast(&mouse, sizeof(MouseState), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Update local particles
    for (int i = start_idx; i < end_idx; i++) {
        Particle& p = particles[i];
        if (!p.active) continue;
        
        if (friction > 0.0f) {
            apply_friction_optimized(p, friction);
        }
        
        if (mouse.pressed) {
            apply_mouse_force_optimized(p, mouse.x, mouse.y, sim.get_mouse_force(), 
                                       mouse.attract != 0);
        }
        
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        
        resolve_wall_collision_optimized(p, restitution, window_width, window_height);
        limit_velocity_optimized(p, MAX_VELOCITY);
    }
    
    // Gather all particle positions
    std::vector<ParticlePhysics> physics_data(count);
    for (int i = start_idx; i < end_idx; i++) {
        physics_data[i].x = particles[i].x;
        physics_data[i].y = particles[i].y;
        physics_data[i].vx = particles[i].vx;
        physics_data[i].vy = particles[i].vy;
    }
    
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  physics_data.data(), local_count, MPI_PARTICLE_PHYSICS,
                  MPI_COMM_WORLD);
    
    for (int i = 0; i < count; i++) {
        particles[i].x = physics_data[i].x;
        particles[i].y = physics_data[i].y;
    }
    
    // Update grid on all ranks
    grid.update(particles);
    
    // Local collision detection
    for (int i = start_idx; i < end_idx; i++) {
        Particle& p1 = particles[i];
        if (!p1.active) continue;
        
        auto nearby = grid.get_nearby_particles(p1.x, p1.y, particles);
        
        for (int j : nearby) {
            if (j <= i) continue;
            Particle& p2 = particles[j];
            if (!p2.active) continue;
            
            resolve_particle_collision_optimized(p1, p2, restitution);
        }
    }
    
    // Gather collision results
    for (int i = start_idx; i < end_idx; i++) {
        physics_data[i].vx = particles[i].vx;
        physics_data[i].vy = particles[i].vy;
    }
    
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  physics_data.data(), local_count, MPI_PARTICLE_PHYSICS,
                  MPI_COMM_WORLD);
    
    for (int i = 0; i < count; i++) {
        particles[i].vx = physics_data[i].vx;
        particles[i].vy = physics_data[i].vy;
    }
    
    double end_time = Utils::get_time_ms();
    sim.set_physics_time(end_time - start_time);
}

#else
void PhysicsEngine::update_mpi(Simulation& sim, float dt) {
    update_sequential(sim, dt);
}
#endif

// ============================================================================
// GPU Implementations
// ============================================================================

void PhysicsEngine::update_gpu_simple(Simulation& sim, float dt) {
#ifdef USE_CUDA
    double start_time = Utils::get_time_ms();
    update_physics_gpu_simple_cuda(&sim, dt);
    double end_time = Utils::get_time_ms();
    sim.set_physics_time(end_time - start_time);
#else
    update_sequential(sim, dt);
#endif
}

void PhysicsEngine::update_gpu_complex(Simulation& sim, float dt) {
#ifdef USE_CUDA
    double start_time = Utils::get_time_ms();
    update_physics_gpu_complex_cuda(&sim, dt);
    double end_time = Utils::get_time_ms();
    sim.set_physics_time(end_time - start_time);
#else
    update_sequential(sim, dt);
#endif
}
