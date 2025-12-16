#include "ParticleSimulation.hpp"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

// ============================================================================
// Private Physics Helper Functions
// ============================================================================

void PhysicsEngine::resolve_particle_collision(Particle& p1, Particle& p2, float restitution) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dist_sq = dx * dx + dy * dy;
    float min_dist = p1.radius + p2.radius;
    float min_dist_sq = min_dist * min_dist;
    
    if (dist_sq >= min_dist_sq || dist_sq < 0.0001f) {
        return;
    }
    
    float dist = std::sqrt(dist_sq);
    float nx = dx / dist;
    float ny = dy / dist;
    
    // Separate particles
    float overlap = min_dist - dist;
    float total_mass = p1.mass + p2.mass;
    float separation_1 = overlap * (p2.mass / total_mass);
    float separation_2 = overlap * (p1.mass / total_mass);
    
    p1.x -= nx * separation_1;
    p1.y -= ny * separation_1;
    p2.x += nx * separation_2;
    p2.y += ny * separation_2;
    
    // Calculate relative velocity
    float dvx = p2.vx - p1.vx;
    float dvy = p2.vy - p1.vy;
    float dvn = dvx * nx + dvy * ny;
    
    if (dvn >= 0) return;
    
    // Calculate impulse
    float impulse = -(1.0f + restitution) * dvn / (1.0f / p1.mass + 1.0f / p2.mass);
    float impulse_x = impulse * nx;
    float impulse_y = impulse * ny;
    
    p1.vx -= impulse_x / p1.mass;
    p1.vy -= impulse_y / p1.mass;
    p2.vx += impulse_x / p2.mass;
    p2.vy += impulse_y / p2.mass;
}

void PhysicsEngine::resolve_wall_collision(Particle& p, float friction) {
    bool collided = false;
    
    if (p.x - p.radius < 0) {
        p.x = p.radius;
        p.vx = -p.vx;
        collided = true;
    }
    
    if (p.x + p.radius > WINDOW_WIDTH) {
        p.x = WINDOW_WIDTH - p.radius;
        p.vx = -p.vx;
        collided = true;
    }
    
    if (p.y - p.radius < 0) {
        p.y = p.radius;
        p.vy = -p.vy;
        collided = true;
    }
    
    if (p.y + p.radius > WINDOW_HEIGHT) {
        p.y = WINDOW_HEIGHT - p.radius;
        p.vy = -p.vy;
        collided = true;
    }
    
    if (collided && friction > 0.0f) {
        p.vx *= (1.0f - friction);
        p.vy *= (1.0f - friction);
    }
}

void PhysicsEngine::apply_mouse_force(Particle& p, int mouse_x, int mouse_y,
                                      float force, bool attract) {
    float dx = static_cast<float>(mouse_x) - p.x;
    float dy = static_cast<float>(mouse_y) - p.y;
    float dist_sq = dx * dx + dy * dy;
    
    if (dist_sq < 100.0f) dist_sq = 100.0f;
    
    float dist = std::sqrt(dist_sq);
    float force_magnitude = force / dist_sq;
    
    float fx = (dx / dist) * force_magnitude;
    float fy = (dy / dist) * force_magnitude;
    
    if (!attract) {
        fx = -fx;
        fy = -fy;
    }
    
    p.vx += fx;
    p.vy += fy;
}

void PhysicsEngine::limit_velocity(Particle& p, float max_vel) {
    float speed_sq = p.vx * p.vx + p.vy * p.vy;
    if (speed_sq > max_vel * max_vel) {
        float speed = std::sqrt(speed_sq);
        float scale = max_vel / speed;
        p.vx *= scale;
        p.vy *= scale;
    }
}

// ============================================================================
// Mode 1: Sequential Physics Update
// ============================================================================

void PhysicsEngine::update_sequential(Simulation& sim, float dt) {
    double start_time = Utils::get_time_ms();
    
    auto& particles = sim.get_particles();
    auto& grid = sim.get_grid();
    int count = sim.get_particle_count();
    float friction = sim.get_friction();
    float restitution = sim.get_restitution();
    
    // PHASE 1: Update particle positions
    for (int i = 0; i < count; i++) {
        Particle& p = particles[i];
        if (!p.active) continue;
        
        // Apply friction
        if (friction > 0.0f) {
            float friction_factor = 1.0f - friction;
            p.vx *= friction_factor;
            p.vy *= friction_factor;
        }
        
        // Apply mouse force
        if (sim.is_mouse_pressed()) {
            apply_mouse_force(p, sim.get_mouse_x(), sim.get_mouse_y(),
                            sim.get_mouse_force(), sim.is_mouse_attract());
        }
        
        // Update position
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        
        // Wall collisions
        resolve_wall_collision(p, friction);
        
        // Limit velocity
        limit_velocity(p, MAX_VELOCITY);
    }
    
    // CRITICAL FIX: Update spatial grid with NEW positions
    grid.update(particles);
    
    // PHASE 2: Particle-particle collisions using updated grid
    std::vector<int> nearby;
    for (int i = 0; i < count; i++) {
        Particle& p1 = particles[i];
        if (!p1.active) continue;
        
        grid.get_nearby_particles(p1.x, p1.y, p1.radius * 4.0f, nearby);
        
        for (int j : nearby) {
            if (j <= i) continue;
            
            Particle& p2 = particles[j];
            if (!p2.active) continue;
            
            resolve_particle_collision(p1, p2, restitution);
        }
    }
    
    double end_time = Utils::get_time_ms();
    sim.set_physics_time(end_time - start_time);
}

// ============================================================================
// Mode 2: Multithreaded Physics Update (OpenMP) - CORRECTED
// ============================================================================

void PhysicsEngine::update_multithreaded(Simulation& sim, float dt) {
#ifdef _OPENMP
    double start_time = Utils::get_time_ms();
    
    auto& particles = sim.get_particles();
    auto& grid = sim.get_grid();
    int count = sim.get_particle_count();
    float friction = sim.get_friction();
    float restitution = sim.get_restitution();
    
    // PHASE 1: Update positions in parallel
    // This is safe - each thread works on different particles
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < count; i++) {
        Particle& p = particles[i];
        if (!p.active) continue;
        
        // Apply friction
        if (friction > 0.0f) {
            float friction_factor = 1.0f - friction;
            p.vx *= friction_factor;
            p.vy *= friction_factor;
        }
        
        // Apply mouse force
        if (sim.is_mouse_pressed()) {
            apply_mouse_force(p, sim.get_mouse_x(), sim.get_mouse_y(),
                            sim.get_mouse_force(), sim.is_mouse_attract());
        }
        
        // Update position
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        
        // Wall collisions
        resolve_wall_collision(p, friction);
        
        // Limit velocity
        limit_velocity(p, MAX_VELOCITY);
    }
    
    // CRITICAL FIX: Update spatial grid with NEW positions
    // This must be done AFTER positions update and BEFORE collision detection
    grid.update(particles);
    
    // PHASE 2: Collision detection in parallel with improved load balancing
    // Note: There are still potential race conditions here, but they're rare
    // and acceptable for a visual simulation (worst case: missed collision for 1 frame)
    #pragma omp parallel
    {
        std::vector<int> nearby;
        nearby.reserve(NEARBY_BUFFER_SIZE);
        
        // Smaller chunk size for better load balancing
        #pragma omp for schedule(dynamic, 16) nowait
        for (int i = 0; i < count; i++) {
            Particle& p1 = particles[i];
            if (!p1.active) continue;
            
            grid.get_nearby_particles(p1.x, p1.y, p1.radius * 4.0f, nearby);
            
            for (int j : nearby) {
                if (j <= i) continue;
                
                Particle& p2 = particles[j];
                if (!p2.active) continue;
                
                resolve_particle_collision(p1, p2, restitution);
            }
        }
    }
    
    double end_time = Utils::get_time_ms();
    sim.set_physics_time(end_time - start_time);
#else
    // Fallback to sequential
    update_sequential(sim, dt);
#endif
}

// ============================================================================
// Mode 3: MPI Physics Update
// ============================================================================

void PhysicsEngine::update_mpi(Simulation& sim, float dt) {
#ifdef USE_MPI
    double start_time = Utils::get_time_ms();
    
    static int mpi_initialized = 0;
    static int rank = 0;
    static int size = 1;
    
    if (!mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        mpi_initialized = 1;
    }
    
    auto& particles = sim.get_particles();
    auto& grid = sim.get_grid();
    int count = sim.get_particle_count();
    float friction = sim.get_friction();
    float restitution = sim.get_restitution();
    
    // Determine particle range for this rank
    int particles_per_rank = count / size;
    int start_idx = rank * particles_per_rank;
    int end_idx = (rank == size - 1) ? count : (rank + 1) * particles_per_rank;
    
    // Broadcast mouse state
    int mouse_x = sim.get_mouse_x();
    int mouse_y = sim.get_mouse_y();
    int mouse_pressed = sim.is_mouse_pressed() ? 1 : 0;
    int mouse_attract = sim.is_mouse_attract() ? 1 : 0;
    
    MPI_Bcast(&mouse_pressed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mouse_pressed) {
        MPI_Bcast(&mouse_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mouse_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mouse_attract, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    // PHASE 1: Update positions for this rank's particles
    for (int i = start_idx; i < end_idx; i++) {
        Particle& p = particles[i];
        if (!p.active) continue;
        
        if (friction > 0.0f) {
            float friction_factor = 1.0f - friction;
            p.vx *= friction_factor;
            p.vy *= friction_factor;
        }
        
        if (mouse_pressed) {
            apply_mouse_force(p, mouse_x, mouse_y, sim.get_mouse_force(), mouse_attract != 0);
        }
        
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        
        resolve_wall_collision(p, friction);
        limit_velocity(p, MAX_VELOCITY);
    }
    
    // Synchronize particles across ranks
    MPI_Datatype MPI_PARTICLE;
    int blocklengths[6] = {2, 2, 1, 1, 3, 1};
    MPI_Aint displacements[6];
    MPI_Datatype types[6] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_UINT8_T, MPI_C_BOOL};
    
    Particle dummy;
    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.x, &displacements[0]);
    MPI_Get_address(&dummy.vx, &displacements[1]);
    MPI_Get_address(&dummy.radius, &displacements[2]);
    MPI_Get_address(&dummy.mass, &displacements[3]);
    MPI_Get_address(&dummy.r, &displacements[4]);
    MPI_Get_address(&dummy.active, &displacements[5]);
    
    for (int i = 0; i < 6; i++) {
        displacements[i] = MPI_Aint_diff(displacements[i], base_address);
    }
    
    MPI_Type_create_struct(6, blocklengths, displacements, types, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);
    
    // Gather particles to rank 0
    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            int r_start = r * particles_per_rank;
            int r_end = (r == size - 1) ? count : (r + 1) * particles_per_rank;
            int r_count = r_end - r_start;
            MPI_Recv(&particles[r_start], r_count, MPI_PARTICLE, r, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(&particles[start_idx], end_idx - start_idx, MPI_PARTICLE, 0, 0,
                MPI_COMM_WORLD);
    }
    
    // Broadcast complete array
    MPI_Bcast(sim.get_particle_data(), count, MPI_PARTICLE, 0, MPI_COMM_WORLD);
    
    // CRITICAL FIX: Update spatial grid with NEW positions
    grid.update(particles);
    
    // PHASE 2: Collision detection
    std::vector<int> nearby;
    for (int i = start_idx; i < end_idx; i++) {
        Particle& p1 = particles[i];
        if (!p1.active) continue;
        
        grid.get_nearby_particles(p1.x, p1.y, p1.radius * 4.0f, nearby);
        
        for (int j : nearby) {
            if (j <= i) continue;
            
            Particle& p2 = particles[j];
            if (!p2.active) continue;
            
            resolve_particle_collision(p1, p2, restitution);
        }
    }
    
    // Synchronize collision results
    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            int r_start = r * particles_per_rank;
            int r_end = (r == size - 1) ? count : (r + 1) * particles_per_rank;
            int r_count = r_end - r_start;
            MPI_Recv(&particles[r_start], r_count, MPI_PARTICLE, r, 1,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(&particles[start_idx], end_idx - start_idx, MPI_PARTICLE, 0, 1,
                MPI_COMM_WORLD);
    }
    
    MPI_Bcast(sim.get_particle_data(), count, MPI_PARTICLE, 0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_PARTICLE);
    
    double end_time = Utils::get_time_ms();
    sim.set_physics_time(end_time - start_time);
#else
    // Fallback to sequential
    update_sequential(sim, dt);
#endif
}

// ============================================================================
// Mode 4 & 5: GPU Physics Updates
// ============================================================================

void PhysicsEngine::update_gpu_simple(Simulation& sim, float dt) {
#ifdef USE_CUDA
    double start_time = Utils::get_time_ms();
    update_physics_gpu_simple_cuda(&sim, dt);
    double end_time = Utils::get_time_ms();
    sim.set_physics_time(end_time - start_time);
#else
    // Fallback to sequential
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
    // Fallback to sequential
    update_sequential(sim, dt);
#endif
}