/* 
 * CUDA GPU Kernels for Parallel Particle Simulation (C++ Version)
 * Implements both simple and optimized GPU acceleration
 */

#include "ParticleSimulation.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>

// Forward declarations
extern "C" void cleanup_gpu_memory();
extern "C" void init_gpu_memory(int max_particles);

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device-side constants
__constant__ float d_friction;
__constant__ float d_restitution;
__constant__ float d_mouse_force;
__constant__ int d_mouse_x;
__constant__ int d_mouse_y;
__constant__ bool d_mouse_pressed;
__constant__ bool d_mouse_attract;

// GPU memory pointers
static Particle *d_particles = nullptr;
static int *d_grid_indices = nullptr;
static int *d_grid_starts = nullptr;
static int *d_grid_counts = nullptr;
static int gpu_max_particles = 0;
static bool gpu_initialized = false;

// ============================================================================
// GPU Memory Management
// ============================================================================

extern "C" void cleanup_gpu_memory() {
    if (!gpu_initialized) return;
    
    if (d_particles) cudaFree(d_particles);
    if (d_grid_indices) cudaFree(d_grid_indices);
    if (d_grid_starts) cudaFree(d_grid_starts);
    if (d_grid_counts) cudaFree(d_grid_counts);
    
    d_particles = nullptr;
    d_grid_indices = nullptr;
    d_grid_starts = nullptr;
    d_grid_counts = nullptr;
    gpu_initialized = false;
}

extern "C" void init_gpu_memory(int max_particles) {
    if (gpu_initialized && gpu_max_particles >= max_particles) {
        return;
    }
    
    if (gpu_initialized) {
        cleanup_gpu_memory();
    }
    
    gpu_max_particles = max_particles;
    
    // Allocate particle memory
    CUDA_CHECK(cudaMalloc(&d_particles, max_particles * sizeof(Particle)));
    
    // Allocate spatial grid memory
    int grid_size = GRID_WIDTH * GRID_HEIGHT;
    CUDA_CHECK(cudaMalloc(&d_grid_indices, max_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_starts, grid_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_counts, grid_size * sizeof(int)));
    
    gpu_initialized = true;
}

// ============================================================================
// Device Functions
// ============================================================================

__device__ void apply_mouse_force_gpu(Particle *p, int mx, int my, float force, bool attract) {
    float dx = static_cast<float>(mx) - p->x;
    float dy = static_cast<float>(my) - p->y;
    float dist_sq = dx * dx + dy * dy;
    
    if (dist_sq < 100.0f) dist_sq = 100.0f;
    
    float dist = sqrtf(dist_sq);
    float force_magnitude = force / dist_sq;
    
    float fx = (dx / dist) * force_magnitude;
    float fy = (dy / dist) * force_magnitude;
    
    if (!attract) {
        fx = -fx;
        fy = -fy;
    }
    
    p->vx += fx;
    p->vy += fy;
}

__device__ void resolve_wall_collision_gpu(Particle *p, float friction) {
    bool collided = false;
    
    if (p->x - p->radius < 0) {
        p->x = p->radius;
        p->vx = -p->vx;
        collided = true;
    }
    
    if (p->x + p->radius > WINDOW_WIDTH) {
        p->x = WINDOW_WIDTH - p->radius;
        p->vx = -p->vx;
        collided = true;
    }
    
    if (p->y - p->radius < 0) {
        p->y = p->radius;
        p->vy = -p->vy;
        collided = true;
    }
    
    if (p->y + p->radius > WINDOW_HEIGHT) {
        p->y = WINDOW_HEIGHT - p->radius;
        p->vy = -p->vy;
        collided = true;
    }
    
    if (collided && friction > 0.0f) {
        p->vx *= (1.0f - friction);
        p->vy *= (1.0f - friction);
    }
}

__device__ void resolve_particle_collision_gpu(Particle *p1, Particle *p2, float restitution) {
    float dx = p2->x - p1->x;
    float dy = p2->y - p1->y;
    float dist_sq = dx * dx + dy * dy;
    float min_dist = p1->radius + p2->radius;
    float min_dist_sq = min_dist * min_dist;
    
    if (dist_sq >= min_dist_sq || dist_sq < 0.0001f) return;
    
    float dist = sqrtf(dist_sq);
    float nx = dx / dist;
    float ny = dy / dist;
    
    // Separate particles
    float overlap = min_dist - dist;
    float total_mass = p1->mass + p2->mass;
    float separation_1 = overlap * (p2->mass / total_mass);
    float separation_2 = overlap * (p1->mass / total_mass);
    
    p1->x -= nx * separation_1;
    p1->y -= ny * separation_1;
    p2->x += nx * separation_2;
    p2->y += ny * separation_2;
    
    // Calculate relative velocity
    float dvx = p2->vx - p1->vx;
    float dvy = p2->vy - p1->vy;
    float dvn = dvx * nx + dvy * ny;
    
    if (dvn >= 0) return;
    
    // Calculate impulse
    float impulse = -(1.0f + restitution) * dvn / (1.0f / p1->mass + 1.0f / p2->mass);
    float impulse_x = impulse * nx;
    float impulse_y = impulse * ny;
    
    p1->vx -= impulse_x / p1->mass;
    p1->vy -= impulse_y / p1->mass;
    p2->vx += impulse_x / p2->mass;
    p2->vy += impulse_y / p2->mass;
}

// ============================================================================
// Kernel: Update Particles (Simple)
// ============================================================================

__global__ void update_particles_simple(Particle *particles, int count, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle *p = &particles[idx];
    if (!p->active) return;
    
    // Apply friction
    if (d_friction > 0.0f) {
        float friction_factor = 1.0f - d_friction;
        p->vx *= friction_factor;
        p->vy *= friction_factor;
    }
    
    // Apply mouse force
    if (d_mouse_pressed) {
        apply_mouse_force_gpu(p, d_mouse_x, d_mouse_y, d_mouse_force, d_mouse_attract);
    }
    
    // Update position
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    
    // Wall collisions
    resolve_wall_collision_gpu(p, d_friction);
    
    // Limit velocity
    float speed_sq = p->vx * p->vx + p->vy * p->vy;
    constexpr float max_velocity = MAX_VELOCITY;
    if (speed_sq > max_velocity * max_velocity) {
        float speed = sqrtf(speed_sq);
        float scale = max_velocity / speed;
        p->vx *= scale;
        p->vy *= scale;
    }
}

__global__ void detect_collisions_simple(Particle *particles, int count, float restitution) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle *p1 = &particles[idx];
    if (!p1->active) return;
    
    // Check against all other particles (brute force)
    for (int j = idx + 1; j < count; j++) {
        Particle *p2 = &particles[j];
        if (!p2->active) continue;
        
        resolve_particle_collision_gpu(p1, p2, restitution);
    }
}

// ============================================================================
// Kernel: Build Spatial Grid
// ============================================================================

__global__ void build_spatial_grid(Particle *particles, int count,
                                   int *grid_indices, int *grid_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle *p = &particles[idx];
    if (!p->active) return;
    
    int grid_x = static_cast<int>(p->x / GRID_CELL_SIZE);
    int grid_y = static_cast<int>(p->y / GRID_CELL_SIZE);
    
    if (grid_x >= 0 && grid_x < GRID_WIDTH && grid_y >= 0 && grid_y < GRID_HEIGHT) {
        int cell_idx = grid_y * GRID_WIDTH + grid_x;
        int pos = atomicAdd(&grid_counts[cell_idx], 1);
        grid_indices[cell_idx * count + pos] = idx;
    }
}

// ============================================================================
// Kernel: Complex Collision Detection with Spatial Grid
// ============================================================================

__global__ void detect_collisions_complex(Particle *particles, int count,
                                         int *grid_indices, int *grid_starts,
                                         int *grid_counts, float restitution) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle *p1 = &particles[idx];
    if (!p1->active) return;
    
    // Determine grid range to check
    int min_grid_x = static_cast<int>((p1->x - p1->radius * 4.0f) / GRID_CELL_SIZE);
    int max_grid_x = static_cast<int>((p1->x + p1->radius * 4.0f) / GRID_CELL_SIZE);
    int min_grid_y = static_cast<int>((p1->y - p1->radius * 4.0f) / GRID_CELL_SIZE);
    int max_grid_y = static_cast<int>((p1->y + p1->radius * 4.0f) / GRID_CELL_SIZE);
    
    // Clamp to grid bounds
    if (min_grid_x < 0) min_grid_x = 0;
    if (max_grid_x >= GRID_WIDTH) max_grid_x = GRID_WIDTH - 1;
    if (min_grid_y < 0) min_grid_y = 0;
    if (max_grid_y >= GRID_HEIGHT) max_grid_y = GRID_HEIGHT - 1;
    
    // Check nearby cells
    for (int gy = min_grid_y; gy <= max_grid_y; gy++) {
        for (int gx = min_grid_x; gx <= max_grid_x; gx++) {
            int cell_idx = gy * GRID_WIDTH + gx;
            int start = grid_starts[cell_idx];
            int count_in_cell = grid_counts[cell_idx];
            
            for (int i = 0; i < count_in_cell; i++) {
                int j = grid_indices[start + i];
                if (j <= idx) continue;
                
                Particle *p2 = &particles[j];
                if (!p2->active) continue;
                
                resolve_particle_collision_gpu(p1, p2, restitution);
            }
        }
    }
}

// ============================================================================
// Kernel: Optimized Update with Shared Memory
// ============================================================================

__global__ void update_particles_optimized(Particle *particles, int count, float dt) {
    __shared__ Particle s_particles[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // Load particle into shared memory
    if (idx < count && local_idx < 256) {
        s_particles[local_idx] = particles[idx];
    }
    __syncthreads();
    
    if (idx >= count) return;
    
    Particle *p = &s_particles[local_idx];
    if (!p->active) return;
    
    // Apply friction
    if (d_friction > 0.0f) {
        float friction_factor = 1.0f - d_friction;
        p->vx *= friction_factor;
        p->vy *= friction_factor;
    }
    
    // Apply mouse force
    if (d_mouse_pressed) {
        apply_mouse_force_gpu(p, d_mouse_x, d_mouse_y, d_mouse_force, d_mouse_attract);
    }
    
    // Update position
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    
    // Wall collisions
    resolve_wall_collision_gpu(p, d_friction);
    
    // Limit velocity
    float speed_sq = p->vx * p->vx + p->vy * p->vy;
    constexpr float max_velocity = MAX_VELOCITY;
    if (speed_sq > max_velocity * max_velocity) {
        float speed = sqrtf(speed_sq);
        float scale = max_velocity / speed;
        p->vx *= scale;
        p->vy *= scale;
    }
    
    // Write back to global memory
    __syncthreads();
    particles[idx] = *p;
}

// ============================================================================
// Host Functions - GPU Simple Mode
// ============================================================================

extern "C" void update_physics_gpu_simple_cuda(Simulation* sim, float dt) {
    if (!gpu_initialized) {
        init_gpu_memory(sim->get_max_particles());
    }
    
    // Get simulation parameters
    float friction = sim->get_friction();
    float restitution = sim->get_restitution();
    float mouse_force = sim->get_mouse_force();
    int mouse_x = sim->get_mouse_x();
    int mouse_y = sim->get_mouse_y();
    bool mouse_pressed = sim->is_mouse_pressed();
    bool mouse_attract = sim->is_mouse_attract();
    
    // Update device constants
    CUDA_CHECK(cudaMemcpyToSymbol(d_friction, &friction, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_restitution, &restitution, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_force, &mouse_force, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_x, &mouse_x, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_y, &mouse_y, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_pressed, &mouse_pressed, sizeof(bool)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_attract, &mouse_attract, sizeof(bool)));
    
    // Copy particles to device
    int count = sim->get_particle_count();
    CUDA_CHECK(cudaMemcpy(d_particles, sim->get_particle_data(),
                         count * sizeof(Particle),
                         cudaMemcpyHostToDevice));
    
    // Launch kernels
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    
    update_particles_simple<<<blocks, threads>>>(d_particles, count, dt);
    CUDA_CHECK(cudaGetLastError());
    
    detect_collisions_simple<<<blocks, threads>>>(d_particles, count, restitution);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy particles back to host
    CUDA_CHECK(cudaMemcpy(sim->get_particle_data(), d_particles,
                         count * sizeof(Particle),
                         cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Host Functions - GPU Complex Mode
// ============================================================================

extern "C" void update_physics_gpu_complex_cuda(Simulation* sim, float dt) {
    if (!gpu_initialized) {
        init_gpu_memory(sim->get_max_particles());
    }
    
    // Get simulation parameters
    float friction = sim->get_friction();
    float restitution = sim->get_restitution();
    float mouse_force = sim->get_mouse_force();
    int mouse_x = sim->get_mouse_x();
    int mouse_y = sim->get_mouse_y();
    bool mouse_pressed = sim->is_mouse_pressed();
    bool mouse_attract = sim->is_mouse_attract();
    
    // Update device constants
    CUDA_CHECK(cudaMemcpyToSymbol(d_friction, &friction, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_restitution, &restitution, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_force, &mouse_force, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_x, &mouse_x, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_y, &mouse_y, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_pressed, &mouse_pressed, sizeof(bool)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_attract, &mouse_attract, sizeof(bool)));
    
    // Copy particles to device
    int count = sim->get_particle_count();
    CUDA_CHECK(cudaMemcpy(d_particles, sim->get_particle_data(),
                         count * sizeof(Particle),
                         cudaMemcpyHostToDevice));
    
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    
    // Update particles with optimized kernel
    update_particles_optimized<<<blocks, threads>>>(d_particles, count, dt);
    CUDA_CHECK(cudaGetLastError());
    
    // Build spatial grid
    int grid_size = GRID_WIDTH * GRID_HEIGHT;
    CUDA_CHECK(cudaMemset(d_grid_counts, 0, grid_size * sizeof(int)));
    
    build_spatial_grid<<<blocks, threads>>>(d_particles, count,
                                           d_grid_indices, d_grid_counts);
    CUDA_CHECK(cudaGetLastError());
    
    // Compute prefix sum for grid starts (on CPU for simplicity)
    std::vector<int> h_grid_counts(grid_size);
    std::vector<int> h_grid_starts(grid_size);
    
    CUDA_CHECK(cudaMemcpy(h_grid_counts.data(), d_grid_counts,
                         grid_size * sizeof(int), cudaMemcpyDeviceToHost));
    
    h_grid_starts[0] = 0;
    for (int i = 1; i < grid_size; i++) {
        h_grid_starts[i] = h_grid_starts[i-1] + h_grid_counts[i-1];
    }
    
    CUDA_CHECK(cudaMemcpy(d_grid_starts, h_grid_starts.data(),
                         grid_size * sizeof(int), cudaMemcpyHostToDevice));
    
    // Collision detection with spatial grid
    detect_collisions_complex<<<blocks, threads>>>(d_particles, count,
                                                   d_grid_indices, d_grid_starts,
                                                   d_grid_counts, restitution);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy particles back to host
    CUDA_CHECK(cudaMemcpy(sim->get_particle_data(), d_particles,
                         count * sizeof(Particle),
                         cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}
